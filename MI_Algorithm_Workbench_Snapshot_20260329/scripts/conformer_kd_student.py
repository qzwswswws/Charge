"""
Minimal knowledge distillation experiment:
22-channel teacher -> low-channel student on BCI IV 2a.

Design principles:
1. Reuse the existing Conformer backbone from conformer_degradation.py.
2. Keep the student as the original baseline structure.
3. Use only CE + KL distillation for the first pilot.
4. Focus on 4-class distillation; do not mix in extra structural changes.

Example:
    python conformer_kd_student.py --subject 1 --channel_config c3c4 --epochs 250
"""

from __future__ import annotations

import argparse
import datetime
import glob
import os
import random
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.backends import cudnn

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass

from conformer_degradation import BCI2A_CHANNELS, CHANNEL_PRESETS, Conformer

cudnn.benchmark = False
cudnn.deterministic = True

FULL_CLASSES = [1, 2, 3, 4]


def resolve_teacher_checkpoint(models_dir: Path, subject: int) -> Path:
    """Prefer the trusted original full-channel baseline, then fall back to a 22ch degradation run."""
    preferred = []
    pattern = re.compile(rf"^subject_{subject}_[0-9]{{8}}_[0-9]{{6}}$")
    for run_dir in models_dir.glob(f"subject_{subject}_*"):
        if run_dir.is_dir() and pattern.match(run_dir.name):
            ckpt = run_dir / "best_model.pth"
            if ckpt.exists():
                preferred.append(ckpt)

    if preferred:
        return sorted(preferred, reverse=True)[0]

    fallback = sorted(
        models_dir.glob(f"subject_{subject}_ch22_cls4_*/best_model.pth"),
        reverse=True,
    )
    if fallback:
        return fallback[0]

    raise FileNotFoundError(
        f"No 22-channel teacher checkpoint found for subject {subject} under {models_dir}"
    )


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class ExperimentKD:
    def __init__(
        self,
        subject: int,
        student_channel_indices: List[int],
        window_size: int,
        teacher_window_size: int,
        temperature: float,
        alpha: float,
        teacher_ckpt: Optional[str] = None,
    ):
        super().__init__()
        self.subject = subject
        self.student_channel_indices = student_channel_indices
        self.n_student_channels = len(student_channel_indices)
        self.selected_classes = FULL_CLASSES
        self.n_classes = len(self.selected_classes)
        self.window_size = window_size
        self.teacher_window_size = teacher_window_size
        self.temperature = temperature
        self.alpha = alpha
        self.root = os.environ.get("BCI2A_DATA_ROOT", "/home/woqiu/下载/standard_2a_data/")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 72
        while self.batch_size % self.n_classes != 0:
            self.batch_size -= 1

        self.n_epochs = 250
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999

        script_dir = Path(__file__).resolve().parent
        models_dir = script_dir / "models"
        self.teacher_ckpt = Path(teacher_ckpt).resolve() if teacher_ckpt else resolve_teacher_checkpoint(models_dir, subject)

        self.student = Conformer(
            n_channels=self.n_student_channels,
            n_classes=self.n_classes,
            window_size=self.window_size,
        ).to(self.device)
        self.teacher = Conformer(
            n_channels=22,
            n_classes=self.n_classes,
            window_size=self.teacher_window_size,
        ).to(self.device)

        self._load_teacher()
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.criterion_cls = nn.CrossEntropyLoss().to(self.device)
        self.criterion_kd = nn.KLDivLoss(reduction="batchmean").to(self.device)

        ch_tag = f"ch{self.n_student_channels}"
        cls_tag = f"cls{self.n_classes}"
        self.model_tag = "kdstudent"
        self.experiment_name = (
            f"{self.model_tag}_subject_{self.subject}_{ch_tag}_{cls_tag}_"
            f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        self.log_dir = script_dir / f"logsKD.{self.window_size}" / self.experiment_name
        self.model_dir = script_dir / "models" / self.experiment_name
        self.result_dir = script_dir / "results" / self.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)

        ch_file_tag = f"ch{self.n_student_channels}"
        self.log_write = open(script_dir / "results" / f"log_{self.model_tag}_subject{self.subject}_{ch_file_tag}_{cls_tag}.txt", "w")
        self.train_log_path = self.log_dir / "train_log.csv"
        self.test_log_path = self.log_dir / "test_log.csv"
        self.train_log_path.write_text("epoch,batch,ce_loss,kd_loss,total_loss,lr\n", encoding="utf-8")
        self.test_log_path.write_text("epoch,loss,accuracy,time\n", encoding="utf-8")
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def _load_teacher(self) -> None:
        state = safe_torch_load(self.teacher_ckpt)
        missing, unexpected = self.teacher.load_state_dict(state, strict=False)
        allowed_unexpected = {
            "2.clshead.1.weight",
            "2.clshead.1.bias",
            "2.clshead.2.weight",
            "2.clshead.2.bias",
        }
        if missing:
            raise RuntimeError(f"Teacher checkpoint missing keys: {missing[:5]}")
        unexpected_set = set(unexpected)
        if unexpected_set - allowed_unexpected:
            raise RuntimeError(f"Teacher checkpoint has unsupported unexpected keys: {sorted(unexpected_set - allowed_unexpected)}")

    def _load_raw_subject_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train_mat = scipy.io.loadmat(os.path.join(self.root, f"A0{self.subject}T.mat"))
        test_mat = scipy.io.loadmat(os.path.join(self.root, f"A0{self.subject}E.mat"))

        train_full = np.transpose(train_mat["data"], (2, 1, 0))
        train_full = np.expand_dims(train_full, axis=1)
        train_label = np.transpose(train_mat["label"])[0]

        test_full = np.transpose(test_mat["data"], (2, 1, 0))
        test_full = np.expand_dims(test_full, axis=1)
        test_label = np.transpose(test_mat["label"])[0]
        return train_full, train_label, test_full, test_label

    def get_source_data(self):
        train_full_raw, train_label, test_full_raw, test_label = self._load_raw_subject_data()

        train_student_raw = train_full_raw[:, :, self.student_channel_indices, :]
        test_student_raw = test_full_raw[:, :, self.student_channel_indices, :]

        shuffle_num = np.random.permutation(len(train_full_raw))
        train_full_raw = train_full_raw[shuffle_num]
        train_student_raw = train_student_raw[shuffle_num]
        train_label = train_label[shuffle_num]

        full_mean = np.mean(train_full_raw)
        full_std = np.std(train_full_raw)
        student_mean = np.mean(train_student_raw)
        student_std = np.std(train_student_raw)

        train_full = (train_full_raw - full_mean) / full_std
        test_full = (test_full_raw - full_mean) / full_std
        train_student = (train_student_raw - student_mean) / student_std
        test_student = (test_student_raw - student_mean) / student_std

        self.train_full = train_full
        self.train_student = train_student
        self.train_label = train_label
        self.test_full = test_full
        self.test_student = test_student
        self.test_label = test_label
        return train_full, train_student, train_label, test_full, test_student, test_label

    def interaug_pair(self, full_data, student_data, label):
        aug_full = []
        aug_student = []
        aug_label = []
        n_per_class = int(self.batch_size / self.n_classes)

        for cls_idx in range(self.n_classes):
            cls_mask = np.where(label == cls_idx + 1)
            tmp_full = full_data[cls_mask]
            tmp_student = student_data[cls_mask]
            tmp_label = label[cls_mask]

            tmp_aug_full = np.zeros((n_per_class, 1, 22, 1000))
            tmp_aug_student = np.zeros((n_per_class, 1, self.n_student_channels, 1000))

            for ri in range(n_per_class):
                rand_idx = np.random.randint(0, tmp_full.shape[0], 8)
                for rj in range(8):
                    start = rj * 125
                    end = (rj + 1) * 125
                    idx = rand_idx[rj]
                    tmp_aug_full[ri, :, :, start:end] = tmp_full[idx, :, :, start:end]
                    tmp_aug_student[ri, :, :, start:end] = tmp_student[idx, :, :, start:end]

            aug_full.append(tmp_aug_full)
            aug_student.append(tmp_aug_student)
            aug_label.append(tmp_label[:n_per_class])

        aug_full = np.concatenate(aug_full)
        aug_student = np.concatenate(aug_student)
        aug_label = np.concatenate(aug_label)
        shuffle_num = np.random.permutation(len(aug_full))
        aug_full = aug_full[shuffle_num]
        aug_student = aug_student[shuffle_num]
        aug_label = aug_label[shuffle_num]

        aug_full = torch.from_numpy(aug_full).float().to(self.device)
        aug_student = torch.from_numpy(aug_student).float().to(self.device)
        aug_label = torch.from_numpy(aug_label - 1).long().to(self.device)
        return aug_full, aug_student, aug_label

    def kd_loss(self, student_logits, teacher_logits):
        t = self.temperature
        student_log_prob = F.log_softmax(student_logits / t, dim=1)
        teacher_prob = F.softmax(teacher_logits / t, dim=1)
        return self.criterion_kd(student_log_prob, teacher_prob) * (t * t)

    def evaluate_teacher(self, test_full_tensor, test_label_tensor) -> float:
        self.teacher.eval()
        with torch.no_grad():
            _, teacher_logits = self.teacher(test_full_tensor)
            y_pred = torch.max(teacher_logits, 1)[1]
            acc = float((y_pred == test_label_tensor).sum().item()) / float(test_label_tensor.size(0))
        return acc

    def train(self):
        train_full, train_student, train_label, test_full, test_student, test_label = self.get_source_data()

        train_full_tensor = torch.from_numpy(train_full).float()
        train_student_tensor = torch.from_numpy(train_student).float()
        train_label_tensor = torch.from_numpy(train_label - 1).long()
        dataset = torch.utils.data.TensorDataset(train_full_tensor, train_student_tensor, train_label_tensor)
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        test_full_tensor = torch.from_numpy(test_full).float().to(self.device)
        test_student_tensor = torch.from_numpy(test_student).float().to(self.device)
        test_label_tensor = torch.from_numpy(test_label - 1).long().to(self.device)

        teacher_acc = self.evaluate_teacher(test_full_tensor, test_label_tensor)
        self.log_write.write(f"Teacher checkpoint: {self.teacher_ckpt}\n")
        self.log_write.write(f"Teacher test accuracy: {teacher_acc}\n")

        optimizer = torch.optim.Adam(self.student.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        best_acc = 0.0
        aver_acc = 0.0
        eval_count = 0
        y_true_best = None
        y_pred_best = None
        total_step = len(dataloader)

        for epoch in range(self.n_epochs):
            epoch_start = time.time()
            self.student.train()

            for batch_idx, (batch_full, batch_student, batch_label) in enumerate(dataloader):
                batch_full = batch_full.to(self.device)
                batch_student = batch_student.to(self.device)
                batch_label = batch_label.to(self.device)

                aug_full, aug_student, aug_label = self.interaug_pair(self.train_full, self.train_student, self.train_label)
                teacher_input = torch.cat((batch_full, aug_full), dim=0)
                student_input = torch.cat((batch_student, aug_student), dim=0)
                labels = torch.cat((batch_label, aug_label), dim=0)

                with torch.no_grad():
                    _, teacher_logits = self.teacher(teacher_input)
                _, student_logits = self.student(student_input)

                loss_ce = self.criterion_cls(student_logits, labels)
                loss_kd = self.kd_loss(student_logits, teacher_logits)
                loss = self.alpha * loss_ce + (1.0 - self.alpha) * loss_kd

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with self.train_log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        f"{epoch},{batch_idx},{loss_ce.item():.8f},{loss_kd.item():.8f},"
                        f"{loss.item():.8f},{self.lr}\n"
                    )

                global_step = epoch * total_step + batch_idx
                self.writer.add_scalar("Train/CE_Loss_Batch", loss_ce.item(), global_step)
                self.writer.add_scalar("Train/KD_Loss_Batch", loss_kd.item(), global_step)
                self.writer.add_scalar("Train/Total_Loss_Batch", loss.item(), global_step)

            self.student.eval()
            with torch.no_grad():
                _, student_test_logits = self.student(test_student_tensor)
                test_loss = self.criterion_cls(student_test_logits, test_label_tensor)
                y_pred = torch.max(student_test_logits, 1)[1]
                test_acc = float((y_pred == test_label_tensor).sum().item()) / float(test_label_tensor.size(0))
                train_pred = torch.max(student_logits, 1)[1]
                train_acc = float((train_pred == labels).sum().item()) / float(labels.size(0))

            with self.test_log_path.open("a", encoding="utf-8") as f:
                f.write(f"{epoch},{test_loss.item():.8f},{test_acc:.8f},{time.time() - epoch_start:.4f}\n")

            self.writer.add_scalar("Train/Accuracy_Epoch", train_acc, epoch)
            self.writer.add_scalar("Test/Loss", test_loss.item(), epoch)
            self.writer.add_scalar("Test/Accuracy", test_acc, epoch)
            self.writer.add_scalar("Teacher/Test_Accuracy", teacher_acc, epoch)

            if test_acc > best_acc:
                best_acc = test_acc
                y_true_best = test_label_tensor.detach().cpu().numpy()
                y_pred_best = y_pred.detach().cpu().numpy()
                torch.save(self.student.state_dict(), self.model_dir / "best_model.pth")
                conf_matrix = confusion_matrix(y_true_best, y_pred_best)
                np.save(self.result_dir / "best_confusion_matrix.npy", conf_matrix)

            print(
                "Epoch:", epoch,
                "  CE loss: %.6f" % loss_ce.item(),
                "  KD loss: %.6f" % loss_kd.item(),
                "  Test loss: %.6f" % test_loss.item(),
                "  Train accuracy %.6f" % train_acc,
                "  Test accuracy is %.6f" % test_acc,
            )

            self.log_write.write(f"{epoch}    {test_acc}\n")
            eval_count += 1
            aver_acc += test_acc

        aver_acc = aver_acc / max(1, eval_count)
        self.log_write.write(f"The average accuracy is: {aver_acc}\n")
        self.log_write.write(f"The best accuracy is: {best_acc}\n")
        self.log_write.write(f"Teacher accuracy is: {teacher_acc}\n")
        self.writer.close()
        self.log_write.close()
        return best_acc, aver_acc, teacher_acc, y_true_best, y_pred_best


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal KD student training")
    parser.add_argument("--subject", type=int, required=True, help="Subject id 1-9")
    parser.add_argument(
        "--channel_config",
        type=str,
        default="c3c4",
        choices=list(CHANNEL_PRESETS.keys()),
        help="Student channel preset",
    )
    parser.add_argument("--channels", type=str, default=None, help="Custom student channels, comma-separated 0-based")
    parser.add_argument("--classes", type=str, default="1,2,3,4", help="Currently only 4-class KD is supported")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--window_size", type=int, default=8, help="Student local attention window size")
    parser.add_argument("--teacher_window_size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.5, help="CE weight in alpha*CE + (1-alpha)*KD")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher_ckpt", type=str, default=None, help="Optional explicit teacher checkpoint path")
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)
    os.makedirs("./results", exist_ok=True)

    selected_classes = [int(x.strip()) for x in args.classes.split(",")]
    if selected_classes != FULL_CLASSES:
        raise ValueError("This minimal KD script currently supports only 4-class distillation: --classes 1,2,3,4")

    if args.channels:
        student_channel_indices = [int(x.strip()) for x in args.channels.split(",")]
        config_name = f"custom{len(student_channel_indices)}"
    else:
        student_channel_indices = CHANNEL_PRESETS[args.channel_config]
        config_name = args.channel_config

    seed_everything(args.seed)
    starttime = datetime.datetime.now()
    channel_names = [BCI2A_CHANNELS[i] for i in student_channel_indices]
    print(f"\n{'=' * 72}")
    print("  Minimal KD experiment")
    print(f"  Subject:        {args.subject}")
    print(f"  Student config: {config_name} ({len(student_channel_indices)} channels)")
    print(f"  Channels:       {channel_names}")
    print(f"  Teacher:        22-channel 4-class checkpoint")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Temperature:    {args.temperature}")
    print(f"  Alpha(CE):      {args.alpha}")
    print(f"  Seed:           {args.seed}")
    print(f"{'=' * 72}\n")

    exp = ExperimentKD(
        subject=args.subject,
        student_channel_indices=student_channel_indices,
        window_size=args.window_size,
        teacher_window_size=args.teacher_window_size,
        temperature=args.temperature,
        alpha=args.alpha,
        teacher_ckpt=args.teacher_ckpt,
    )
    exp.n_epochs = args.epochs
    best_acc, aver_acc, teacher_acc, _, _ = exp.train()
    duration = datetime.datetime.now() - starttime

    print(f"\n  Complete! Student Best={best_acc:.4f}  Student Aver={aver_acc:.4f}  Teacher={teacher_acc:.4f}  Duration={duration}")
    print(f"TEACHER_CKPT: {exp.teacher_ckpt}")
    print(
        f"RESULT_CSV: {args.subject},{config_name},{len(student_channel_indices)},"
        f"{len(FULL_CLASSES)},{args.window_size},{args.seed},{args.epochs},"
        f"{best_acc:.6f},{aver_acc:.6f},{teacher_acc:.6f},{duration}"
    )


if __name__ == "__main__":
    main()
