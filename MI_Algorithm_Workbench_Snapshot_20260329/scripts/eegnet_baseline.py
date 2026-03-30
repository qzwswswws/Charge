"""
EEGNet baseline for MI_Algorithm_Workbench.

Goal:
1. Keep the current 2a data protocol and CLI as close as possible to
   conformer_degradation.py.
2. Replace the model body with an EEGNet-style compact CNN.
3. Support both full-channel / 4-class and low-channel / 2-class baselines.

Usage:
    python eegnet_baseline.py --channel_config full --subject 1 --epochs 250
    python eegnet_baseline.py --channel_config c3c4 --classes 1,2 --subject 1
"""

from __future__ import annotations

import argparse
import datetime
import math
import os
import random
import time

import numpy as np
import scipy.io
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torch.backends import cudnn

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:  # type: ignore[override]
        """Fallback no-op writer when tensorboard is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            pass

        def add_scalar(self, *args, **kwargs) -> None:
            pass

        def close(self) -> None:
            pass


cudnn.benchmark = False
cudnn.deterministic = True

gpus = [0]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


BCI2A_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]

CHANNEL_PRESETS = {
    "full": list(range(22)),
    "central8": [1, 5, 6, 7, 8, 10, 11, 12],
    "c3czc4": [7, 9, 11],
    "c3c4": [7, 11],
    "c3cz": [7, 9],
    "czc4": [9, 11],
    "c1czc2": [8, 9, 10],
}


class EEGNet(nn.Module):
    """PyTorch EEGNet adapted from the official arl-eegmodels architecture."""

    def __init__(
        self,
        n_classes: int,
        n_channels: int,
        n_samples: int = 1000,
        dropout_rate: float = 0.5,
        kern_length: int = 64,
        f1: int = 8,
        d: int = 2,
        f2: int = 16,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples

        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, kern_length), padding="same", bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * d, (n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(f1 * d, f1 * d, (1, 16), padding="same", groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            features = self._forward_features(dummy)
            flat_dim = features.shape[1]

        self.classifier = nn.Linear(flat_dim, n_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return x.flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._forward_features(x)
        return self.classifier(features)


class ExP:
    def __init__(self, nsub, channel_indices=None, selected_classes=None, train_fraction=1.0):
        super().__init__()

        self.channel_indices = channel_indices if channel_indices else list(range(22))
        self.n_channels = len(self.channel_indices)
        self.selected_classes = selected_classes if selected_classes else [1, 2, 3, 4]
        self.n_classes = len(self.selected_classes)
        self.train_fraction = float(train_fraction)
        self.label_remap = {c: i + 1 for i, c in enumerate(sorted(self.selected_classes))}

        self.batch_size = 72
        while self.batch_size % self.n_classes != 0:
            self.batch_size -= 1

        self.n_epochs = 250
        self.lr = 0.001
        self.b1 = 0.9
        self.b2 = 0.999
        self.nSub = nsub
        self.root = os.environ.get("BCI2A_DATA_ROOT", "/home/woqiu/下载/standard_2a_data/")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.criterion_cls = torch.nn.CrossEntropyLoss().to(self.device)
        self.model = EEGNet(
            n_classes=self.n_classes,
            n_channels=self.n_channels,
            n_samples=1000,
        ).to(self.device)

        ch_tag = f"ch{self.n_channels}"
        cls_tag = f"cls{self.n_classes}"
        frac_tag = f"frac{int(round(self.train_fraction * 100)):03d}"
        self.experiment_name = (
            f"eegnet_subject_{self.nSub}_{ch_tag}_{cls_tag}_{frac_tag}_"
            f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.log_dir = f"./logsEEGNet/{self.experiment_name}"
        self.model_dir = f"./models/{self.experiment_name}"
        self.result_dir = f"./results/{self.experiment_name}"
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

        self.log_write = open(f"./results/log_eegnet_subject{self.nSub}_{ch_tag}_{cls_tag}_{frac_tag}.txt", "w")
        self.train_log_path = f"{self.log_dir}/train_log.csv"
        self.test_log_path = f"{self.log_dir}/test_log.csv"
        with open(self.train_log_path, "w", encoding="utf-8") as f:
            f.write("epoch,batch,loss,lr\n")
        with open(self.test_log_path, "w", encoding="utf-8") as f:
            f.write("epoch,loss,accuracy,time\n")
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def interaug(self, timg, label):
        aug_data = []
        aug_label = []
        for cls_idx in range(self.n_classes):
            class_rows = np.where(label == cls_idx + 1)
            tmp_data = timg[class_rows]
            n_per_class = int(self.batch_size / self.n_classes)
            tmp_aug_data = np.zeros((n_per_class, 1, self.n_channels, 1000), dtype=np.float32)
            for ri in range(n_per_class):
                for rj in range(8):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                    tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = (
                        tmp_data[rand_idx[rj], :, :, rj * 125:(rj + 1) * 125]
                    )
            aug_data.append(tmp_aug_data)
            aug_label.append(np.full(n_per_class, cls_idx + 1, dtype=label.dtype))

        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data).to(self.device).float()
        aug_label = torch.from_numpy(aug_label - 1).to(self.device).long()
        return aug_data, aug_label

    def get_source_data(self):
        total_data = scipy.io.loadmat(self.root + "A0%dT.mat" % self.nSub)
        train_data = total_data["data"]
        train_label = total_data["label"]
        train_data = np.transpose(train_data, (2, 1, 0))
        train_data = np.expand_dims(train_data, axis=1)
        train_label = np.transpose(train_label)[0]

        test_tmp = scipy.io.loadmat(self.root + "A0%dE.mat" % self.nSub)
        test_data = test_tmp["data"]
        test_label = test_tmp["label"]
        test_data = np.transpose(test_data, (2, 1, 0))
        test_data = np.expand_dims(test_data, axis=1)
        test_label = np.transpose(test_label)[0]

        if sorted(self.selected_classes) != [1, 2, 3, 4]:
            train_mask = np.isin(train_label, self.selected_classes)
            train_data = train_data[train_mask]
            train_label = np.array([self.label_remap[l] for l in train_label[train_mask]])
            test_mask = np.isin(test_label, self.selected_classes)
            test_data = test_data[test_mask]
            test_label = np.array([self.label_remap[l] for l in test_label[test_mask]])

        if len(self.channel_indices) < 22:
            train_data = train_data[:, :, self.channel_indices, :]
            test_data = test_data[:, :, self.channel_indices, :]

        if self.train_fraction < 1.0:
            subset_indices = []
            for cls in sorted(np.unique(train_label)):
                cls_indices = np.where(train_label == cls)[0]
                np.random.shuffle(cls_indices)
                keep_n = max(1, int(round(len(cls_indices) * self.train_fraction)))
                keep_n = min(keep_n, len(cls_indices))
                subset_indices.extend(cls_indices[:keep_n].tolist())

            subset_indices = np.array(subset_indices, dtype=int)
            np.random.shuffle(subset_indices)
            train_data = train_data[subset_indices]
            train_label = train_label[subset_indices]

        self.allData = train_data
        self.allLabel = train_label
        shuffle_num = np.random.permutation(len(self.allData))
        self.allData = self.allData[shuffle_num]
        self.allLabel = self.allLabel[shuffle_num]

        self.testData = test_data
        self.testLabel = test_label

        target_mean = np.mean(self.allData)
        target_std = np.std(self.allData)
        self.allData = (self.allData - target_mean) / target_std
        self.testData = (self.testData - target_mean) / target_std
        return self.allData, self.allLabel, self.testData, self.testLabel

    def train(self):
        img, label, test_data, test_label = self.get_source_data()

        img = torch.from_numpy(img)
        label = torch.from_numpy(label - 1)
        dataset = TensorDataset(img, label)
        self.dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label - 1)
        test_data = Variable(test_data.to(self.device).float())
        test_label = Variable(test_label.to(self.device).long())

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        best_acc = 0.0
        aver_acc = 0.0
        num = 0
        y_true_best = 0
        y_pred_best = 0
        total_step = len(self.dataloader)

        for e in range(self.n_epochs):
            epoch_start_time = time.time()
            self.model.train()
            for i, (img, label) in enumerate(self.dataloader):
                img = Variable(img.to(self.device).float())
                label = Variable(label.to(self.device).long())

                aug_data, aug_label = self.interaug(self.allData, self.allLabel)
                img = torch.cat((img, aug_data))
                label = torch.cat((label, aug_label))

                outputs = self.model(img)
                loss = self.criterion_cls(outputs, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                with open(self.train_log_path, "a", encoding="utf-8") as f:
                    f.write(f"{e},{i},{loss},{self.lr}\n")

                global_step = e * total_step + i
                self.writer.add_scalar("Train/Loss_Batch", loss, global_step)

            self.model.eval()
            logits = self.model(test_data)
            test_loss = self.criterion_cls(logits, test_label)
            y_pred = torch.max(logits, 1)[1]
            test_acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
            train_pred = torch.max(outputs, 1)[1]
            train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))

            with open(self.test_log_path, "a", encoding="utf-8") as f:
                f.write(f"{e},{test_loss},{test_acc},{time.time() - epoch_start_time}\n")

            self.writer.add_scalar("Train/Loss_Epoch", loss, e)
            self.writer.add_scalar("Train/Accuracy_Epoch", train_acc, e)
            self.writer.add_scalar("Test/Loss", test_loss, e)
            self.writer.add_scalar("Test/Accuracy", test_acc, e)

            if test_acc > best_acc:
                best_acc = test_acc
                y_true_best = test_label
                y_pred_best = y_pred
                torch.save(self.model.state_dict(), f"{self.model_dir}/best_model.pth")
                conf_matrix = confusion_matrix(test_label.cpu().numpy(), y_pred.cpu().numpy())
                np.save(f"{self.result_dir}/best_confusion_matrix.npy", conf_matrix)

            print(
                "Epoch:", e,
                "  Train loss: %.6f" % loss.detach().cpu().numpy(),
                "  Test loss: %.6f" % test_loss.detach().cpu().numpy(),
                "  Train accuracy %.6f" % train_acc,
                "  Test accuracy is %.6f" % test_acc,
            )

            self.log_write.write(str(e) + "    " + str(test_acc) + "\n")
            num += 1
            aver_acc += test_acc

        aver_acc = aver_acc / num
        print("The average accuracy is:", aver_acc)
        print("The best accuracy is:", best_acc)
        self.log_write.write("The average accuracy is: " + str(aver_acc) + "\n")
        self.log_write.write("The best accuracy is: " + str(best_acc) + "\n")
        self.writer.close()
        self.log_write.close()

        return best_acc, aver_acc, y_true_best, y_pred_best


def parse_args():
    parser = argparse.ArgumentParser(description="EEGNet baseline")
    parser.add_argument("--subject", type=int, required=True, help="被试编号 1-9")
    parser.add_argument("--window_size", type=int, default=8, help="兼容现有接口，EEGNet 本体不使用该参数")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument(
        "--channel_config",
        type=str,
        default=None,
        choices=list(CHANNEL_PRESETS.keys()),
        help="预设通道配置: full/central8/c3czc4/c3c4/c3cz/czc4/c1czc2",
    )
    parser.add_argument("--channels", type=str, default=None, help="自定义通道索引(0-based), 逗号分隔, 如 7,9,11")
    parser.add_argument("--classes", type=str, default="1,2,3,4", help="类别选择(1-based), 逗号分隔, 如 1,2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_fraction", type=float, default=1.0, help="训练集保留比例 (0,1], 测试集保持不变")
    return parser.parse_args()


def main():
    args = parse_args()

    if not (0.0 < args.train_fraction <= 1.0):
        raise ValueError("--train_fraction 必须在 (0, 1] 范围内")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if args.channel_config:
        channel_indices = CHANNEL_PRESETS[args.channel_config]
        config_name = args.channel_config
    elif args.channels:
        channel_indices = [int(x.strip()) for x in args.channels.split(",")]
        config_name = f"custom{len(channel_indices)}"
    else:
        channel_indices = list(range(22))
        config_name = "full"

    selected_classes = [int(x.strip()) for x in args.classes.split(",")]
    channel_names = [BCI2A_CHANNELS[i] for i in channel_indices]

    print(f"\n{'=' * 60}")
    print("  EEGNet baseline")
    print(f"  Subject:  {args.subject}")
    print(f"  Config:   {config_name} ({len(channel_indices)} channels)")
    print(f"  Channels: {channel_names}")
    print(f"  Classes:  {selected_classes} ({len(selected_classes)}-class)")
    print(f"  TrainFrac:{args.train_fraction:.2f}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Seed:     {args.seed}")
    print(f"  Device:   {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"{'=' * 60}\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    os.makedirs("./results", exist_ok=True)

    starttime = datetime.datetime.now()
    exp = ExP(
        nsub=args.subject,
        channel_indices=channel_indices,
        selected_classes=selected_classes,
        train_fraction=args.train_fraction,
    )
    exp.n_epochs = args.epochs
    best_acc, aver_acc, _, _ = exp.train()
    endtime = datetime.datetime.now()
    duration = endtime - starttime

    print(
        f"\n  完成! Best={best_acc:.4f} ({best_acc * 100:.2f}%)  "
        f"Aver={aver_acc:.4f}  耗时={duration}"
    )
    print(
        f"\nRESULT_CSV: {args.subject},{config_name},{len(channel_indices)},"
        f"{len(selected_classes)},{args.window_size},{args.seed},{args.epochs},"
        f"{best_acc:.6f},{aver_acc:.6f},{duration}"
    )


if __name__ == "__main__":
    main()
