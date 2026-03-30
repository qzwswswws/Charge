"""
Clean baseline entry for BCI Competition IV 2b.

This script adapts the current MI_Algorithm_Workbench training style to the
3-channel / 2-class BCI IV 2b setting, with configurable data root and a
lightweight readiness check mode.

Expected preprocessed files under BCI2B_DATA_ROOT:
    B0{subject}01T.mat
    B0{subject}02T.mat
    B0{subject}03T.mat
    B0{subject}04E.mat
    B0{subject}05E.mat

Each .mat file should contain:
    data  -> shape roughly (1000, 3, n_trials)
    label -> shape roughly (n_trials, 1) or (1, n_trials)

Usage:
    python conformer_2b_baseline.py --subject 1 --check_only
    python conformer_2b_baseline.py --subject 1 --epochs 250
"""

from __future__ import annotations

import argparse
import datetime
import os
import random
import time
from pathlib import Path

import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size: int = 40, n_channels: int = 3):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (n_channels, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 20)),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange("b e h w -> b (h w) e"),
        )

    def forward(self, x):
        x = self.shallownet(x)
        return self.projection(x)


class LocalAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float, window_size: int):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask=None):
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)

        b, h, n, d = queries.shape
        local_mask = torch.ones(n, n, device=x.device)
        for i in range(n):
            local_mask[i, max(0, i - self.window_size):min(n, i + self.window_size + 1)] = 0
        local_mask = local_mask.bool().unsqueeze(0).unsqueeze(0)

        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        energy.masked_fill_(local_mask, torch.finfo(torch.float32).min)

        scaling = self.emb_size ** 0.5
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhal, bhlv -> bhav", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.projection(out)


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        return x + res


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int, drop_p: float):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        emb_size: int,
        window_size: int,
        num_heads: int = 5,
        drop_p: float = 0.5,
        forward_expansion: int = 4,
        forward_drop_p: float = 0.5,
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    LocalAttention(emb_size, num_heads, drop_p, window_size),
                    nn.Dropout(drop_p),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                    nn.Dropout(drop_p),
                )
            ),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int, emb_size: int, window_size: int):
        super().__init__(*[TransformerEncoderBlock(emb_size, window_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int, n_classes: int):
        super().__init__()
        n_tokens = (1000 - 25 + 1 - 75) // 20 + 1
        fc_input = n_tokens * emb_size
        self.fc = nn.Sequential(
            nn.Linear(fc_input, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out


class Conformer2B(nn.Sequential):
    def __init__(self, emb_size: int = 40, depth: int = 6, n_classes: int = 2, n_channels: int = 3, window_size: int = 8):
        super().__init__(
            PatchEmbedding(emb_size, n_channels=n_channels),
            TransformerEncoder(depth, emb_size, window_size),
            ClassificationHead(emb_size, n_classes),
        )


class ExP2B:
    def __init__(self, nsub: int, window_size: int, data_root: str):
        self.nSub = nsub
        self.window_size = window_size
        self.root = Path(data_root)
        self.batch_size = 100
        self.n_epochs = 250
        self.n_channels = 3
        self.n_classes = 2
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Conformer2B(
            n_channels=self.n_channels,
            n_classes=self.n_classes,
            window_size=window_size,
        ).to(self.device)
        self.criterion_cls = torch.nn.CrossEntropyLoss().to(self.device)

        self.experiment_name = (
            f"b2_subject_{self.nSub}_ch3_cls2_"
            f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.log_dir = f"./logs2b.{window_size}/{self.experiment_name}"
        self.model_dir = f"./models/{self.experiment_name}"
        self.result_dir = f"./results/{self.experiment_name}"
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

        self.log_write = open(f"./results/log_2b_subject{self.nSub}_ch3_cls2.txt", "w")
        self.train_log_path = f"{self.log_dir}/train_log.csv"
        self.test_log_path = f"{self.log_dir}/test_log.csv"
        with open(self.train_log_path, "w", encoding="utf-8") as f:
            f.write("epoch,batch,loss,lr\n")
        with open(self.test_log_path, "w", encoding="utf-8") as f:
            f.write("epoch,loss,accuracy,time\n")
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def expected_files(self):
        train_files = [self.root / f"B0{self.nSub}0{i}T.mat" for i in (1, 2, 3)]
        test_files = [self.root / f"B0{self.nSub}0{i}E.mat" for i in (4, 5)]
        return train_files, test_files

    def check_data_ready(self):
        train_files, test_files = self.expected_files()
        existing = [str(p) for p in train_files + test_files if p.exists()]
        missing = [str(p) for p in train_files + test_files if not p.exists()]
        return existing, missing

    def interaug(self, timg, label):
        aug_data = []
        aug_label = []
        for cls4aug in range(self.n_classes):
            cls_idx = np.where(label == cls4aug + 1)
            tmp_data = timg[cls_idx]
            n_per_class = int(self.batch_size / self.n_classes)
            tmp_aug_data = np.zeros((n_per_class, 1, self.n_channels, 1000))
            for ri in range(n_per_class):
                for rj in range(8):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                    tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = (
                        tmp_data[rand_idx[rj], :, :, rj * 125:(rj + 1) * 125]
                    )
            aug_data.append(tmp_aug_data)
            aug_label.append(np.full(n_per_class, cls4aug + 1, dtype=label.dtype))

        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data).to(self.device).float()
        aug_label = torch.from_numpy(aug_label - 1).to(self.device).long()
        return aug_data, aug_label

    @staticmethod
    def _load_mat_trialwise(path: Path):
        data_obj = scipy.io.loadmat(path)
        data = data_obj["data"]
        label = data_obj["label"]
        data = np.transpose(data, (2, 1, 0))
        data = np.expand_dims(data, axis=1)
        label = np.transpose(label)[0]
        return data, label

    def get_source_data(self):
        train_files, test_files = self.expected_files()
        train_data = []
        train_label = []
        for path in train_files:
            data_tmp, label_tmp = self._load_mat_trialwise(path)
            train_data.append(data_tmp)
            train_label.append(label_tmp)

        self.allData = np.concatenate(train_data)
        self.allLabel = np.concatenate(train_label)

        shuffle_num = np.random.permutation(len(self.allData))
        self.allData = self.allData[shuffle_num]
        self.allLabel = self.allLabel[shuffle_num]

        test_data = []
        test_label = []
        for path in test_files:
            data_tmp, label_tmp = self._load_mat_trialwise(path)
            test_data.append(data_tmp)
            test_label.append(label_tmp)

        self.testData = np.concatenate(test_data)
        self.testLabel = np.concatenate(test_label)

        target_mean = np.mean(self.allData)
        target_std = np.std(self.allData)
        self.allData = (self.allData - target_mean) / target_std
        self.testData = (self.testData - target_mean) / target_std
        return self.allData, self.allLabel, self.testData, self.testLabel

    def train(self):
        img, label, test_data, test_label = self.get_source_data()

        img = torch.from_numpy(img)
        label = torch.from_numpy(label - 1)
        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label - 1)
        test_data = Variable(test_data.to(self.device).float())
        test_label = Variable(test_label.to(self.device).long())

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        bestAcc = 0.0
        averAcc = 0.0
        num = 0
        Y_true = 0
        Y_pred = 0

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

                _, outputs = self.model(img)
                loss = self.criterion_cls(outputs, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                with open(self.train_log_path, "a", encoding="utf-8") as f:
                    f.write(f"{e},{i},{loss},{self.lr}\n")
                global_step = e * total_step + i
                self.writer.add_scalar("Train/Loss_Batch", loss, global_step)

            self.model.eval()
            _, cls_out = self.model(test_data)
            test_loss = self.criterion_cls(cls_out, test_label)
            y_pred = torch.max(cls_out, 1)[1]
            test_acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
            train_pred = torch.max(outputs, 1)[1]
            train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))

            with open(self.test_log_path, "a", encoding="utf-8") as f:
                f.write(f"{e},{test_loss},{test_acc},{time.time()-epoch_start_time}\n")

            self.writer.add_scalar("Train/Loss_Epoch", loss, e)
            self.writer.add_scalar("Train/Accuracy_Epoch", train_acc, e)
            self.writer.add_scalar("Test/Loss", test_loss, e)
            self.writer.add_scalar("Test/Accuracy", test_acc, e)

            if test_acc > bestAcc:
                bestAcc = test_acc
                Y_true = test_label
                Y_pred = y_pred
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
            averAcc += test_acc

        averAcc = averAcc / num
        print("The average accuracy is:", averAcc)
        print("The best accuracy is:", bestAcc)
        self.log_write.write("The average accuracy is: " + str(averAcc) + "\n")
        self.log_write.write("The best accuracy is: " + str(bestAcc) + "\n")
        self.writer.close()
        self.log_write.close()
        return bestAcc, averAcc, Y_true, Y_pred


def parse_args():
    parser = argparse.ArgumentParser(description="BCI IV 2b baseline")
    parser.add_argument("--subject", type=int, required=True, help="被试编号 1-9")
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.environ.get(
            "BCI2B_DATA_ROOT",
            "/home/woqiu/下载/git/MI_Algorithm_Workbench/datasets/standard_2b_strict_TE/",
        ),
        help="2b 预处理 mat 文件目录",
    )
    parser.add_argument("--check_only", action="store_true", help="只检查数据文件是否齐全")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    os.makedirs("./results", exist_ok=True)

    exp = ExP2B(nsub=args.subject, window_size=args.window_size, data_root=args.data_root)
    exp.n_epochs = args.epochs

    existing, missing = exp.check_data_ready()
    print(f"\n{'='*60}")
    print("  BCI IV 2b baseline")
    print(f"  Subject:   {args.subject}")
    print(f"  DataRoot:  {args.data_root}")
    print(f"  Window:    {args.window_size}")
    print(f"  Epochs:    {args.epochs}")
    print(f"  Seed:      {args.seed}")
    print(f"  Device:    {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"  Found:     {len(existing)}")
    print(f"  Missing:   {len(missing)}")
    if existing:
        print("  Existing files:")
        for path in existing:
            print(f"    - {path}")
    if missing:
        print("  Missing files:")
        for path in missing:
            print(f"    - {path}")
    print(f"{'='*60}\n")

    if args.check_only:
        return

    if missing:
        raise FileNotFoundError("2b 数据文件不完整，无法启动训练。请先补齐预处理 mat 文件。")

    starttime = datetime.datetime.now()
    bestAcc, averAcc, _, _ = exp.train()
    endtime = datetime.datetime.now()
    duration = endtime - starttime

    print(f"\n  完成! Best={bestAcc:.4f} ({bestAcc*100:.2f}%)  Aver={averAcc:.4f}  耗时={duration}")
    print(f"\nRESULT_CSV: {args.subject},b2_baseline,3,2,{args.window_size},{args.seed},{args.epochs},{bestAcc:.6f},{averAcc:.6f},{duration}")


if __name__ == "__main__":
    main()
