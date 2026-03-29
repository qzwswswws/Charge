"""
Low-channel Batch 2:
在 Batch 1 的 pooling head 基础上，只增加 2导差分分支。

原则:
1. 保留原始 PatchEmbedding / TransformerEncoder。
2. 保留 Batch 1 的 attention pooling head。
3. 仅在 2导场景下额外构造 C3-C4 差分通道。
4. 不重写前端，不引入更大的结构变量。

用法示例:
    python conformer_lowchannel_b2_diff.py --channel_config c3c4 --classes 1,2 --subject 1
"""

import argparse
import os
import random
import datetime
import time

import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
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

try:
    from einops import rearrange
except ModuleNotFoundError:
    def rearrange(*args, **kwargs):
        raise RuntimeError("einops is required for this script.")

cudnn.benchmark = False
cudnn.deterministic = True

gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

BCI2A_CHANNELS = [
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    'P1', 'Pz', 'P2', 'POz'
]

CHANNEL_PRESETS = {
    'full': list(range(22)),
    'central8': [1, 5, 6, 7, 8, 10, 11, 12],
    'c3czc4': [7, 9, 11],
    'c3c4': [7, 11],
}


class TwoChannelDiffBranch(nn.Module):
    """仅在 2导时添加差分通道，不改变其他设置。"""

    def __init__(self, enabled=True):
        super().__init__()
        self.enabled = enabled

    def forward(self, x):
        if (not self.enabled) or x.size(2) != 2:
            return x
        diff = x[:, :, 0:1, :] - x[:, :, 1:2, :]
        return torch.cat([x, diff], dim=2)


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40, effective_channels=22):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (effective_channels, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 20)),
            nn.Dropout(0.5),
        )
        self.projection = nn.Conv2d(40, 40, (1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.shallownet(x)
        x = self.projection(x)
        x = x.squeeze(2).transpose(1, 2).contiguous()
        return x


class LocalAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout, window_size):
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
        batch_size, n_tokens, _ = x.shape
        head_dim = self.emb_size // self.num_heads
        queries = self.queries(x).view(batch_size, n_tokens, self.num_heads, head_dim).permute(0, 2, 1, 3)
        keys = self.keys(x).view(batch_size, n_tokens, self.num_heads, head_dim).permute(0, 2, 1, 3)
        values = self.values(x).view(batch_size, n_tokens, self.num_heads, head_dim).permute(0, 2, 1, 3)

        _, _, n_tokens, _ = queries.shape
        local_mask = torch.ones(n_tokens, n_tokens, device=x.device)
        for i in range(n_tokens):
            local_mask[i, max(0, i - self.window_size):min(n_tokens, i + self.window_size + 1)] = 0
        local_mask = local_mask.bool().unsqueeze(0).unsqueeze(0)

        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        energy.masked_fill_(local_mask, torch.finfo(torch.float32).min)

        scaling = self.emb_size ** 0.5
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, n_tokens, self.emb_size)
        return self.projection(out)


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        residual = x
        x = self.fn(x, **kwargs)
        return x + residual


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, window_size, num_heads=5, drop_p=0.5,
                 forward_expansion=4, forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                LocalAttention(emb_size, num_heads, drop_p, window_size),
                nn.Dropout(drop_p),
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p),
            ))
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size, window_size):
        super().__init__(*[TransformerEncoderBlock(emb_size, window_size) for _ in range(depth)])


class AttentionPoolingHead(nn.Module):
    def __init__(self, emb_size, n_classes, hidden_dim=64):
        super().__init__()
        self.attn_score = nn.Linear(emb_size, 1)
        self.classifier = nn.Sequential(
            nn.Linear(emb_size, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        weights = F.softmax(self.attn_score(x), dim=1)
        pooled = torch.sum(weights * x, dim=1)
        logits = self.classifier(pooled)
        return pooled, logits


class ConformerB2(nn.Module):
    def __init__(self, emb_size=40, depth=6, n_classes=4, n_channels=22,
                 window_size=8, use_diff_branch=True):
        super().__init__()
        self.input_branch = TwoChannelDiffBranch(enabled=use_diff_branch)
        effective_channels = 3 if (use_diff_branch and n_channels == 2) else n_channels
        self.patch = PatchEmbedding(emb_size=emb_size, effective_channels=effective_channels)
        self.encoder = TransformerEncoder(depth, emb_size, window_size)
        self.head = AttentionPoolingHead(emb_size, n_classes)

    def forward(self, x):
        x = self.input_branch(x)
        x = self.patch(x)
        x = self.encoder(x)
        pooled, logits = self.head(x)
        return pooled, logits


class ExP:
    def __init__(self, nsub, window_size, channel_indices=None, selected_classes=None,
                 use_diff_branch=True):
        super().__init__()
        self.channel_indices = channel_indices if channel_indices else list(range(22))
        self.n_channels = len(self.channel_indices)
        self.selected_classes = selected_classes if selected_classes else [1, 2, 3, 4]
        self.n_classes = len(self.selected_classes)
        self.label_remap = {c: i + 1 for i, c in enumerate(sorted(self.selected_classes))}

        self.batch_size = 72
        while self.batch_size % self.n_classes != 0:
            self.batch_size -= 1

        self.n_epochs = 250
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.nSub = nsub
        self.root = os.environ.get("BCI2A_DATA_ROOT", "/home/woqiu/下载/standard_2a_data/")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.criterion_cls = torch.nn.CrossEntropyLoss().to(self.device)
        self.model = ConformerB2(
            n_channels=self.n_channels,
            n_classes=self.n_classes,
            window_size=window_size,
            use_diff_branch=use_diff_branch,
        ).to(self.device)

        ch_tag = f"ch{self.n_channels}"
        cls_tag = f"cls{self.n_classes}"
        self.model_tag = "lowchb2"
        self.experiment_name = (
            f"{self.model_tag}_subject_{self.nSub}_{ch_tag}_{cls_tag}_"
            f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.log_dir = f"./logsLowChB2/{self.experiment_name}"
        self.model_dir = f"./models/{self.experiment_name}"
        self.result_dir = f"./results/{self.experiment_name}"
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

        self.log_write = open(f"./results/log_{self.model_tag}_subject{self.nSub}_{ch_tag}_{cls_tag}.txt", "w")
        self.train_log_path = f"{self.log_dir}/train_log.csv"
        self.test_log_path = f"{self.log_dir}/test_log.csv"
        with open(self.train_log_path, 'w') as f:
            f.write("epoch,batch,loss,lr\n")
        with open(self.test_log_path, 'w') as f:
            f.write("epoch,loss,accuracy,time\n")
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def interaug(self, timg, label):
        aug_data = []
        aug_label = []
        for cls4aug in range(self.n_classes):
            cls_idx = np.where(label == cls4aug + 1)
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]

            n_per_class = int(self.batch_size / self.n_classes)
            tmp_aug_data = np.zeros((n_per_class, 1, self.n_channels, 1000))
            for ri in range(n_per_class):
                for rj in range(8):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                    tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = \
                        tmp_data[rand_idx[rj], :, :, rj * 125:(rj + 1) * 125]

            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:n_per_class])

        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data).float().to(self.device)
        aug_label = torch.from_numpy(aug_label - 1).long().to(self.device)
        return aug_data, aug_label

    def get_source_data(self):
        total_data = scipy.io.loadmat(self.root + 'A0%dT.mat' % self.nSub)
        train_data = total_data['data']
        train_label = total_data['label']
        train_data = np.transpose(train_data, (2, 1, 0))
        train_data = np.expand_dims(train_data, axis=1)
        train_label = np.transpose(train_label)[0]

        test_tmp = scipy.io.loadmat(self.root + 'A0%dE.mat' % self.nSub)
        test_data = test_tmp['data']
        test_label = test_tmp['label']
        test_data = np.transpose(test_data, (2, 1, 0))
        test_data = np.expand_dims(test_data, axis=1)
        test_label = np.transpose(test_label)[0]

        if sorted(self.selected_classes) != [1, 2, 3, 4]:
            train_mask = np.isin(train_label, self.selected_classes)
            train_data = train_data[train_mask]
            train_label = train_label[train_mask]
            train_label = np.array([self.label_remap[l] for l in train_label])

            test_mask = np.isin(test_label, self.selected_classes)
            test_data = test_data[test_mask]
            test_label = test_label[test_mask]
            test_label = np.array([self.label_remap[l] for l in test_label])

        if len(self.channel_indices) < 22:
            train_data = train_data[:, :, self.channel_indices, :]
            test_data = test_data[:, :, self.channel_indices, :]

        self.allData = train_data
        self.allLabel = train_label
        shuffle_num = np.random.permutation(len(self.allData))
        self.allData = self.allData[shuffle_num, :, :, :]
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
        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        test_data = Variable(torch.from_numpy(test_data).float().to(self.device))
        test_label = Variable(torch.from_numpy(test_label - 1).long().to(self.device))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        best_acc = 0
        aver_acc = 0
        num = 0
        y_true_best = 0
        y_pred_best = 0
        total_step = len(self.dataloader)

        for epoch in range(self.n_epochs):
            epoch_start_time = time.time()
            self.model.train()
            for batch_idx, (batch_img, batch_label) in enumerate(self.dataloader):
                batch_img = Variable(batch_img.float().to(self.device))
                batch_label = Variable(batch_label.long().to(self.device))

                aug_data, aug_label = self.interaug(self.allData, self.allLabel)
                batch_img = torch.cat((batch_img, aug_data))
                batch_label = torch.cat((batch_label, aug_label))

                _, outputs = self.model(batch_img)
                loss = self.criterion_cls(outputs, batch_label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                with open(self.train_log_path, 'a') as f:
                    f.write(f"{epoch},{batch_idx},{loss.item():.8f},{self.lr}\n")

                global_step = epoch * total_step + batch_idx
                self.writer.add_scalar('Train/Loss_Batch', loss.item(), global_step)

            self.model.eval()
            with torch.no_grad():
                _, logits = self.model(test_data)
                test_loss = self.criterion_cls(logits, test_label)
                y_pred = torch.max(logits, 1)[1]
                test_acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == batch_label).cpu().numpy().astype(int).sum()) / float(batch_label.size(0))

            with open(self.test_log_path, 'a') as f:
                f.write(f"{epoch},{test_loss.item():.8f},{test_acc:.8f},{time.time() - epoch_start_time:.4f}\n")

            self.writer.add_scalar('Train/Loss_Epoch', loss.item(), epoch)
            self.writer.add_scalar('Train/Accuracy_Epoch', train_acc, epoch)
            self.writer.add_scalar('Test/Loss', test_loss.item(), epoch)
            self.writer.add_scalar('Test/Accuracy', test_acc, epoch)

            if test_acc > best_acc:
                best_acc = test_acc
                y_true_best = test_label
                y_pred_best = y_pred
                torch.save(self.model.state_dict(), f"{self.model_dir}/best_model.pth")
                conf_matrix = confusion_matrix(test_label.cpu().numpy(), y_pred.cpu().numpy())
                np.save(f"{self.result_dir}/best_confusion_matrix.npy", conf_matrix)

            print(
                'Epoch:', epoch,
                '  Train loss: %.6f' % loss.item(),
                '  Test loss: %.6f' % test_loss.item(),
                '  Train accuracy %.6f' % train_acc,
                '  Test accuracy is %.6f' % test_acc
            )

            self.log_write.write(str(epoch) + "    " + str(test_acc) + "\n")
            num += 1
            aver_acc += test_acc

        aver_acc = aver_acc / num
        print('The average accuracy is:', aver_acc)
        print('The best accuracy is:', best_acc)
        self.log_write.write('The average accuracy is: ' + str(aver_acc) + "\n")
        self.log_write.write('The best accuracy is: ' + str(best_acc) + "\n")
        self.writer.close()
        self.log_write.close()
        return best_acc, aver_acc, y_true_best, y_pred_best


def parse_args():
    parser = argparse.ArgumentParser(description='Low-channel Batch 2: diff branch')
    parser.add_argument('--subject', type=int, required=True, help='被试编号 1-9')
    parser.add_argument('--window_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--channel_config', type=str, default=None,
                        choices=list(CHANNEL_PRESETS.keys()),
                        help='预设通道配置: full/central8/c3czc4/c3c4')
    parser.add_argument('--channels', type=str, default=None,
                        help='自定义通道索引(0-based), 逗号分隔, 如 7,9,11')
    parser.add_argument('--classes', type=str, default='1,2,3,4',
                        help='类别选择(1-based), 逗号分隔, 如 1,2')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

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
        channel_indices = [int(x.strip()) for x in args.channels.split(',')]
        config_name = f"custom{len(channel_indices)}"
    else:
        channel_indices = list(range(22))
        config_name = "full"

    selected_classes = [int(x.strip()) for x in args.classes.split(',')]
    channel_names = [BCI2A_CHANNELS[i] for i in channel_indices]

    print(f"\n{'=' * 60}")
    print("  Low-channel Batch 2: diff branch")
    print(f"  Subject:  {args.subject}")
    print(f"  Config:   {config_name} ({len(channel_indices)} channels)")
    print(f"  Channels: {channel_names}")
    print(f"  Classes:  {selected_classes} ({len(selected_classes)}-class)")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Seed:     {args.seed}")
    print(f"{'=' * 60}\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    os.makedirs('./results', exist_ok=True)

    starttime = datetime.datetime.now()
    exp = ExP(
        nsub=args.subject,
        window_size=args.window_size,
        channel_indices=channel_indices,
        selected_classes=selected_classes,
        use_diff_branch=True,
    )
    exp.n_epochs = args.epochs
    best_acc, aver_acc, _, _ = exp.train()

    duration = datetime.datetime.now() - starttime
    print(f"\n  完成! Best={best_acc:.4f} ({best_acc * 100:.2f}%)  Aver={aver_acc:.4f}  耗时={duration}")
    print("MODEL_TAG: lowchannel_b2_diff")
    print(
        f"RESULT_CSV: {args.subject},{config_name},{len(channel_indices)},"
        f"{len(selected_classes)},{args.window_size},{args.seed},{args.epochs},"
        f"{best_acc:.6f},{aver_acc:.6f},{duration}"
    )


if __name__ == '__main__':
    main()

