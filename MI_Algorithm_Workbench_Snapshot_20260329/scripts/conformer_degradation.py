"""
通道退化实验脚本 (Channel Degradation Experiment)
基于 conformer_copy329.py，支持可配置的通道选择和类别选择。

用法:
    # 全通道 4分类
    python conformer_degradation.py --channel_config full --subject 1 --epochs 250
    # 2通道 4分类
    python conformer_degradation.py --channel_config c3c4 --subject 1 --epochs 250
    # 自定义通道
    python conformer_degradation.py --channels 7,9,11 --subject 1 --epochs 250
    # 2通道 2分类（左手 vs 右手）
    python conformer_degradation.py --channel_config c3c4 --classes 1,2 --subject 1
"""

import argparse
import os
import sys
import random
import datetime
import time
import math

import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn

cudnn.benchmark = False
cudnn.deterministic = True

gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ============================================================
#  BCI Competition IV 2a: 22-channel standard montage
# ============================================================
BCI2A_CHANNELS = [
    'Fz',  'FC3', 'FC1', 'FCz', 'FC2', 'FC4',     # 0-5
    'C5',  'C3',  'C1',  'Cz',  'C2',  'C4', 'C6', # 6-12
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',              # 13-17
    'P1',  'Pz',  'P2',  'POz'                      # 18-21
]

CHANNEL_PRESETS = {
    'full':     list(range(22)),                      # 全部 22 导
    'central8': [1, 5, 6, 7, 8, 10, 11, 12],         # FC3,FC4,C5,C3,C1,C2,C4,C6
    'c3czc4':   [7, 9, 11],                           # C3, Cz, C4
    'c3c4':     [7, 11],                              # C3, C4
}


# ============================================================
#  Model Components
# ============================================================

class PatchEmbedding(nn.Module):
    """卷积前端：空间卷积核大小 = (n_channels, 1)，自适应通道数。"""
    def __init__(self, emb_size=40, n_channels=22):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (n_channels, 1), (1, 1)),  # 自适应通道数
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 20)),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, 40, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x):
        x = self.shallownet(x)
        x = self.projection(x)
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
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)

        b, h, n, d = queries.shape
        local_mask = torch.ones(n, n, device=x.device)
        for i in range(n):
            local_mask[i, max(0, i - self.window_size):min(n, i + self.window_size + 1)] = 0
        local_mask = local_mask.bool().unsqueeze(0).unsqueeze(0)

        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        energy.masked_fill_(local_mask, torch.finfo(torch.float32).min)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


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
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size, window_size):
        super().__init__(*[TransformerEncoderBlock(emb_size, window_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        # fc input = n_tokens * emb_size
        # n_tokens = (1000 - 25 + 1 - 75) // 20 + 1 = 46
        # fc_input = 46 * 40 = 1840  (不随通道数变化)
        n_tokens = (1000 - 25 + 1 - 75) // 20 + 1  # 46
        fc_input = n_tokens * emb_size  # 1840

        self.fc = nn.Sequential(
            nn.Linear(fc_input, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)   # 自适应类别数
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out


class Conformer(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, n_classes=4, n_channels=22, window_size=8, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size, n_channels=n_channels),
            TransformerEncoder(depth, emb_size, window_size),
            ClassificationHead(emb_size, n_classes)
        )


# ============================================================
#  Experiment Class
# ============================================================

class ExP():
    def __init__(self, nsub, window_size, channel_indices=None, selected_classes=None):
        """
        Args:
            nsub: 被试编号 (1-9)
            window_size: 局部注意力窗口大小
            channel_indices: 通道索引列表 (0-based), None=全部22导
            selected_classes: 类别列表 (1-based, 如 [1,2,3,4]), None=全部4类
        """
        super(ExP, self).__init__()

        # 通道与类别配置
        self.channel_indices = channel_indices if channel_indices else list(range(22))
        self.n_channels = len(self.channel_indices)
        self.selected_classes = selected_classes if selected_classes else [1, 2, 3, 4]
        self.n_classes = len(self.selected_classes)
        # 标签重映射: 原始标签 -> 1-indexed 连续标签
        # 例如 {1:1, 2:2} 或 {2:1, 4:2}
        self.label_remap = {c: i + 1 for i, c in enumerate(sorted(self.selected_classes))}

        self.batch_size = 72
        # batch_size 必须能被 n_classes 整除（用于数据增强）
        while self.batch_size % self.n_classes != 0:
            self.batch_size -= 1

        self.n_epochs = 250
        self.c_dim = self.n_classes
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.nSub = nsub
        self.start_epoch = 0
        self.root = os.environ.get("BCI2A_DATA_ROOT", "/home/woqiu/下载/standard_2a_data/")

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.model = Conformer(
            n_channels=self.n_channels,
            n_classes=self.n_classes,
            window_size=window_size
        ).cuda()

        # 实验标识
        ch_tag = f"ch{self.n_channels}"
        cls_tag = f"cls{self.n_classes}"
        self.experiment_name = (
            f"subject_{self.nSub}_{ch_tag}_{cls_tag}_"
            f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.log_dir = f"./logsSh.{window_size}/{self.experiment_name}"
        self.model_dir = f"./models/{self.experiment_name}"
        self.result_dir = f"./results/{self.experiment_name}"

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

        self.log_write = open(f"./results/log_subject{self.nSub}_{ch_tag}_{cls_tag}.txt", "w")

        # CSV 日志
        self.train_log_path = f"{self.log_dir}/train_log.csv"
        self.test_log_path = f"{self.log_dir}/test_log.csv"
        with open(self.train_log_path, 'w') as f:
            f.write("epoch,batch,loss,lr\n")
        with open(self.test_log_path, 'w') as f:
            f.write("epoch,loss,accuracy,time\n")

        self.writer = SummaryWriter(log_dir=self.log_dir)

    def interaug(self, timg, label):
        """S&R 数据增强，通道数和类别数参数化。"""
        aug_data = []
        aug_label = []
        for cls4aug in range(self.n_classes):
            cls_idx = np.where(label == cls4aug + 1)  # 标签是 1-indexed
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

        aug_data = torch.from_numpy(aug_data).cuda().float()
        aug_label = torch.from_numpy(aug_label - 1).cuda().long()  # 1-indexed -> 0-indexed
        return aug_data, aug_label

    def get_source_data(self):
        """加载数据，支持通道筛选和类别筛选。"""
        # 训练数据
        total_data = scipy.io.loadmat(self.root + 'A0%dT.mat' % self.nSub)
        train_data = total_data['data']   # (1000, 22, n_trials)
        train_label = total_data['label'] # (n_trials, 1)

        train_data = np.transpose(train_data, (2, 1, 0))  # (n_trials, 22, 1000)
        train_data = np.expand_dims(train_data, axis=1)    # (n_trials, 1, 22, 1000)
        train_label = np.transpose(train_label)[0]         # (n_trials,)

        # 测试数据
        test_tmp = scipy.io.loadmat(self.root + 'A0%dE.mat' % self.nSub)
        test_data = test_tmp['data']
        test_label = test_tmp['label']

        test_data = np.transpose(test_data, (2, 1, 0))
        test_data = np.expand_dims(test_data, axis=1)
        test_label = np.transpose(test_label)[0]

        # ---- 类别筛选 ----
        if sorted(self.selected_classes) != [1, 2, 3, 4]:
            # 训练集
            train_mask = np.isin(train_label, self.selected_classes)
            train_data = train_data[train_mask]
            train_label = train_label[train_mask]
            # 标签重映射
            train_label = np.array([self.label_remap[l] for l in train_label])
            # 测试集
            test_mask = np.isin(test_label, self.selected_classes)
            test_data = test_data[test_mask]
            test_label = test_label[test_mask]
            test_label = np.array([self.label_remap[l] for l in test_label])

        # ---- 通道筛选 ----
        if len(self.channel_indices) < 22:
            train_data = train_data[:, :, self.channel_indices, :]
            test_data = test_data[:, :, self.channel_indices, :]

        # 打乱训练数据
        self.allData = train_data
        self.allLabel = train_label
        shuffle_num = np.random.permutation(len(self.allData))
        self.allData = self.allData[shuffle_num, :, :, :]
        self.allLabel = self.allLabel[shuffle_num]

        self.testData = test_data
        self.testLabel = test_label

        # 标准化 (用训练集统计量)
        target_mean = np.mean(self.allData)
        target_std = np.std(self.allData)
        self.allData = (self.allData - target_mean) / target_std
        self.testData = (self.testData - target_mean) / target_std

        return self.allData, self.allLabel, self.testData, self.testLabel

    def train(self):
        img, label, test_data, test_label = self.get_source_data()

        img = torch.from_numpy(img)
        label = torch.from_numpy(label - 1)  # 1-indexed -> 0-indexed

        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=True
        )

        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label - 1)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=self.batch_size, shuffle=True
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2)
        )

        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))

        bestAcc = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0

        total_step = len(self.dataloader)
        for e in range(self.n_epochs):
            epoch_start_time = time.time()
            self.model.train()
            for i, (img, label) in enumerate(self.dataloader):
                img = Variable(img.cuda().type(self.Tensor))
                label = Variable(label.cuda().type(self.LongTensor))

                aug_data, aug_label = self.interaug(self.allData, self.allLabel)
                img = torch.cat((img, aug_data))
                label = torch.cat((label, aug_label))

                tok, outputs = self.model(img)
                loss = self.criterion_cls(outputs, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                with open(self.train_log_path, 'a') as f:
                    f.write(f"{e},{i},{loss},{self.lr}\n")

                global_step = e * total_step + i
                self.writer.add_scalar('Train/Loss_Batch', loss, global_step)

            # 测试
            if (e + 1) % 1 == 0:
                self.model.eval()
                Tok, Cls = self.model(test_data)

                test_loss = self.criterion_cls(Cls, test_label)
                y_pred = torch.max(Cls, 1)[1]
                test_acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) \
                           / float(test_label.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) \
                            / float(label.size(0))

                with open(self.test_log_path, 'a') as f:
                    f.write(f"{e},{test_loss},{test_acc},{time.time()-epoch_start_time}\n")

                self.writer.add_scalar('Train/Loss_Epoch', loss, e)
                self.writer.add_scalar('Train/Accuracy_Epoch', train_acc, e)
                self.writer.add_scalar('Test/Loss', test_loss, e)
                self.writer.add_scalar('Test/Accuracy', test_acc, e)

                if test_acc > bestAcc:
                    bestAcc = test_acc
                    Y_true = test_label
                    Y_pred = y_pred
                    torch.save(self.model.state_dict(), f"{self.model_dir}/best_model.pth")
                    conf_matrix = confusion_matrix(
                        test_label.cpu().numpy(), y_pred.cpu().numpy()
                    )
                    np.save(f"{self.result_dir}/best_confusion_matrix.npy", conf_matrix)

                print('Epoch:', e,
                      '  Train loss: %.6f' % loss.detach().cpu().numpy(),
                      '  Test loss: %.6f' % test_loss.detach().cpu().numpy(),
                      '  Train accuracy %.6f' % train_acc,
                      '  Test accuracy is %.6f' % test_acc)

                self.log_write.write(str(e) + "    " + str(test_acc) + "\n")
                num = num + 1
                averAcc = averAcc + test_acc

        averAcc = averAcc / num
        print('The average accuracy is:', averAcc)
        print('The best accuracy is:', bestAcc)
        self.log_write.write('The average accuracy is: ' + str(averAcc) + "\n")
        self.log_write.write('The best accuracy is: ' + str(bestAcc) + "\n")
        self.writer.close()
        self.log_write.close()

        return bestAcc, averAcc, Y_true, Y_pred


# ============================================================
#  CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='通道退化实验')
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

    # 随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 解析通道
    if args.channel_config:
        channel_indices = CHANNEL_PRESETS[args.channel_config]
        config_name = args.channel_config
    elif args.channels:
        channel_indices = [int(x.strip()) for x in args.channels.split(',')]
        config_name = f"custom{len(channel_indices)}"
    else:
        channel_indices = list(range(22))
        config_name = "full"

    # 解析类别
    selected_classes = [int(x.strip()) for x in args.classes.split(',')]

    channel_names = [BCI2A_CHANNELS[i] for i in channel_indices]
    print(f"\n{'='*60}")
    print(f"  通道退化实验")
    print(f"  Subject:  {args.subject}")
    print(f"  Config:   {config_name} ({len(channel_indices)} channels)")
    print(f"  Channels: {channel_names}")
    print(f"  Classes:  {selected_classes} ({len(selected_classes)}-class)")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Seed:     {args.seed}")
    print(f"{'='*60}\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    os.makedirs('./results', exist_ok=True)

    starttime = datetime.datetime.now()

    exp = ExP(
        nsub=args.subject,
        window_size=args.window_size,
        channel_indices=channel_indices,
        selected_classes=selected_classes
    )
    exp.n_epochs = args.epochs
    bestAcc, averAcc, Y_true, Y_pred = exp.train()

    endtime = datetime.datetime.now()
    duration = endtime - starttime

    print(f"\n  完成! Best={bestAcc:.4f} ({bestAcc*100:.2f}%)  "
          f"Aver={averAcc:.4f}  耗时={duration}")

    # 输出机器可读的结果行
    print(f"\nRESULT_CSV: {args.subject},{config_name},{len(channel_indices)},"
          f"{len(selected_classes)},{args.window_size},{args.seed},{args.epochs},"
          f"{bestAcc:.6f},{averAcc:.6f},{duration}")


if __name__ == '__main__':
    main()
