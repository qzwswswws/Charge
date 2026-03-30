# P10 PhysioNet eegmmidb 9 被试外部验证总结

## 1. 文档目的

本文档用于固定 `PhysioNet eegmmidb` 在当前工作区中的第一轮正式结果，并说明它在论文中的正确定位。

这轮结果的价值不在于替代 `BCI Competition IV 2a/2b` 主线，而在于回答：

> 当前中央区低通道二分类方案，能否在一个不同来源、不同采样率、不同任务组织方式的公开 MI 数据集上继续跑通？

---

## 2. 实验定位

本轮 `PhysioNet eegmmidb` 实验应被定位为：

- 外部数据集补充验证
- 单随机种子正式结果
- 轻量协议下的跨数据源检查

它不是：

- `2a/2b` 主线结果的替代品
- 与 `2a/2b` 完全同口径的直接数值比较

---

## 3. 数据与协议

### 3.1 数据集

- 数据集：`PhysioNet EEG Motor Movement/Imagery Dataset (eegmmidb)`
- 官方页：<https://physionet.org/content/eegmmidb/1.0.0/>
- 当前使用任务：想象左右手
- 当前使用运行：`R04`、`R08`、`R12`

### 3.2 预处理入口

- 原始数据目录：[physionet_eegmmidb_raw](/home/woqiu/下载/git/MI_Algorithm_Workbench/datasets/physionet_eegmmidb_raw)
- 预处理脚本：[preprocess_physionet_eegmmidb.py](/home/woqiu/下载/git/MI_Algorithm_Workbench/preprocessing/preprocess_physionet_eegmmidb.py)
- 预处理输出目录：[standard_physionet_eegmmidb](/home/woqiu/下载/git/MI_Algorithm_Workbench/datasets/standard_physionet_eegmmidb)

### 3.3 当前实验协议

- 通道：`C3 / Cz / C4`
- 类别：`2-class`
- 标签映射：`T1 -> left`，`T2 -> right`
- 训练运行：`R04 + R08`
- 测试运行：`R12`
- epoch 长度：`4 s`
- 采样率：`160 Hz`
- 单 trial 输入长度：`640` 点
- `window_size = 8`
- `epochs = 250`
- `seed = 42`

### 3.4 与 `2a/2b` 的重要区别

这一点后续论文必须写清楚：

- `eegmmidb` 不是 `BCI Competition IV 2a/2b`
- 当前协议是 `run-level split`
- 当前每被试只使用 `R04/R08/R12`
- 每个被试总共只有 `45` 个目标 trial
  - `30` 个训练
  - `15` 个测试

因此，这里的单被试数据长度明显短于当前 `2a/2b` 主线协议。  
这条线的意义更偏向“外部有效性检查”，而不是与 `2a/2b` 做完全等价的数值对齐。

---

## 4. 预处理核对结论

当前已对 `S001/S003/S005/S008/S009` 的 `R04/R08/R12` 做过原始 `EDF` 与预处理 `.mat` 的逐条核对，结果表明：

- 原始 `EDF` 中每个目标 run 都有 `15` 个 `T1/T2`
- 预处理后每个 `.mat` 都保留 `15` 个 trial
- 输出 shape 一致为 `(640, 3, 15)`
- 通道一致为 `C3 / Cz / C4`

因此，当前至少在已验证被试上，没有发现“trial 数不匹配”或“通道切错”的明显问题。

---

## 5. 脚本与结果文件

### 5.1 训练入口

- baseline 脚本：[conformer_physionet_eegmmidb_baseline.py](/home/woqiu/下载/git/MI_Algorithm_Workbench/baselines/conformer_physionet_eegmmidb_baseline.py)
- `5` 被试 pilot 运行器：[run_physionet_eegmmidb_pilot.py](/home/woqiu/下载/git/MI_Algorithm_Workbench/baselines/run_physionet_eegmmidb_pilot.py)
- `9` 被试扩展运行器：[run_physionet_eegmmidb_full.py](/home/woqiu/下载/git/MI_Algorithm_Workbench/baselines/run_physionet_eegmmidb_full.py)

### 5.2 结果文件

- `5` 被试 pilot：[physionet_eegmmidb_pilot_latest.csv](/home/woqiu/下载/git/MI_Algorithm_Workbench/results_summaries/physionet_eegmmidb_pilot_latest.csv)
- `9` 被试正式结果：[physionet_eegmmidb_full_9sub_latest.csv](/home/woqiu/下载/git/MI_Algorithm_Workbench/results_summaries/physionet_eegmmidb_full_9sub_latest.csv)

---

## 6. 结果

### 6.1 `5` 被试 pilot

| 指标 | 数值 |
| --- | --- |
| Avg Best Acc | `0.7600` |
| Avg Aver Acc | `0.5581` |

### 6.2 `9` 被试扩展结果

| Subject | Best Acc | Aver Acc |
| --- | --- | --- |
| S1 | `0.800000` | `0.591733` |
| S2 | `0.666667` | `0.552000` |
| S3 | `0.733333` | `0.544267` |
| S4 | `0.933333` | `0.779467` |
| S5 | `0.666667` | `0.518400` |
| S6 | `0.600000` | `0.439733` |
| S7 | `1.000000` | `0.924267` |
| S8 | `0.866667` | `0.590133` |
| S9 | `0.733333` | `0.545867` |

`9` 被试平均结果为：

| 指标 | 数值 |
| --- | --- |
| Avg Best Acc | `0.777778` |
| Avg Aver Acc | `0.609541` |

---

## 7. 结果解读

### 7.1 可以明确支持的结论

- `eegmmidb` 这条线已经从单被试 smoke 扩展到 `9` 被试正式结果。
- 在 `C3/Cz/C4 / 2-class`、且每被试只有 `45` 个目标 trial 的轻量协议下，平均最佳准确率仍达到 `77.78%`。
- 从 `5` 被试扩到 `9` 被试后，平均最佳准确率没有下降，反而从 `0.7600` 提升到 `0.7778`。

### 7.2 需要保留的边界

- 当前仍是 `seed=42` 的单种子结果。
- 协议是 `run-level split`，不是 `2a/2b` 的竞赛式 `T/E` 划分。
- 由于单被试 trial 数更少，结果更适合写成“外部数据集支持证据”，不宜直接写成与 `2a/2b` 完全同口径的数值对比。

### 7.3 当前最稳的论文定位

最稳的写法是：

> 在不同于 `BCI Competition IV 2a/2b` 的公开运动想象数据集 `PhysioNet eegmmidb` 上，采用 `C3/Cz/C4` 三导、左右手二分类、`R04+R08` 训练与 `R12` 测试的轻量协议进行补充验证。结果表明，在 `9` 个被试上模型获得 `77.78%` 的平均最佳准确率。尽管该协议下单被试目标 trial 数明显少于 `2a/2b` 主线实验，但该结果仍说明当前中央区低通道方案具备跨数据源的可迁移性。

---

## 8. 下一步建议

如果继续在 `PhysioNet` 上扩展，最合理的顺序是：

1. 先把当前 `9` 被试结果写入论文或总述文档。
2. 如仍需补强，再考虑多随机种子复现。
3. 不建议立刻扩到大量被试或复杂多任务版本，以免偏离论文主线。

---

## 9. 一句话结论

如果把 `P10` 压缩成一句话，可以写成：

> `PhysioNet eegmmidb` 的 `9` 被试结果表明，即使在单被试 trial 更短的轻量协议下，`C3/Cz/C4` 三导左右手二分类方案仍保持较强表现，因此可作为当前论文的一条有价值的外部数据集补充验证线。
