# P8 BCI Competition IV 2b 外部验证 9 被试总结

## 1. 文档目的

本文档用于固定 `P8` 在 `BCI Competition IV 2b` 上扩展到 `9` 被试后的正式结果。若需要当前最强的统计口径，应优先结合 [BCIIV2B_MultiSeed_Summary.md](/home/woqiu/下载/git/MI_Algorithm_Workbench/00_AI_Management/Output_Drafts/BCIIV2B_MultiSeed_Summary.md) 一并引用。

---

## 2. 实验定位

本轮实验的定位不是替代 `2a` 主线，而是回答如下问题：

> 在 `2a` 上已经成立的“中央区三导二分类可行”结论，能否在 `BCI Competition IV 2b` 上获得更完整的外部支持？

因此，本轮 `9` 被试扩展应被定位为：

- 外部数据集补充验证
- 单种子正式结果
- 为后续多随机种子复现提供前置依据

---

## 3. 实验设置

### 3.1 数据与预处理

- 数据集：`BCI Competition IV 2b`
- 原始文件：`45` 个 `gdf`
- 标签文件：`45` 个 `classlabel .mat`
- 预处理脚本：[preprocess_bci_iv_2b.py](/home/woqiu/下载/git/MI_Algorithm_Workbench/preprocessing/preprocess_bci_iv_2b.py)
- 标准输入目录：`datasets/standard_2b_strict_TE/`

预处理口径为：

- 事件起点：`768`
- 截取窗口：事件后偏移 `750` 个采样点起，长度 `1000` 个采样点
- 采样率：`250 Hz`
- 保留通道：`C3 / Cz / C4`
- 带通：`4-40 Hz`

### 3.2 模型与运行设置

- baseline 入口：[conformer_2b_baseline.py](/home/woqiu/下载/git/MI_Algorithm_Workbench/baselines/conformer_2b_baseline.py)
- 全量运行器：[run_2b_full.py](/home/woqiu/下载/git/MI_Algorithm_Workbench/baselines/run_2b_full.py)
- 通道数：`3`
- 类别数：`2`
- `window_size = 8`
- `epochs = 250`
- `seed = 42`
- 设备：`CPU`

### 3.3 被试范围

本轮已扩展到全部 `9` 个被试。

---

## 4. 结果

结果文件：

- [bciiv2b_full_9sub_latest.csv](/home/woqiu/下载/git/MI_Algorithm_Workbench/results_summaries/bciiv2b_full_9sub_latest.csv)

### 4.1 逐被试结果

| Subject | Best Acc | Aver Acc |
| --- | --- | --- |
| S1 | `0.784375` | `0.726738` |
| S2 | `0.625000` | `0.574200` |
| S3 | `0.581250` | `0.540887` |
| S4 | `0.987500` | `0.975675` |
| S5 | `0.871875` | `0.791200` |
| S6 | `0.884375` | `0.812500` |
| S7 | `0.856250` | `0.816438` |
| S8 | `0.943750` | `0.905588` |
| S9 | `0.921875` | `0.891225` |

### 4.2 平均结果

| 指标 | 数值 |
| --- | --- |
| Avg Best Acc | `0.828472` |
| Avg Aver Acc | `0.781606` |

---

## 5. 结果解读

### 5.1 可以明确支持的结论

- `2b` 外部验证已从 `5` 被试 pilot 扩展到 `9` 被试全量结果。
- 在 `C3/Cz/C4 / 2-class` 条件下，`9` 被试平均最佳准确率达到 `82.85%`，整体水平较强。
- 除 `S2` 和 `S3` 外，其余 `7` 个被试均表现较好，说明当前方案在 `2b` 上并非只对个别被试成立。

### 5.2 需要谨慎的边界

- 当前仍为 `seed=42` 的单种子结果，尚未形成多随机种子的统计稳定性结论。
- `2b` 与 `2a` 在协议与数据属性上并不完全相同，因此更稳妥的表述是“获得外部支持”，而非“实现跨数据集完全一致的数值复现”。

---

## 6. 与当前主线的关系

结合现有 `2a` 结果，可以形成如下判断：

- `2a` 上，`C3/Cz/C4 / 2-class` 在 `5` 被试 `3 seeds` 下达到  
  `Best Acc = 0.854630 ± 0.002122`
- `2b` 上，当前 `9` 被试单种子结果达到  
  `Avg Best Acc = 0.828472`

这说明：

- 中央区三导二分类并不是只在 `2a` 上偶然成立的现象。
- 换到 `2b` 后，尽管协议不同，整体量级仍然保持在较强区间。
- 因而该方向已经具备比较扎实的外部有效性雏形。

---

## 7. 论文中的推荐定位

当前最稳的论文表述方式是：

> 为进一步考察所提出中央区低通道二分类方案的外部适用性，在 `BCI Competition IV 2b` 数据集上完成了 `9` 被试验证。在 `C3/Cz/C4` 三导、二分类条件下，模型获得 `82.85%` 的平均最佳准确率和 `78.16%` 的平均准确率。该结果表明，当前方案在外部经典 MI 数据集上仍具有较好的可用性，从而为论文主线提供了进一步的跨数据集支持。与此同时，由于当前结果仍基于单随机种子，后续仍应通过多随机种子复现进一步强化其统计可靠性。

---

## 8. 后续说明

基于当前进展，本文件记录的是 `seed=42` 的单种子正式结果。  
由于 `2b` 的多随机种子复现现已完成，若写论文正文中的统计稳定性部分，更推荐直接引用：

- [BCIIV2B_MultiSeed_Summary.md](/home/woqiu/下载/git/MI_Algorithm_Workbench/00_AI_Management/Output_Drafts/BCIIV2B_MultiSeed_Summary.md)

---

## 9. 一句话结论

如果把当前 `P8` 的结果压缩成一句话，可以写成：

> `BCI Competition IV 2b` 的 `9` 被试单种子结果已经证明，`C3/Cz/C4` 三导二分类方案在外部数据集上保持较强性能，而其统计稳定性现已由后续多随机种子结果进一步补强。 
