# 混淆矩阵与错误分析总结

## 1. 文档目的

本文档用于记录 `P2` 的执行结果，即基于当前已完成实验的最佳混淆矩阵，对关键设置进行聚合错误分析，并形成后续可直接转化为论文讨论文字的材料。

本轮分析尽量遵守两条原则：

- **不重跑训练**
- **优先直接利用当前主线结果目录中的 `best_confusion_matrix.npy`**

为保证可追溯性，本次分析不是简单地选择“最新目录”，而是根据结果汇总表中的 `best_acc` 反向匹配到准确率一致的结果目录。

核心脚本为：

- [analyze_confusion_matrices.py](/home/woqiu/下载/git/MI_Algorithm_Workbench/visualization/analyze_confusion_matrices.py)

关键输出包括：

- [confusion_matrix_key_conditions.png](/home/woqiu/下载/git/MI_Algorithm_Workbench/visualization/confusion_matrix_key_conditions.png)
- [confusion_matrix_kd_pilot.png](/home/woqiu/下载/git/MI_Algorithm_Workbench/visualization/confusion_matrix_kd_pilot.png)
- [class_recall_comparison.png](/home/woqiu/下载/git/MI_Algorithm_Workbench/visualization/class_recall_comparison.png)
- [confusion_matrix_selection_manifest.csv](/home/woqiu/下载/git/MI_Algorithm_Workbench/results_summaries/confusion_matrix_selection_manifest.csv)
- [confusion_analysis_summary.csv](/home/woqiu/下载/git/MI_Algorithm_Workbench/results_summaries/confusion_analysis_summary.csv)
- [class_recall_precision_summary.csv](/home/woqiu/下载/git/MI_Algorithm_Workbench/results_summaries/class_recall_precision_summary.csv)

---

## 2. 分析对象

本轮重点分析了以下五组聚合混淆矩阵：

| 条件 | 说明 |
| --- | --- |
| `full_4class_9sub` | `22导 / 4分类` baseline，`9` 被试 |
| `c3c4_4class_9sub` | `C3/C4 / 4分类` baseline，`9` 被试 |
| `c3c4_2class_9sub` | `C3/C4 / 2分类` baseline，`9` 被试 |
| `c3c4_4class_pilot5_baseline` | `C3/C4 / 4分类` baseline，KD 对应的 `5` 被试 pilot |
| `c3c4_4class_pilot5_kd` | `22导 teacher -> 2导 student` KD，`5` 被试 pilot |

类标定义如下：

- `4分类`：`Left / Right / Feet / Tongue`
- `2分类`：`Left / Right`

---

## 3. 结果匹配可靠性

本次自动匹配生成了：

- [confusion_matrix_selection_manifest.csv](/home/woqiu/下载/git/MI_Algorithm_Workbench/results_summaries/confusion_matrix_selection_manifest.csv)

该文件记录了：

- 条件
- 被试
- 目标 `best_acc`
- 实际选中的结果目录
- 由混淆矩阵反算得到的准确率
- 二者差值

当前匹配结果显示：

- `22导 / 4分类`、`2导 / 4分类`、`2导 / 2分类` 主线矩阵都与汇总表准确率高度一致
- 差值量级基本在 `1e-5 ~ 1e-4`
- `KD pilot` 的 `5` 个被试则全部实现了近乎精确匹配

因此，这批混淆矩阵可以视为与当前主结果表一致的可信分析对象。

---

## 4. 聚合准确率概览

聚合结果见：

- [confusion_analysis_summary.csv](/home/woqiu/下载/git/MI_Algorithm_Workbench/results_summaries/confusion_analysis_summary.csv)

关键数字如下：

| 条件 | 总样本数 | 聚合准确率 |
| --- | --- | --- |
| `22导 / 4分类` | `2592` | `0.734182` |
| `C3/C4 / 4分类` | `2592` | `0.497299` |
| `C3/C4 / 2分类` | `1296` | `0.707562` |
| `C3/C4 / 4分类 pilot baseline` | `1440` | `0.534722` |
| `C3/C4 / 4分类 pilot KD` | `1440` | `0.547222` |

这与已有主结果表完全一致，也说明：

- `P2` 不是在构造新的结果，而是在解释当前结果为什么会这样。

---

## 5. 关键错误模式分析

## 5.1 `22导 / 4分类`：四类总体较均衡

在 `22导 / 4分类` 条件下，四类的召回率分别为：

| 类别 | Recall |
| --- | --- |
| Left | `0.7238` |
| Right | `0.7454` |
| Feet | `0.7562` |
| Tongue | `0.7114` |

这说明：

- 全通道条件下，四类识别虽然存在混淆，但总体仍较均衡。
- 不存在某一类完全崩溃的情况。

主要混淆对包括：

- `Left -> Right = 83`
- `Right -> Left = 81`
- `Feet -> Tongue = 88`
- `Tongue -> Feet = 93`

可见在全通道条件下：

- 左右手之间仍有一定侧化重叠
- 脚和舌头之间也存在明显混淆

但这些混淆仍未破坏整体四分类可用性。

---

## 5.2 `C3/C4 / 4分类`：错误明显集中在“左右手交叉”与“脚-舌头混淆”

在 `C3/C4 / 4分类` 条件下，四类召回率变为：

| 类别 | Recall |
| --- | --- |
| Left | `0.5602` |
| Right | `0.4738` |
| Feet | `0.3873` |
| Tongue | `0.5679` |

与 `22导 / 4分类` 相比，召回率下降最明显的是：

- `Feet`: `0.7562 -> 0.3873`
- `Right`: `0.7454 -> 0.4738`

其次是：

- `Left`: `0.7238 -> 0.5602`
- `Tongue`: `0.7114 -> 0.5679`

这说明：

- 双导四分类下，**脚类最先崩溃**
- 右手次之
- 舌头虽然也下降，但相对保留程度略高于脚

最强的错误通道包括：

- `Right -> Left = 151`
- `Left -> Right = 132`
- `Feet -> Tongue = 197`
- `Tongue -> Feet = 124`

因此，`2导 / 4分类` 的退化不是一种单一错误模式，而是同时包含：

1. **左右手之间的侧化区分能力下降**
2. **中线相关类别（尤其 Feet）与 Tongue 的严重混淆**

这条结果非常重要，因为它告诉我们：

- 双导条件下的问题并不只是“少了几个百分点”
- 而是类别间信息结构本身发生了明显塌缩

---

## 5.3 `C3/C4 / 2分类`：左右手仍保持可接受且相对平衡的识别

在 `C3/C4 / 2分类` 条件下，聚合混淆矩阵为：

| True \ Pred | Left | Right |
| --- | --- | --- |
| Left | `473` | `175` |
| Right | `204` | `444` |

对应召回率：

| 类别 | Recall |
| --- | --- |
| Left | `0.7299` |
| Right | `0.6852` |

对应精确率：

| 类别 | Precision |
| --- | --- |
| Left | `0.6987` |
| Right | `0.7173` |

这说明：

- 双导二分类下，左右手两类都维持了较为接近的识别能力
- 虽然仍存在一定交叉误判，但已不再出现四分类条件下的类别塌缩问题

换句话说：

> 双导条件下，任务重定义并不是“绕开难题”，而是让任务与可用信息结构重新匹配。

这正是当前论文主线中“`2导 / 4分类` 作为退化证据，`2导 / 2分类` 作为工程落地方案”的最强支撑之一。

---

## 5.4 KD 的提升主要集中在左右手，而不是完全恢复四分类

将 `KD pilot` 与其对应的 `5` 被试双导四分类 baseline 比较：

### pilot baseline（5 被试）召回率

| 类别 | Recall |
| --- | --- |
| Left | `0.5917` |
| Right | `0.5333` |
| Feet | `0.3806` |
| Tongue | `0.6333` |

### KD pilot（5 被试）召回率

| 类别 | Recall |
| --- | --- |
| Left | `0.6556` |
| Right | `0.5778` |
| Feet | `0.3611` |
| Tongue | `0.5944` |

召回率变化为：

| 类别 | 变化 |
| --- | --- |
| Left | `+0.0639` |
| Right | `+0.0444` |
| Feet | `-0.0194` |
| Tongue | `-0.0389` |

这说明：

- KD 的收益主要体现在**左右手两类**
- 对 `Feet` 与 `Tongue` 两类，并没有形成同步提升
- 换句话说，KD 在当前设定下更像是在**增强左右侧化相关判别**，而不是全面恢复所有双导四分类信息

这一点很值得在论文中如实写出，因为它使结论更可信：

> KD 并非万能补偿器，它更可能优先迁移那些本就在双导中仍保留一定可分性的类别信息。

---

## 6. 图表建议用法

## 6.1 主图

推荐将以下图表纳入正文：

- [confusion_matrix_key_conditions.png](/home/woqiu/下载/git/MI_Algorithm_Workbench/visualization/confusion_matrix_key_conditions.png)

适合作为：

- `22导 / 4分类`
- `2导 / 4分类`
- `2导 / 2分类`

三者的核心错误模式对照图。

## 6.2 补充图

推荐将以下图放入补充分析或讨论章节：

- [confusion_matrix_kd_pilot.png](/home/woqiu/下载/git/MI_Algorithm_Workbench/visualization/confusion_matrix_kd_pilot.png)
- [class_recall_comparison.png](/home/woqiu/下载/git/MI_Algorithm_Workbench/visualization/class_recall_comparison.png)

它们非常适合支持：

- KD 到底改善了哪些类别
- 哪些类别仍然是双导四分类的根本瓶颈

---

## 7. 对论文写作的直接价值

这轮 `P2` 的结果可以直接增强以下几个位置：

### 7.1 解释为什么 `2导 / 4分类` 不适合做最终落地主线

现在不仅能说它“准确率低”，还能具体说：

- 脚类召回率下降最严重
- 左右手之间也出现明显交叉混淆
- 脚和舌头的混淆在双导条件下进一步放大

### 7.2 支撑为什么 `2导 / 2分类` 是合理的工程重定义

现在可以更准确地写：

- 双导二分类下左右手两类都保持了较为均衡的召回率与精确率
- 任务重定义有效避免了双导四分类下的结构性类别塌缩

### 7.3 限定 KD 的真实作用边界

现在可以更加克制、也更有说服力地写：

- KD 有效
- 但其当前收益主要集中在左右手类
- 对 Feet / Tongue 的恢复能力仍有限

---

## 8. 一句话总结

`P2` 的关键结论可以概括为：

> 双导四分类性能下降并非均匀退化，而是集中表现为左右手交叉误判增加以及 Feet-Tongue 类别混淆放大；双导二分类则显著缓解了这一结构性错误模式；KD 的补偿效果主要加强了左右手判别，而尚未根本恢复双导四分类中的中线类信息。
