# Degradation Visualization Notes

## 1. 本次新增图表

已基于当前结果汇总表生成以下两张图：

- [degradation_curve_4class.png](/home/woqiu/下载/git/MI_Algorithm_Workbench/visualization/degradation_curve_4class.png)
- [c3c4_4class_vs_2class.png](/home/woqiu/下载/git/MI_Algorithm_Workbench/visualization/c3c4_4class_vs_2class.png)

生成脚本为：

- [plot_lowchannel_results.py](/home/woqiu/下载/git/MI_Algorithm_Workbench/visualization/plot_lowchannel_results.py)

---

## 2. 图 1：四分类通道退化曲线

图 1 基于 [degradation_4class.csv](/home/woqiu/下载/git/MI_Algorithm_Workbench/results_summaries/degradation_4class.csv) 的 `9` 被试均值绘制，展示 `22 -> 8 -> 3 -> 2` 导条件下的性能变化。

关键数字如下：

- `22导 / 4分类`: `Avg Best = 73.42%`, `Avg Aver = 60.05%`
- `8导 / 4分类`: `Avg Best = 65.97%`, `Avg Aver = 55.17%`
- `3导 / 4分类`: `Avg Best = 59.11%`, `Avg Aver = 51.70%`
- `2导 / 4分类`: `Avg Best = 49.73%`, `Avg Aver = 44.41%`

该图支持以下结论：

- 通道数下降与识别性能下降之间存在明确单调关系。
- `8` 导仍保留了较强的可用性。
- `3` 导虽明显退化，但仍具有分析意义。
- `2` 导 `4` 分类已进入明显高风险区，不适合作为最终工程落地主线。

---

## 3. 图 2：C3/C4 下四分类与二分类对比

图 2 同时使用：

- [degradation_4class.csv](/home/woqiu/下载/git/MI_Algorithm_Workbench/results_summaries/degradation_4class.csv)
- [degradation_2class.csv](/home/woqiu/下载/git/MI_Algorithm_Workbench/results_summaries/degradation_2class.csv)

比较对象为同样的 `C3/C4` 双导输入：

- `2导 / 4分类`: `Avg Best = 49.73%`, `Avg Aver = 44.41%`
- `2导 / 2分类`: `Avg Best = 70.76%`, `Avg Aver = 64.88%`

该图支持以下结论：

- 在双导条件下，任务从 `4` 分类重定义为 `左右手二分类` 后，性能显著回升。
- 该提升不是来自额外通道或知识蒸馏，而是在同一数据集、同一协议、同一双导输入下，通过任务重定义获得的。
- 因此，`2导 4分类` 更适合作为“低通道退化证据”，而 `2导 2分类` 更适合作为“工程落地入口”。

---

## 4. 当前图表在论文中的建议定位

建议后续在论文中将这两张图分别承担不同角色：

- 图 1：作为“通道退化证据图”，用于说明低通道限制是真实存在的。
- 图 2：作为“任务重定义合理性图”，用于说明为什么最终工程展示应聚焦 `C3/C4` 左右手二分类。

这样可以形成清晰叙事链：

`全通道高性能 -> 低通道退化 -> 双导四分类失稳 -> 双导二分类成为合理落地方案`

---

## 5. 下一步建议

在图表固化后，下一步应优先验证：

- [conformer_lowchannel_v1.py](/home/woqiu/下载/git/MI_Algorithm_Workbench/baselines/conformer_lowchannel_v1.py) 在 `c3c4 / 2分类` 上是否优于当前 baseline
- 同时观察它在 `c3czc4 / 4分类` 上是否能追回一部分低通道损失

如果 `lowchannel_v1` 在上述设置下有稳定收益，则可以进入更大范围的 `pilot -> 9被试扩展`。

