# Low-Channel Stepwise Optimization Plan

## 1. 目的

本文件用于约束后续低通道优化的推进方式：

- 不再继续“大改版一次性堆很多新结构”
- 改为**最多 4 批次**的逐步消融式推进
- 每一批只增加一个主要变量
- 如果连续 `4` 批都没有优于原 baseline，则回到原模型主线

这份规则的直接背景是：

- 现有 [conformer_lowchannel_v1.py](/home/woqiu/下载/git/MI_Algorithm_Workbench/baselines/conformer_lowchannel_v1.py) 一次性改动过多
- 当前已见结果没有稳定超过 baseline
- 因而需要把“哪里有效、哪里无效”拆开验证

---

## 2. 基线口径

### 2.1 主比较对象

主比较对象固定为：

- [conformer_degradation.py](/home/woqiu/下载/git/MI_Algorithm_Workbench/baselines/conformer_degradation.py)

主评价任务固定为：

- `c3c4 / 2-class`

原因：

- 这是当前最接近工程落地目标的设置
- 也是双导条件下最有希望形成稳定结果的任务

### 2.2 pilot 被试集合

为了减少每批迭代成本，统一使用以下 `5` 个 pilot 被试：

- `S1`
- `S3`
- `S5`
- `S8`
- `S9`

它们覆盖了相对强、中、弱不同水平，不容易被单一被试误导。

### 2.3 pilot 基线数值

基于现有结果表，pilot 被试上的 baseline 均值如下：

#### `c3c4 / 2-class`

- `Avg Best = 0.750000`
- `Avg Aver = 0.697506`

#### `c3czc4 / 4-class`

- `Avg Best = 0.633340`
- `Avg Aver = 0.562900`

#### `c3c4 / 4-class`

- `Avg Best = 0.534700`
- `Avg Aver = 0.484140`

---

## 3. 停止规则

后续最多允许尝试 `4` 批结构改动：

- `Batch 1`
- `Batch 2`
- `Batch 3`
- `Batch 4`

若到 `Batch 4` 结束时，仍没有任一批在主评价任务上稳定优于 baseline，则执行以下决策：

1. 停止继续扩展低通道新结构
2. 将 [conformer_degradation.py](/home/woqiu/下载/git/MI_Algorithm_Workbench/baselines/conformer_degradation.py) 恢复为低通道主 baseline
3. 后续主线回到：
   - `2导 / 2分类` 原 baseline
   - `EdgeMIFormer` 或论文整理

这里的“优于 baseline”默认指：

- 首先看 `c3c4 / 2-class` 的 pilot 均值是否高于 baseline
- 若均值非常接近，则再看是否至少 `3/5` 个被试优于 baseline

---

## 4. 批次优先级

## Batch 1：只改分类头

优先级：最高  
风险：最低  
改动原则：**只改 head，不改前端、不改 encoder**

目标：

- 验证 low-channel 问题是否主要来自 `flatten + MLP head`

做法：

- 把原模型的 flatten 分类头改成 `attention pooling head`
- 保留原始：
  - PatchEmbedding
  - LocalAttention
  - Transformer depth
  - 数据增强
  - 训练协议

理由：

- 这是当前最小、最干净的结构改动
- 对低通道最有可能的好处是减少过拟合与无效 token 展平
- 如果连这一步都没有收益，说明“问题不只是 head”

---

## Batch 2：在 Batch 1 基础上加入 2导差分分支

优先级：第二  
风险：低到中等  
改动原则：**只在 `2导` 场景下增加输入重组**

目标：

- 验证 `C3/C4` 的侧化差分是否能帮助 `2-class`

做法：

- 对 `c3c4` 输入构造：
  - `C3`
  - `C4`
  - `C3-C4`
- 或者保守版只增加一个差分支路

理由：

- 该修改仍不引入额外通道信息
- 但比 Batch 1 多改了一层输入表征，因此排在第二批

---

## Batch 3：轻量通道权重/门控

优先级：第三  
风险：中等  
改动原则：**在保留原前端的基础上做轻量通道重标定**

目标：

- 验证低通道下是否存在“错误等权融合”问题

做法：

- 在输入或浅层特征上增加轻量 gate / SE 样式加权
- 不改主干时序卷积和 encoder 深度

理由：

- 比 Batch 1、2 更容易引入新变量
- 但仍明显比 `lowchannel_v1` 的整套前端重写更克制

---

## Batch 4：轻量前端改造

优先级：第四  
风险：最高  
改动原则：**只在前三批都不成功时才进入**

目标：

- 验证低通道是否确实需要更换前端形式

做法：

- 可尝试更轻的多尺度时间卷积或晚期融合
- 但必须比当前 `lowchannel_v1` 更克制
- 禁止在这一批同时叠加太多组件

理由：

- 前端一旦重写，变量增多，结果解释会快速变差
- 因此它必须是最后一批，而不是第一批

---

## 5. 当前执行结论

从当前已有试验结果看，`lowchannel_v1` 暂时不应继续作为主线，因为：

- `3导 / 4分类` 已明显落后 baseline
- `2导 / 2分类` 在已完成的 `S1/S3/S5` 上也未优于 baseline
- 该模型改动过大，难以定位具体失效原因

因此，接下来的正确做法不是继续在 `v1` 上硬调，而是：

> 从最小结构改动开始，按 `Batch 1 -> Batch 2 -> Batch 3 -> Batch 4` 逐步推进；如果 4 批都没有稳定超过原 baseline，就回到原模型，不再继续扩展这条支线。

---

## 6. Batch 1 当前结论

### 6.1 已完成对比

`Batch 1` 采用的实现是：

- [conformer_lowchannel_b1_poolhead.py](/home/woqiu/下载/git/MI_Algorithm_Workbench/baselines/conformer_lowchannel_b1_poolhead.py)

它只做了一件事：

- 把原 baseline 的 `flatten + MLP head` 改为 `attention pooling head`

在主评价任务 `c3c4 / 2-class` 上，已完成的 `5` 个 pilot 被试结果为：

- `Batch 1 Avg Best = 0.744444`
- `Batch 1 Avg Aver = 0.684883`

对应 baseline 为：

- `Baseline Avg Best = 0.750000`
- `Baseline Avg Aver = 0.697506`

逐被试统计：

- `Best Acc` 改善：`2 / 5`
- `Aver Acc` 改善：`1 / 5`

### 6.2 结论判定

按本文件前述 stop rule，`Batch 1` 目前应判为：

- **未过线**

原因：

- 主评价任务均值没有超过 baseline
- 被试级改善数量也不足以支持“稳定优于 baseline”的判断

因此，不建议继续补跑 `Batch 1` 的剩余大规模实验，而应直接进入 `Batch 2`。

---

## 7. Batch 2 当前定义

`Batch 2` 采用的策略是：

- 保留 `Batch 1` 的 pooling head
- 不改前端主干、不改 encoder
- **只在 `2导` 条件下增加 `C3-C4` 差分分支**

对应实现将放在：

- [conformer_lowchannel_b2_diff.py](/home/woqiu/下载/git/MI_Algorithm_Workbench/baselines/conformer_lowchannel_b2_diff.py)

主评价任务仍固定为：

- `c3c4 / 2-class`

也就是说，`Batch 2` 只回答一个问题：

> 在保持 Batch 1 其余设置不变的前提下，仅加入 `C3-C4` 侧化差分，是否能让双导二分类稳定超过原 baseline？
