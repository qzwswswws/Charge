# P7 EEGNet 开源资料索引

## 1. 文档目的

本文档用于记录当前已经下载到工作区的 `EEGNet` 开源参考资料，并明确它们各自更适合扮演什么角色，避免后续 `P7` 启动时重复筛选。

## 2. 已下载资料

### 2.1 官方 `arl-eegmodels`

本地路径：

- [arl-eegmodels](/home/woqiu/下载/git/MI_Algorithm_Workbench/external_refs/P7_EEGNet/arl-eegmodels)

关键文件：

- [README.md](/home/woqiu/下载/git/MI_Algorithm_Workbench/external_refs/P7_EEGNet/arl-eegmodels/README.md)
- [EEGModels.py](/home/woqiu/下载/git/MI_Algorithm_Workbench/external_refs/P7_EEGNet/arl-eegmodels/EEGModels.py)

定位：

- 这是 `EEGNet` 的高价值源头参考。
- 优点是模型出处直接、论文对应关系清楚。
- 附带实现还包含 `ShallowConvNet` 和 `DeepConvNet`，对后续扩展传统深度基线也有帮助。

当前限制：

- 该实现基于 `Keras / TensorFlow`
- 与当前工作区的 `PyTorch` 训练链不直接兼容

适合用途：

- 作为 `P7` 的结构与论文对应参考
- 用来核对超参数和层级设计
- 作为论文中“参考实现来源”的说明依据

### 2.2 非官方 `PyTorch EEGNet` 参考

本地路径：

- [EEGNet-pytorch-ref](/home/woqiu/下载/git/MI_Algorithm_Workbench/external_refs/P7_EEGNet/EEGNet-pytorch-ref)

关键文件：

- [README.md](/home/woqiu/下载/git/MI_Algorithm_Workbench/external_refs/P7_EEGNet/EEGNet-pytorch-ref/README.md)
- [EEGNet-PyTorch.ipynb](/home/woqiu/下载/git/MI_Algorithm_Workbench/external_refs/P7_EEGNet/EEGNet-pytorch-ref/EEGNet-PyTorch.ipynb)

定位：

- 这是一个便于快速迁移思路的 `PyTorch` 参考。
- 优点是与当前工作区技术栈更接近。

当前限制：

- 不是原作者官方实现
- 主要是 notebook 形态
- README 显示其环境与数据口径较旧

适合用途：

- 作为 `PyTorch` 版结构迁移参考
- 用来加快我们在当前工作区中实现 `EEGNet baseline` 的速度

## 3. 当前建议的使用策略

最稳的做法不是直接照搬某一个仓库，而是：

1. 以官方 [arl-eegmodels](/home/woqiu/下载/git/MI_Algorithm_Workbench/external_refs/P7_EEGNet/arl-eegmodels) 为结构与论文口径的主参考
2. 以 [EEGNet-pytorch-ref](/home/woqiu/下载/git/MI_Algorithm_Workbench/external_refs/P7_EEGNet/EEGNet-pytorch-ref) 为 `PyTorch` 迁移时的工程参考
3. 最终在当前工作区里单独实现一个干净的 `EEGNet baseline`，沿用我们已经固定好的：
   - 数据读取
   - 日志格式
   - `window_size`
   - `seed`
   - `results_summaries` 输出风格

## 4. 对 P7 的直接意义

这批资料已经足够支持我们开始 `P7` 的最小经典对照实验：

- 先做 `EEGNet`
- 优先对齐：
  - `22导 / 4分类`
  - `C3/C4 / 2分类`
- 如果 `EEGNet` 跑通且有论文价值，再决定是否补 `ShallowConvNet` 或 `FBCSP + SVM`

## 5. 一句话结论

如果把当前开源资料准备状态压缩成一句话，可以写成：

> `P7` 所需的 `EEGNet` 开源参考已经下载到工作区，其中官方 `arl-eegmodels` 更适合作为结构与论文口径的主参考，而 `PyTorch` notebook 实现更适合作为工程迁移参考。 
