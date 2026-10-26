# Student 有效点掩码版（2026-10-26）

本文件夹提供带有效点掩码的 Student 与配套实验脚本。仅处理点云填充掩码，不修改数据划分或随机种子策略；不包含数据集或权重。

根目录 `PNL_reweight_Segment_Align.py` 与 `MMFi` 中的共享模型相同，兼容仓库原有 `PYTHONPATH="$PWD:$PWD/Student:$PWD/Teacher"` 导入方式。

## 目录与来源

| 目录 | 原位置 | 范围 |
| --- | --- | --- |
| MMFi | 原始代码；根目录两个约束/分析脚本 | Student、Soft-Align、LSKD、WKDL、Shin/Tac-Mamba 适配、约束 warm-up、分析及复杂度脚本；配套 Teacher 和数据加载器 |
| mmO_HAR | fanhua_ours | scratch、radar/KD 预训练迁移、约束版迁移及共享 Student |
| RadHAR | fanhua_RadHAR | 现有 recording-group 划分版训练与迁移脚本及共享 Student，保留作旧实验代码，不代表重新加入论文 |
| tests | 新增 | CPU 单元测试，不访问实际数据集 |

MMFi 下两个额外脚本为 `KD_Align_Constrained_warmup.py` 和 `Analyze_Original_SoftAlign.py`。三个子目录中的 `PNL_reweight_Segment_Align.py` 完全相同。

## 掩码行为

- 只在原始五维点属性上确定有效性，不能在 T-Net、卷积、BatchNorm 后重新判断。
- 在 STN3d、STNkd 和最终 PointNet 的三处点级最大池化中，用负无穷排除无效点，避免零填充值压过负的有效特征。
- 全无效点帧的 PointNet 输出为零，避免出现负无穷或 NaN；空帧仍保留在 LSTM 时间序列中，不是时间掩码。
- Tac-Mamba 适配模型及向量化部署包装器也接入相同掩码。
- 旧接口 `model(x)`、`model(x, True)` 不变，默认把五个属性全为零的点当成填充；不是仅检查 XYZ。
- 可显式调用 `model(x, point_mask=mask)`，其中 `x` 为 `(B,T,5,N)`，`mask` 为布尔 `(B,T,N)`，True 表示有效。直接使用 PointNet 时输入为 `(B,5,N)`，掩码为 `(B,N)`。
- 自采数据四个加载器和 RadHAR 加载器在标准化前识别原始零填充点，标准化后重新置零，保证旧训练循环也可自动识别。有效点的标准化公式及原来的统计量估计方式不变。

```python
# 建议：若外部数据管线能提供真实长度/validity，应优先传显式掩码。
point_mask = raw_radar.ne(0).any(dim=2)  # raw_radar: (B,T,5,N)，标准化/增强之前
logits, segment_features = model(x, return_features=True, point_mask=point_mask)
```

## 使用

将所需子目录与仓库根目录配置配合使用，运行相应脚本。不要只上传 KD 脚本而继续导入旧版 Student。数据路径、预训练权重和输出目录均需按本地环境设置；个人服务器绝对路径已替换为 `data/`、`checkpoints/`、`outputs/` 和 `config.yaml` 等相对路径，默认按进程当前工作目录解释。

```bash
# 从本文件夹根目录运行，无需 GPU 或数据集
python tests/test_point_mask.py
```

测试需 PyTorch、NumPy。训练仍需原有 PyTorch/torchvision、NumPy、SciPy、scikit-learn、PyYAML、tqdm、TensorBoard、OpenCV 等依赖；复杂度统计按原脚本安装 thop。

## 兼容性与边界

- 不增加可学习参数，不更名原层，不改变 state_dict；原 checkpoint 可 strict=True 加载，但修改后的含填充输入输出可能变化。
- 本版是 **masked max pooling**，不是 masked BatchNorm。训练态 BatchNorm 的统计量仍可能受填充数量影响，因此不能宣称训练过程完全与填充数量无关。评估态的补零不变性另有测试。
- 默认零哨兵不能区分“真实的五维全零点”和“填充点”；如有这种数据，或增强会改变填充值，应在外部保存并传入显式 mask。归一化后真实点恰好五维全零也有同样限制。
- 未更改采样、时间分段、蒸馏损失、优化器、交叉验证或 Teacher。旧脚本中的其他设计选择不属于本次审核。
- 本地修改不等同于已核实服务器历史掩码实现，也不会追溯改变已报告的实验结果。若要用本版产生新结果，需重新运行对应实验。
