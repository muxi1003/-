# YOLO11n-Pose + CBAM 注意力消融实验

## 结构定义

当前基线确实是 `YOLO11n-Pose`，不是 YOLOv8。YOLO11 主干的 P5 端原本已经包含 `C2PSA`，因此本实验的准确表述是：

> 在 YOLO11n-Pose 的 P3/8、P4/16、P5/32 三个关键点检测输入特征上增加 CBAM 通道-空间注意力，形成多尺度特征重标定。

脚本同时提供只作用于小目标高分辨率特征的 `cbam-p3`，用于判断全尺度注意力是否造成不必要的开销或优化干扰。

实现不会修改安装环境中的 Ultralytics 源码。CBAM 使用 Ultralytics 8.4.40 自带模块，包裹第 16、19、22 层，保持原始层编号与 Pose Head 输入尺寸不变。预训练权重先载入，再初始化 CBAM，因此原 YOLO11n-Pose 参数能够完整迁移。

标准 CBAM 的两级 sigmoid 门控在零附近会把特征幅值缩小到约 1/4，不适合直接插入预训练模型。本实现把通道和空间门控初始化为 0.5，并增加初值为 4 的可训练逐通道 1x1 校准层，使 CBAM 插入时严格等价于恒等映射；这能避免注意力模型仅因初始特征幅值被破坏而在消融中处于劣势。

## 当前实测结果（3 轮方向性实验）

三组模型均从同一个 `yolo11n-pose.pt` 开始，在完整的 5991 张训练图像上训练 3 轮，并在同一批 1497 张验证图像上复算。以下结果只能用于筛选结构，不能替代正式论文实验。

| 变体 | 参数量 | GFLOPs | Pose mAP50-95 | Box mAP50-95 | 批量 1 延迟 ms/图 |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLO11n-Pose 基线 | 2,654,173 | 6.5598 | 0.994805 | 0.869209 | 10.535 |
| CBAM-P3 | 2,658,495 | 6.5660 | 0.994652 | **0.874871** | 10.890 |
| CBAM-P3/P4/P5 | 2,741,379 | 6.6335 | **0.994946** | 0.864295 | 11.416 |

与基线相比，`CBAM-P3` 仅增加 4,322 个参数（0.163%）和 0.0062 GFLOPs（0.095%），Box mAP50-95 提高 0.00566，但 Pose mAP50-95 降低 0.00015。`CBAM-P3/P4/P5` 增加 87,206 个参数（3.29%）和 0.0737 GFLOPs（1.12%），批量 1 的结构延迟增加约 8.4%，且 Box mAP50-95 下降 0.00491。验证流程记录的推理时间约为 1.85--1.99 ms/图，多次运行的波动与变体差异相近，不应据此声称 P3 提速。当前应保留 `CBAM-P3` 作为正式消融候选，淘汰全尺度版本；现有证据尚不能声称注意力已经提高关键点识别精度。

Pose AP 已接近 0.995，存在明显天花板效应。正式评价还应增加关键点像素误差或归一化平均误差（NME），并把两套关键点模型分别接入相同的温度提取与呼吸率流程，比较 RR R²、MAE、RMSE 和呼吸次数准确率。

实测明细：

- `runs/pose_attention/full3_seed0_p3_comparison.csv`
- `runs/pose_attention/full3_seed0_comparison.csv`
- `runs/pose_attention/profile_comparison.csv`

## 运行命令

先比较结构成本和批量为 1 的推理速度：

```powershell
E:\real\anaconda\envs\plant_gpu\python.exe scripts\yolo11_pose_attention_experiment.py profile
```

先做一次小数据冒烟训练，验证梯度、损失、验证和权重回载：

```powershell
E:\real\anaconda\envs\plant_gpu\python.exe scripts\yolo11_pose_attention_experiment.py train --variant cbam-p345 --smoke --name cbam_p345_smoke
```

短轮数消融建议显式使用 `--close-mosaic 0`，避免默认“最后 10 轮关闭 Mosaic”在总轮数不足 10 时覆盖全部训练过程。

正式公平消融必须让两组使用相同数据、初始权重、轮数、种子和超参数：

```powershell
E:\real\anaconda\envs\plant_gpu\python.exe scripts\yolo11_pose_attention_experiment.py train --variant baseline --epochs 100 --seed 0 --name baseline_seed0
E:\real\anaconda\envs\plant_gpu\python.exe scripts\yolo11_pose_attention_experiment.py train --variant cbam-p3 --epochs 100 --seed 0 --name cbam_p3_seed0
```

全尺度版本可作为反向消融，使用 `--variant cbam-p345 --name cbam_p345_seed0`。

训练后统一复算验证指标与速度：

```powershell
E:\real\anaconda\envs\plant_gpu\python.exe scripts\yolo11_pose_attention_experiment.py compare `
  --baseline-weights runs\pose_attention\baseline_seed0\weights\best.pt `
  --attention-weights runs\pose_attention\cbam_p3_seed0\weights\best.pt `
  --attention-variant cbam-p3
```

## 论文判定标准

主要精度指标应使用 `Pose mAP@0.5:0.95`，同时报告 `Pose mAP@0.5`、参数量、GFLOPs、验证集推理时间和批量为 1 的 FPS。只有注意力模型在相同设置下稳定提高关键点指标，且速度代价可接受，才能写成有效改进。

当前 `images/train` 与 `images/val` 是按帧划分，可能让同一视频的相邻帧同时出现在训练集和验证集。该划分适合先验证代码，但论文中的泛化结论应再使用按视频或按牛只分组的独立验证，并至少运行 3 个随机种子报告均值和标准差。

P2 小目标头、自适应 ROI 和时频联合 RR 的 73 视频消融结果见 `docs/yolo11_p3_adaptive_roi_time_frequency_p2_ablation.md`。这些变体均未超过默认 `paper_repro`，因此没有接入默认流程。
