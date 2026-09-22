# 奶牛鼻孔热红外呼吸检测（Cow RR）

从奶牛**鼻孔热红外视频序列**估计**呼吸次数**与**呼吸率（RR, 次/分）**：复现原论文方法，分析头部运动与信号缺失造成的计数错误，并推进巡检小车的现场采集与部署。

> **关于目录名**：仓库目录 `yoloV8` 是历史命名。**当前训练网络为 YOLO11n-Pose，不是 YOLOv8**。使用任何权重前请核对实际结构与哈希。
>
> **关于状态**：本仓库是进行中的科研工程，不是已完成的成品。权威现状、指标台账与全部分歧记录见
> [`docs/cow_rr_project_log.md`](docs/cow_rr_project_log.md)（项目总记录，按 N 编号持续维护）。

---

## 1. 方法流程

```
热红外视频
  └─ 抽帧                    pre_data/split_thermal_video.py
      └─ labelme 标注         1 个鼻子框 + 2 个鼻孔关键点（左/右）
          └─ 格式转换         pre_data/labelme2YOLO_batch.py | labelme2YOLO_single.py
              └─ 划分数据集    pre_data/all_splitDataset.py | splitDataset.py
                  └─ 训练姿态模型   train_yolov8n-pose.py → YOLO11n-Pose
                      └─ RGB → 温度映射   temperature_extraction/getRandomForestRegress/
                          └─ 双侧鼻孔温度序列   temperature_extraction/single_nose/
                              └─ 曲线融合 + 寻峰   draw_curve/curve.py | curve_g.py
                                  └─ 呼吸次数 / 呼吸率 RR
```

标注规范（见 [`dataset.yaml`](dataset.yaml)）：

| 项 | 值 |
|---|---|
| 类别数 `nc` | 1（`nose`） |
| 关键点 `kpt_shape` | `[2, 3]`（左右鼻孔，x/y/visibility） |
| 左右翻转索引 `flip_idx` | `[1, 0]` |

分步操作说明见 [`readme.txt`](readme.txt)（原始操作手册）。

---

## 2. 目录结构

| 路径 | 内容 |
|---|---|
| `scripts/` | 实验与分析主脚本（100+ 个，含消融、外测、审计、报告生成） |
| `pre_data/` | 抽帧、labelme→YOLO 转换、数据集划分、标注查看 |
| `draw_curve/` | 温度曲线融合、平滑、寻峰与 RR 计算 |
| `temperature_extraction/` | 随机森林 RGB→温度回归、单鼻孔温度提取 |
| `tests/` | 单元测试（姿态注意力、RR 运动/鲁棒/频谱等） |
| `docs/` | 研究文档、实验方案、论文草稿与**项目总记录** |
| `Experiment/` | 三部分证据补强实验包（消融 / 冻结独立测试 / 可靠事件参考），入口见其 [`README.md`](Experiment/README.md) |
| `sources/` | 文献调研笔记 |
| `literature_extract/` | 文献要点提取 |
| `dataset/labels/` | YOLO 格式标注（`dataset/images1.csv` 为图像索引） |
| `models/`、`runs/`、`Dataset_new/` | **未纳入本仓库**（模型权重、训练输出、数据集），见第 5 节 |

根目录：`train_yolov8n-pose.py`（训练）、`change_pt_to_onnx.py`（导出 ONNX）、`dataset.yaml`、`readme.txt`。

---

## 3. 数据与真值（重要约束）

项目使用多套**不可混用**的数据与分析集，引用任何数字前请先确认口径：

| 分析集 | 说明 |
|---|---|
| 73 视频复现集 | 复现原论文；不同快照的 R² 不可互相替代 |
| 林甸 49 固定窗 | 约 30 秒片段；标注为 39 `completed` + 10 `uncertain` |
| 久福 271 冻结窗 | 跨牧场外测；每个原视频取**首 30 秒**，不挑最好片段 |

**必须遵守**（摘自项目总记录第 1.2 节）：

1. 不同结果快照的 R²（如 `innovation_repro` 与 `paper_repro_metrics.csv`）属于不同基线，需与各自的冻结真值、时长和预测比较，**不能只挑较大的数字**。
2. `truth-calibrated` 利用人工参考的诊断结果**不能替代**独立预测主结果。
3. 人工计数、既有模型与历史指标不得擅自更改；修改后须另存版本并重算受影响结果。
4. 未实测的环境与质量信息不编造；历史日期/地点属于不同证据层级。

指标台账（含基线编号、样本数、真值与时长版本）见项目总记录**第 3 节**。

---

## 4. 环境与运行

主要第三方依赖（由代码 import 统计得出，仓库未附 `requirements.txt`）：

```
python >= 3.9
ultralytics      # YOLO11n-Pose 训练与推理
torch
opencv-python    # cv2，视频抽帧与图像处理
numpy
pandas
scipy            # 信号滤波与寻峰
scikit-learn     # 随机森林温度回归
joblib           # 模型序列化
matplotlib       # 曲线与诊断图
tqdm
```

快速开始：

```bash
# 1) 训练姿态模型（需先准备 Dataset_new/ 数据集）
python train_yolov8n-pose.py

# 2) 温度映射模型
python temperature_extraction/getRandomForestRegress/RGB_nihe.py

# 3) 提取双侧鼻孔温度序列，再计算呼吸曲线与 RR
python draw_curve/curve.py
```

模型推理（示例）：

```bash
yolo pose predict model=<best.pt 路径> source=<视频路径>.mp4 show=True
```

---

## 5. 仓库内容边界

本仓库**只纳入源码、配置、文档与轻量结果（CSV/JSON/MD/HTML）**，体积约 287 MB。以下内容按 [`.gitignore`](.gitignore) 排除，**未上传**：

| 排除内容 | 原因 |
|---|---|
| `Dataset_new/`、`dataset/images1/`、`*.jpg`、`*.png`、`*.mp4` | 数据集、视频与逐帧图像（数 GB～数十 GB） |
| `models/`、`runs/`、`*.pt`、`*.pkl`、`*.onnx` | 模型权重，可用脚本重新训练生成 |
| `outputs/`、`Experiment/tools_deps/`、`*.exe`、`*.zip` | 运行输出与打包产物 |
| `.conda/`、`__pycache__/` | 本地环境与缓存 |
| `.aris/`、`.agents/`、`.codex/`、`.codex_tmp/`、`.claude/` 等 | 本地 AI 工具链与技能库，非项目内容 |
| `.env`、`*.pem`、`id_ed25519*` | 本地凭据，禁止提交 |

因此克隆后**无法直接复现完整实验**：需要自备数据集与训练权重。这也意味着本仓库适合作为**代码与文档的版本管理载体**，而不是数据分发渠道。

---

## 6. 关键文档入口

| 文档 | 用途 |
|---|---|
| [`docs/cow_rr_project_log.md`](docs/cow_rr_project_log.md) | **项目总记录**：当前状态、约束、指标台账、N 编号全过程 |
| [`Experiment/README.md`](Experiment/README.md) | 三部分证据补强实验包入口与采用决定 |
| [`docs/thermal_rr_reproducibility_readme.md`](docs/thermal_rr_reproducibility_readme.md) | 复现说明 |
| [`docs/thermal_rr_external_validation_protocol.md`](docs/thermal_rr_external_validation_protocol.md) | 外部验证协议 |
| [`docs/thermal_rr_manuscript_draft.md`](docs/thermal_rr_manuscript_draft.md) | 论文草稿 |
| [`docs/rr_robot_stop_and_measure_plan_20260907.md`](docs/rr_robot_stop_and_measure_plan_20260907.md) | 巡检小车到站停车测量方案 |
| [`docs/rr_robot_ros2_flir_edge_pro_integration_20260907.md`](docs/rr_robot_ros2_flir_edge_pro_integration_20260907.md) | ROS 2 + FLIR ONE Edge Pro 集成设计 |

---

## 7. 硬件与采集

- 热相机：**FLIR ONE Edge Pro**（沿用原论文设备）
- 巡检小车：已有导航，使用 **ROS 2**；车载电脑与部署性能尚未确认
- 首版采用**到站停车测量**；ROS 接入、手机数据桥接与车载实测仍属设计任务

数据采集地点：林甸县牧场（2023-08-05 至 2023-08-10）、久福牧场。

---

## 8. 说明

- 本项目为科研用途，仓库中的实验结果处于持续修订状态，**引用具体指标前请以项目总记录第 3 节为准**。
- 未经确认的环境信息、设备能力与实验数据不作推断；历史日期不明确的节点在总记录中标注为「补录」。
