# 全部已用视频的 YOLO11n-Pose 重训准备

状态：**只完成数据准备；尚未重训 YOLO，也没有产生新的温度曲线或 RR 结果。**

## 成员与去重口径

| 来源 | 片段 | 帧 | 说明 |
|---|---:|---:|---|
| 73 个历史短视频 | 73 | 7,488 | 复用现有 JPG 和 YOLO-Pose TXT；旧 train/val 逐帧混分仅作来源，不沿用其分组 |
| 林甸 anchored30 | 49 | 12,741 | 复用 49 个 MP4 对应的现有 JPG；额外 `bs1479` 帧目录不在 49 个 MP4 中，未纳入 |
| P1 | 8 | 2,114 | 原 MP4 指定 30 秒源帧转为 PNG |
| P2 | 16 | 4,227 | 同上；R1 的 6 窗是复核别名，不重复切帧 |
| P3 | 53 | 13,972 | A/B 两批，同上 |
| R3 久福 | 12 | 3,099 | 原 MP4 首 30 秒源帧转为 PNG；R3 另 12 个林甸窗已属于 anchored30 |
| **合计** | **211** | **43,641** | 去除了 R1/R3 的清单重复项；不同文件可能仍有重叠画面，不等于 211 份独立原始录像 |

这里的“全部视频”按先前实验**实际用过的片段/窗口**定义：73 个历史短片、49 个 anchored30、P1/P2/P3 的冻结 30 秒源帧范围及 R3 的 12 个久福首 30 秒。P1/P2/P3 所在的原始长录像在窗外的帧没有被纳入；R1 不另外生成相同的 P2 帧。

待人工核对帧 36,153 张，其中 12,741 张是已存在的 anchored30 JPG，23,412 张是此次从 P1/P2/P3/R3 源视频切出的 PNG。现有 7,488 个 TXT 中 7,485 个非空、3 个为空，均通过标签格式检查；**格式通过不代表解剖位置准确，空文件的生物学状态也未由本次复核**。

## 划分与用途

- 种子 `20260929`，以`牧场 + 文件名牛号的数字部分`分组，整组进入 train 或 val；不是把相邻帧随机分到两边。同号跨片段保守合并，真实耳标和跨场身份仍须核实。
- train 34,842 帧，val 8,799 帧（20.16%）；林甸 116/29 牛号组、久福 10/2 牛号组。`groups_80_20.csv` 是冻结分组，`segments.csv` 和 `frames.csv` 给出每段/帧的来源及分配；源视频哈希在 `segments.csv`，P1/P2/P3/R3 的逐帧时间在 `frames.csv`，旧 73/49 帧未补造逐帧时间。
- 检查确认同一牛号组不跨 train/val；按 SHA-256 完全相同的源视频字节也不跨两边。这个检查不能发现重编码后内容相同但哈希不同的片段，故仍以牛号分组为主要保护。
- `reference_aliases.csv` 列出 R1→P2 的 6 行及 R3 林甸→anchored49 的 12 行，方便核查去重；这是按参考清单和源帧范围去重，不声称跨 73/49 的内容重复已做逐帧比对。
- 该 val 仅供**新 YOLO 的分组内定位验证**，不是未接触的 RR 方法确认集，也不是跨牧场外测。P3 的既有 RR 结果已被查看并用于失败归因，更不能重新称为未见的方法测试。历史 `models/best.pt` 的逐图训练成员不能完整核实，不能以这次 val 直接作它与新模型的公平定位增益结论。
- 后续若比较“整曲线质量选择＋约束峰规则”，必须锁定同一个 YOLO 权重、温度映射、窗口、真值和可评分集合做四组配对。若比较旧/新 YOLO，还须以同一新划分重新训练基线权重；改变 YOLO 后 RR 变化本身不能证明信号处理创新。

## 当前文件与命令

使用 `E:\real\anaconda\envs\plant_gpu\python.exe` 运行本目录脚本：

```powershell
python prepare_corpus.py inventory
python prepare_corpus.py extract
python prepare_corpus.py verify
python prepare_corpus.py aliases
python review_gate.py init
python review_gate.py check
```

`inventory`/`init` 一次性创建冻结清单及人工复核表，已有文件时拒绝覆盖；`extract` 逐段支持续跑，写出 `corpus_v1/extraction_complete.json` 才表示所有新帧切出。`verify` 检查成员、路径、分组与逐段完成记录。源文件未改动，新增 PNG 保存视频解码像素，避免再次 JPG 压缩影响颜色测温。

源视频按帧号**从头顺序解码**后截取，不对中途窗口直接随机 seek。抽样对比顺序解码源帧与 PNG 像素完全相同；随机 seek 在某个中途窗口得到不同像素，不能代替这个检查。历史 JPG 仍有原有压缩，不声称全部输入都无损。候选预训练权重 `yolo11n-pose.pt` 的 SHA-256 为 `869e83fcdfdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0`；历史 `models/best.pt` 为 `b9c318b33323a2f3887bd786b3997a8c6cf6c2098ec8fdf0d1f9b91ad3b5fbe7`。这两个哈希只标识文件版本，不证明训练成员。

`corpus_v1/manual_review_status.csv` 是每张**新帧**的待办表。人工逐帧检查鼻部框与左右鼻孔关键点后，在 `manual_label_path` 写 YOLO-Pose TXT，并填写 `review_status=approved`、`reviewer`、`reviewed_at`；鼻部确实不可见时也要逐帧确认，建立**空 TXT**，不能用“文件缺失”表示阴性。非空行格式与原标签一致：`class x y w h left_x left_y left_v right_x right_y right_v`，坐标归一化，`class=0`，`v` 为 0/1/2。自动建议可以辅助标注，但未人工复核的伪标签不得标为 approved。

只有 `review_gate.py check` 对全部 36,153 帧返回通过后，才运行 `python review_gate.py materialize --name yolo_dataset_v1` 组装按牛分组的数据集；它不覆盖已有数据集。之后先运行 `python train_grouped_pose.py --preflight-only`，再运行 `python train_grouped_pose.py --name yolo11n_pose_grouped_v1`。后者从哈希固定的 `yolo11n-pose.pt` 开始，在新目录训练，不覆盖旧权重；本机 Ultralytics 8.4.40 的设置目录由脚本隔离到本实验目录。旧 `train_yolov8n-pose.py` 指向消失的旧 YAML 且设置 `exist_ok=True`，**不要直接运行**。训练完成后还须按冻结 RR 流程对每段推理并画曲线；当前尚未到这一步。
