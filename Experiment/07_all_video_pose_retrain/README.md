# 全部已用视频的 YOLO11n-Pose 重训准备

状态（2026-10-17）：**训练及四组RR复放均已完成：135轮正常早停，最佳模型来自85轮；199片段全部复放，12:27写入complete与COMPLETED.json。**

实际输出：199份温度CSV（40542帧）、四组各199预测共796行、796份曲线CSV、199张PNG；权重在`training_runs/yolo11n_pose_lindian_20261016_v1/weights/best.pt`，结果在`replay_lindian_20261016_v1/metrics.csv`、`paired_intervals.csv`及`plots/`。200轮是上限，不是必跑轮数；训练结束后进程正常退出，当前没有PID不代表再次中断。

原始结果并非全部改善：47完整窗C0P0/C1P1的MAE为3.064594/2.850282，配对95%区间跨0，完全计数16/47降至13/47；P3二十四窗MAE为4.666667/5.416667，合并候选退化。详细原始汇总与边界已记入项目总记录N096，暂不替换默认模型或RR方法；计算完成不代表论文创新已被独立证实。

## 指标横向对照（N097）

更新（N098）：优先查看[含逐项合计与输入核对的v3总表](metric_tables_20261017_v3/metric_comparison.xlsx)及[完整说明](metric_tables_20261017_v3/指标对照说明.md)。`按指标_all/train/val`每项指标最后一行为155/127/28条记录的合计，方法各占一列；另有`仅约30秒窗`表（all82、train69、val13）。合计从逐窗预测重算，不将五队列MAE或R²简单平均。

**人工复核的30秒帧确实已使用，不是未导入标注。** 用户指定复核表中的33054个林甸approved新帧，与本次非S73复放帧键完全一致；包含anchored49的12741帧以及P1/P2/P3。目录`lindian_anchored30_frames`实有50个子目录，本次49个纳入，额外bs1479未纳入。与此同时，准备脚本另外保留旧73的7488帧，故同ID可有S73和A49两条分支：16170075分别124和260帧；A49训练图像/标签与来源哈希一致，biaozhu中相应260份JSON已使用。详见[input_membership_reconciliation.json](metric_tables_20261017_v3/input_membership_reconciliation.json)。不能把155条混合时长记录改称155个全新30秒窗，也不能根据R²高低反推来源或把旧人工次数移给延长窗。

用户提供PPT已实际读取：第4页明确列73为6–20秒、49为约30秒，第12页历史R²为0.946595；它与本次新YOLO、原文规则/质量选择/峰规则四组复放不是同一结果快照。以上是文件和程序输入核对，不是凭旧聊天记忆判定。v1及中间v2保留历史，v3新增明确“30秒复核数据已用、同时另保留旧73”的解释。

RR“准确率”在v3同时列**平均相对准确度**（原逐窗max(0,1-相对误差)定义）和**完全计数率**，不与事件F1或YOLO mAP混同。3.790651是五队列MAE等权宏平均，155条记录的对照MAE实际为3.916826；合并R²由155对参考/预测重算。事件F1合计仅82个有事件参考的窗口，不冒称155窗。

只纳入本轮约30秒的YOLO验证组13窗时：C0P0/C1P0/C0P1/C1P1的MAE为2.000103/1.384905/2.922267/2.461463，平均相对准确度95.36%/96.54%/94.22%/95.34%。质量选择相对原文规则的配对MAE区间[-1.385131,0.154154]仍跨0。含旧73的混合验证28窗则MAE3.697621到3.039970，区间[-1.110153,-0.230666]，属于事后内部正向证据，而非未见RR确认；[主张边界](metric_tables_20261017_v3/CLAIMS_FROM_RESULTS.md)保持候选贡献、部分支持，不据此自动采用或称稳定创新已证实。

v3导出命令在原命令基础上加`--include-pooled`，并使用不存在的新输出目录；新增`pooled_rr_metrics.py`负责逐窗聚合、牛号聚类配对区间及输入核对。4项针对加权/重新计算R²、事件分母、跨队列同牛聚类、空值的测试通过，另回读核对720个四组表格单元格与每项末行合计位置；原模型、训练集、标注及冻结预测未改。以下v1链接保留追溯。

- [Excel指标总表](metric_tables_20261017_v1/metric_comparison.xlsx)：13张表，含最佳轮YOLO验证、四种方法各占一列的RR全部/train/val、相应配对区间、199窗四组预测、原始汇总、135轮训练记录及来源哈希。
- [完整中文指标表](metric_tables_20261017_v1/指标对照说明.md)：分队列列出MAE、RMSE、R²、完全计数、平均计数准确度、TP/FP/FN/F1及适用口径；原始精度保存在Excel/CSV。
- [逐窗横向CSV](metric_tables_20261017_v1/window_predictions_wide.csv)：每个片段一行，四组预测/误差/选曲/输出状态分别成列。
- [复算验证记录](metric_tables_20261017_v1/verification.json)：56行指标、42组配对区间及9356行事件匹配计数核对通过；来源文件未改。Excel为无公式的导入快照，不是在线计算模型。

最佳轮85的鼻部框Precision/Recall/mAP50/mAP50-95为0.995710/0.995980/0.994900/0.851560；鼻孔关键点为0.986730/0.985220/0.984320/0.984300。已核对best.pt保存指标与results.csv第85轮一致；不是第135轮指标，也不代表呼吸检测达到相同准确率。

以后每次结果沿用“队列分开、指标作行、方法各一列”的格式，并另存版本目录，不覆盖本轮或历史结果。可复用导出脚本：

```powershell
& 'E:\real\anaconda\envs\plant_gpu\python.exe' 'E:\real\use_code\yoloV8\Experiment\07_all_video_pose_retrain\export_metric_tables.py' --output 'E:\real\use_code\yoloV8\Experiment\07_all_video_pose_retrain\metric_tables_新版本'
```

导出前自动核对来源、逐窗RR及事件计数，重建既有bootstrap区间，校验最佳轮和权重哈希；已有输出目录时拒绝覆盖。新实验需同时显式指定对应`--replay`、`--training`、`--dataset`，不能只改输出目录就当作新的实验结果。所有当前RR结果均属开发复放，train/val仅表示YOLO牛号分组；P2验证组没有完整参考，P1/P3验证组各仅1个可评分窗，R²不填0。

活动流水线：`pipeline_logs/yolo11n_pose_lindian_20261016_v1/status.json`；原`training.log`保留，恢复日志在其`resumes/20261016_153008/resume_training.log`。新监督PID868、子进程PID20640（启动时快照，后续以实时查询为准），原35204/18804已退出。训练输出仍为`training_runs/yolo11n_pose_lindian_20261016_v1`，完成后自动衔接原冻结RR复放。`status.json=complete`且复放目录有`COMPLETED.json`才表示整条流程完成。

当前状态不能只看保存的stage：使用`python inspect_pipeline.py pipeline_logs/yolo11n_pose_lindian_20261016_v1/status.json`核对进程和30秒心跳。新状态包含`heartbeat_at`；父/子进程已退出时返回interrupted，存活但心跳超过120秒则返回stale_heartbeat。程序无法在自身被强制终止后更新文件，所以原始JSON中的stage仍必须结合实时查询解释。

中断记录见`interruption_20261016_151825.json`。停止与Codex应用更新、沙箱服务重启时间重合，属于有事件支持的原因推断，不是已捕获的退出调用栈。恢复前会完整备份last.pt、best.pt、results.csv及args.yaml，恢复优化器/EMA/AMP及轮次，未保存的第29轮重跑；训练上限仍200轮，不是额外再跑200轮，不承诺随机序列逐位相同。不会修改系统更新设置，也不能保证关闭或更新Codex后后台进程仍存活。

## 本轮完整接收

- `annotation_final_20261016_v1/summary.json`：33054个新增标签全部通过并导入，原始JSON未改；33000份原JSON保存在`original_json.zip`。R3久福3099帧仍pending，不阻塞本轮林甸。
- 用户已放回`bs177611`的260份JSON；`200713`前54帧由用户明确确认鼻部不可见，生成空TXT并登记到`annotation_confirmation_20261016.json`，没有伪造原JSON。5777个空标签包含这54帧，其他空标签均来自明确空shapes JSON。
- 新提交的3帧nose框越界不足1像素，独立转换副本仅把矩形裁到图像边界，最大0.72414像素；原JSON和鼻孔点均不改，逐坐标日志见`box_boundary_clips.csv`。严格转换器默认行为未放宽。
- 导入前后的复核CSV均保留；`reviewed_at`记录本次用户确认/接收日期，不冒充逐帧人工操作时间。旧7488帧仍注明历史标签来源，未声称本轮重做人工作业。
- 数据集为`yolo_dataset_lindian_20261016_v1`，199片段40542帧，train32226/val8316，牛号组116/29；不是逐帧随机拆分。
- CUDA矩阵及YOLO真实前向检查通过；独立按文档复验输出`WITNESS (1, 56, 8400)`。预检发现旧权重校验串漏抄一个f，已按实际文件修正，不是权重文件发生变更。

`replay_lindian_20261016_v1/`已冻结同一人工参考和帧时间绑定：73短片独立次分析，anchored49的47个完整窗、P1的5窗、P2的6窗、P3的24窗可评分。部分参考及两段事后口头36次不进入完整RR主评分，R1分歧不静默替换旧P2参考。

`run_lindian_pipeline.py`以独立日志顺序执行：训练门禁 → YOLO11n-Pose训练（200轮上限、早停50、batch8、640、seed20260929）→ best.pt重推40542帧 → 固定BGR随机森林、半径20、置信度0.5、无缺口重取 → 同温度输入四组C0P0/C1P0/C0P1/C1P1 → 曲线、逐窗预测、事件匹配、分队列/分train-val指标与按牛号组bootstrap的配对区间。状态为failed时立即停止，不覆盖旧模型或自动换阈值。只有训练和RR实际完成后才能报告新性能；全批仍是开发复放，不称未见RR确认或完整Chen端到端复现。

以下为历史准备与部分接收记录，保留追溯。

2026-10-14当前安排：用户使用X-AnyLabeling辅助后逐帧检查并修正，部分JSON已存入`biaozhu/`；继续完成林甸，其余R3久福12片段暂缓。本轮保留原牛号分组，纳入199片段/40542帧（train32226、val8316），其中33054帧需要新增人工复核标注，7488帧沿用历史73标签来源。R3久福3099帧单列暂缓。用户继续在`biaozhu/`保存JSON，完成后统一转换和登记人工复核来源，无需手工重复填写33054行CSV。

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
python review_gate.py prepare-label-dirs
python review_gate.py check
```

当前X-AnyLabeling提交用`python audit_xanylabeling_submission.py`生成独立进度核验，入口为`annotation_intake_20261014_v1/README.md`，逐片段进度见`segment_progress.csv`，缺JSON见`missing_frames.csv`，结构冲突见`annotation_issues.csv`。继续补标后以`--output`指定新的版本目录重审，保留旧快照。JSON的`checked=false`保留为软件字段，不否定用户逐帧复核说明；`converted_labels/`目前只用于转换核验，尚未进入训练。

本次实查：24030份JSON中23770份匹配、23760帧结构通过、10帧冲突；缺9284份JSON（P3的35未提交片段9230帧及200713前54帧）。完整且结构通过片段83/126，另有3471份明确空JSON。具体补标和10帧修正要点见`annotation_intake_20261014_v1/修正与补标清单.md`；范围外260份来自`bs1479`，原件保留。

`inventory`/`init` 一次性创建冻结清单及人工复核表，已有文件时拒绝覆盖；`extract` 逐段支持续跑，写出 `corpus_v1/extraction_complete.json` 才表示所有新帧切出。`verify` 检查成员、路径、分组与逐段完成记录。源文件未改动，新增 PNG 保存视频解码像素，避免再次 JPG 压缩影响颜色测温。

2026-10-14修正：原 `init` 只把 `manual_label_path` 写进CSV，漏建标签目录。已补齐 `corpus_v1/manual_labels` 及138个片段子目录，并核验全部36153条路径的父目录存在；当前人工TXT文件仍为0。现在 `init` 自动创建这些目录；已有复核表时用 `prepare-label-dirs` 补建，保留原表及已有标签，不创建空TXT占位。`check` 分别报告目录缺失、标签文件数和人工审核状态。

源视频按帧号**从头顺序解码**后截取，不对中途窗口直接随机 seek。抽样对比顺序解码源帧与 PNG 像素完全相同；随机 seek 在某个中途窗口得到不同像素，不能代替这个检查。历史 JPG 仍有原有压缩，不声称全部输入都无损。候选预训练权重 `yolo11n-pose.pt` 的 SHA-256 为 `869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0`（2026-10-16重新实测，旧记录漏抄一个f，训练预检已同步修正）；历史 `models/best.pt` 为 `b9c318b33323a2f3887bd786b3997a8c6cf6c2098ec8fdf0d1f9b91ad3b5fbe7`。这两个哈希只标识文件版本，不证明训练成员。

`corpus_v1/manual_review_status.csv` 是每张**新帧**的待办表。人工逐帧检查鼻部框与左右鼻孔关键点后，在 `manual_label_path` 写 YOLO-Pose TXT，并填写 `review_status=approved`、`reviewer`、`reviewed_at`；鼻部确实不可见时也要逐帧确认，建立**空 TXT**，不能用“文件缺失”表示阴性。非空行格式与原标签一致：`class x y w h left_x left_y left_v right_x right_y right_v`，坐标归一化，`class=0`，`v` 为 0/1/2。自动建议可以辅助标注，但未人工复核的伪标签不得标为 approved。

本轮完整林甸JSON接收、转换并登记复核状态后，用`python review_gate.py check --scope lindian`检查33054帧，R3久福不影响此范围的完成度。通过后运行`python review_gate.py materialize --scope lindian --name yolo_dataset_lindian_v1`，再执行`python train_grouped_pose.py --dataset yolo_dataset_lindian_v1 --preflight-only`和正式训练命令`python train_grouped_pose.py --dataset yolo_dataset_lindian_v1 --name yolo11n_pose_lindian_v1`。训练脚本从READY记录读取范围，从哈希固定的`yolo11n-pose.pt`开始并保存独立权重；温度映射、固定窗口及四组峰处理规则按原协议重跑曲线和RR。默认`--scope all`仍要求含R3久福的36153帧。当前未启动训练或RR重跑。
