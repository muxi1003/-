# Thermal RR 投稿策略与二区以上期刊准备清单

更新日期：2026-07-04

本文档用于把当前热红外奶牛呼吸频率创新实验推进到二区以上期刊投稿。注意：JCR 分区、中科院分区、影响因子和期刊政策每年都会变化，正式投稿前必须再次核对当年最新版 JCR/中科院分区和期刊官网作者指南。本文档中的期刊排序是基于主题适配度和当前实验成熟度的投稿策略，不替代正式分区核查。

外部验证与元数据采集的执行协议见 `docs/thermal_rr_external_validation_protocol.md`。该协议定义了当前 73 个视频内部结果的证据边界、必须补齐的元数据字段、真实 `cow_id`/日期/相机分组验证、外部测试集冻结规则，以及 `truth-calibrated` 只能作为上限分析的写作边界。正式声称“二区以上投稿 ready”前，应先按该协议让 readiness 审计从 `not_ready` 变为 `ready`。

文献驱动的创新优先级矩阵见 `docs/thermal_rr_literature_innovation_matrix.md`。当前最稳妥的主创新是质量感知残差峰数校正；已实现的下一层提精度候选是 FFT/自相关/峰值计数的保守信号共识补充实验；提升论文意义的关键仍是 THI/ATHI、真实 `cow_id` 分组和外部测试集。

## 1. 当前稿件状态判断

当前稿件已经具备“算法创新雏形 + 可复现结果 + 论文图表 + 英文草稿”的基础，但还没有达到强二区投稿证据。主要优势是默认流程已经形成较强基线，质量感知残差校正能在内部验证中提升 RR R2、MAE 和精确计数。主要短板是数据量只有 73 个视频，bootstrap 置信区间仍跨过 0，真实 `cow_id/date/scene/camera_id` 分组元数据缺失，THI/ATHI 环境解释字段缺失，外部测试集缺失。

新增的信号共识补充实验进一步提高了内部候选精度：固定 out-of-fold `RR R2 = 0.946345`、`MAE = 1.4955 bpm`、`RMSE = 2.9512 bpm`、`exact count = 55/73`；nested CV `RR R2 = 0.940796`、`exact count = 53/73`；prefix-group fixed threshold `RR R2 = 0.936263`、`exact count = 53/73`。该结果可作为补充创新点，但不替代真实 `cow_id` GroupKFold 和外部测试集。

新增的 selective RR reporting 分析把信号共识结果转成部署复核机制：严格自动报告规则接受 13/73 个视频，内部 exact count = 13/13、`RR R2 = 0.999907`、`MAE = 0.0611 bpm`；score 阈值 0.35 接受 21/73 个视频，exact count = 19/21、`RR R2 = 0.987118`、`MAE = 0.4902 bpm`。该分析的论文价值在于“高置信自动输出 + 低置信人工复核”，不是新的总体精度主结果。

当前主结果来自：

- `Dataset_new/72video/al_images/paper_repro_quality_residual_paper_assets/paper_main_results_table.csv`
- `Dataset_new/72video/al_images/paper_repro_quality_residual_figures/rr_quality_residual_combined.png`
- `docs/thermal_rr_manuscript_draft.md`

当前最稳妥的论文主张应写成：

> A quality-aware residual peak-count correction framework improved internal validation performance of a thermal-infrared dairy-cow respiratory-rate pipeline, while grouped and bootstrap analyses indicated that further cow-level, scene-level, and external validation is required before claiming broad generalization.

不应写成：

> The method has solved dairy-cow respiratory-rate monitoring under all head-motion conditions.

也不应把 `truth-calibrated R2 = 0.999655` 写成主方法性能。

### 1.1 RR R2 写作口径

论文中应把不同 R2 结果分层报告，不能把所有数值放在同一证据等级上比较。

| 结果口径 | 当前 RR R2 | 可否作为主结果 | 论文中推荐写法 |
|---|---:|---|---|
| 默认端到端流程 | 0.928992 | 可以 | 作为当前复现基线或默认 thermal RR pipeline 性能。 |
| 质量感知残差校正，固定阈值 out-of-fold | 0.942885 | 可以，但要说明内部验证 | 作为主要创新结果之一，需同时报告 MAE、RMSE、exact count 和 bootstrap CI。 |
| 信号共识补充，固定阈值 out-of-fold | 0.946345 | 可以作为候选扩展 | 作为精度提升补充创新点，必须说明其仍是内部候选结果。 |
| 质量感知残差校正，nested threshold CV | 0.937336 | 可以，更保守 | 作为更稳健的主文或补充验证结果，强调阈值选择也在交叉验证内完成。 |
| 信号共识补充，nested threshold CV | 0.940796 | 可以作为保守候选验证 | 说明信号共识在嵌套验证下仍保留小幅收益。 |
| leave-one-prefix-group-out 训练组内选阈值 | 0.936263 | 可以作为内部分组外推验证 | 只能称为 prefix-group internal validation，不能称为真实牛只泛化。 |
| selective RR 严格自动报告子集 | 0.999907 | 不作为总体主结果 | 可作为 uncertainty-aware deployment triage，必须同时报告 coverage = 17.8%。 |
| truth-calibrated 上限 | 0.999655 | 不可以作为主结果 | 只能称为 truth-assisted upper bound、oracle analysis 或后验诊断结果。 |

主文最推荐报告的顺序是：默认流程、质量感知固定阈值 out-of-fold、信号共识固定 out-of-fold、nested threshold CV、prefix-group holdout、selective reporting triage。`truth-calibrated` 可以放在补充材料中，用于说明如果逐视频峰值参数能被理想选择，现有温度曲线本身接近人工计数；但因为该结果使用了参考真值辅助选择参数，不能证明算法在未知视频上的真实预测能力。

## 2. 候选期刊优先级

| 优先级 | 期刊 | 推荐程度 | 当前稿件适配度 | 主要风险 | 投稿前必须补强 |
|---:|---|---|---|---|---|
| 1 | Computers and Electronics in Agriculture | 首选冲刺 | 农业计算机视觉、PLF、算法系统最匹配 | 需要更强泛化验证和工程完整性 | 真实 GroupKFold、外部测试集、完整消融、代码/数据可复现说明 |
| 2 | Biosystems Engineering | 首选/并列 | 工程系统、农业生物系统、模型解释适配 | 只做算法后处理可能显得应用意义不足 | 补 THI/ATHI、场景元数据、误差来源和系统部署讨论 |
| 3 | Journal of Thermal Biology | 强主题备选 | 热红外、温度、生理热调节和热应激方向适配 | 如果缺 THI/ATHI，容易变成纯视觉算法，生理意义不足 | 必须补热应激解释或至少补环境温湿度和生理讨论 |
| 4 | Journal of Dairy Science | 高影响但高门槛 | 奶牛健康福利方向适配 | 动物科学数据设计、样本量、外部验证要求更高 | 需要 cow-level 元数据、实验设计、福利/热应激解释和更大样本 |
| 5 | Smart Agricultural Technology | 稳妥备选 | 智慧农业系统、应用型算法适配 | 分区和影响力需投稿前核查 | 保留完整工程流程、图表和应用场景说明 |
| 6 | Agriculture | 稳妥备选 | 已有相近热红外 RR 论文，主题容易匹配 | 创新强度可能不足以支撑高目标 | 强化相对已有 YOLOv8+鼻孔分割论文的差异 |

## 3. 推荐投稿路径

第一选择是把目标定位在 `Computers and Electronics in Agriculture` 或 `Biosystems Engineering`。这两个期刊更适合把本文写成一个完整的农业工程/PLF 算法系统，而不是单纯热生理现象论文。当前的质量感知残差校正、误差分类、分组验证、图表资产和复现脚本都服务于这个方向。

如果能在短期内补齐环境温湿度、THI/ATHI、头部运动评分、遮挡评分、鼻孔可见度评分，并补一个外部测试集，则可以继续冲刺 `Computers and Electronics in Agriculture`。如果只能补环境和生理解释，但外部测试集较弱，则 `Journal of Thermal Biology` 会更稳，因为文章主线可以转为热红外 RR 与热应激监测。若能补充真实 cow_id、日期、场景和动物科学解释，才建议考虑 `Journal of Dairy Science`。

## 4. 每个候选期刊的稿件改造重点

### 4.1 Computers and Electronics in Agriculture

稿件应强调算法系统创新、鲁棒性和可复现工程流程。标题可以保留 “Motion-Robust Infrared Thermography...” 或改成更工程化的 “Quality-Aware Signal Consensus for Thermal-Video Respiratory Rate Monitoring in Dairy Cows”。Introduction 应突出 PLF 中非接触生理监测、头部运动、遮挡、跨场景泛化和数据稀缺。Results 应把 Table 1、Figure 2、分组验证、selective reporting triage 和 bootstrap CI 放在核心位置。

投稿前必须补强：

- 至少一个真实分组验证：`cow_id` 或 `collection_date` GroupKFold。
- 最好新增 20-50 个外部测试视频。
- 增加 ablation：去掉 spectral features、去掉 endpoint-gap features、只用 baseline、只用 fixed threshold。
- 在 Methods 中明确所有训练/阈值选择过程，避免审稿人认为数据泄漏。

### 4.2 Biosystems Engineering

稿件应强调系统工程、误差来源、模型解释和农业生物系统意义。Compared with Computers and Electronics in Agriculture, this target can tolerate a slightly more applied system-analysis narrative, but it still requires rigorous validation. Discussion 应详细解释为什么 endpoint/boundary 错误、弱峰和噪声峰是热红外鼻孔 RR 的系统瓶颈。

投稿前必须补强：

- 将 `paper_error_taxonomy_table.csv` 写成正式误差机制分析。
- 增加典型曲线案例图，展示修正前后峰序列。
- 补充 heat-stress context readiness 或真实 THI/ATHI 分析。
- 明确该方法如何进入牧场部署流程：自动通过、高风险复核、低质量拒判。

### 4.3 Journal of Thermal Biology

稿件应从热红外和温度生理角度改写。当前稿件如果只报告算法 R2，主题可能偏离该刊核心。要提高适配度，必须把 RR 与热应激、环境温湿度、THI/ATHI 或热调节行为联系起来。该刊更关心温度如何影响动物生理，而不仅是检测算法。

投稿前必须补强：

- 填写 `ambient_temperature_c` 和 `relative_humidity_percent`，生成 `thi_analysis`。
- 按 THI 类别比较 RR 或 corrected RR。
- 讨论热红外鼻孔温度曲线如何反映呼吸热交换。
- 不要把文章写成纯 YOLO/随机森林工程论文。

### 4.4 Journal of Dairy Science

稿件需要更强动物科学设计。JDS 读者会关心牛只、日龄、泌乳阶段、产奶量、胎次、热应激状态、健康和福利意义。当前工作还缺这些变量，因此目前不建议直接投 JDS。

投稿前必须补强：

- cow_id、parity、days in milk、milk yield、health status 或 heat-stress labels。
- 明确人工 reference RR 的采集协议和可靠性。
- 按牛只或日期做严格外部/分组验证。
- 将技术指标转化为动物管理意义，例如热应激预警、复核队列或福利监测。

## 5. 投稿前强制检查清单

### 数据与验证

- [ ] `paper_metadata_template.csv` 中 `cow_id` 至少覆盖 73/73。
- [ ] 先查看 `paper_metadata_annotation_minimum_q2_checklist.md`，确认 P0/P1 必填字段和证明命令。
- [ ] 使用 `paper_metadata_annotation_fill_template.csv` 填写可导入元数据；`paper_repro_metadata_annotation_sheet.csv` 只作为带曲线、误差和优先级的上下文表。
- [ ] 如需可视化标注，打开 `paper_repro_metadata_annotation_dashboard.html` 或 paper-assets 中的 `paper_metadata_annotation_dashboard.html` 辅助查看样例帧、曲线和复核图。
- [ ] 导入前运行 `import_rr_metadata_annotations.py --annotation-csv Dataset_new\72video\al_images\paper_repro_quality_residual_paper_assets\paper_metadata_annotation_fill_template.csv` 做 dry-run；人工核对 validation 后再加 `--write-template`。
- [ ] 标注时先按 `q2_annotation_score` 降序处理 `batch_1_residual_errors_and_representative_cases`，再处理高不确定性 manual-review 视频。
- [ ] `collection_date` 或 `scene_id` 至少覆盖 73/73。
- [ ] 运行 `rr_quality_residual_metadata_group_validation.py --group-columns cow_id`。
- [ ] 若有日期，运行 `--group-columns cow_id collection_date`。
- [ ] 至少补一个外部测试集，或明确说明没有外部测试的限制。
- [ ] 外部测试集不参与阈值选择。
- [ ] 运行 `audit_rr_metadata_quality.py`，确认元数据 claim gates 和 external split 泄漏筛查不再报告 Q2 blocking 项。
- [ ] 运行 `freeze_rr_external_method.py`，记录外部验证前的 method freeze id。
- [ ] 运行 `rr_external_split_validation.py`，确认 `external split validation ready` 为 PASS 后再报告外部测试指标。
- [ ] 运行 `audit_rr_submission_readiness.py`，确认 `Q2-or-higher submission readiness` 从 `not_ready` 变为 `ready`。

### 热应激/健康意义

- [ ] `ambient_temperature_c` 和 `relative_humidity_percent` 覆盖 73/73 或外部测试集。
- [ ] 运行 `build_rr_heat_stress_context.py` 并生成 THI 分层。
- [ ] 若采用 ATHI，明确 ATHI 公式和引用来源。
- [ ] 不在元数据缺失时报告 RR-THI 相关性。

### 算法与统计

- [ ] 主结果不用 truth-calibrated。
- [ ] 阈值网格最佳结果只放 sensitivity analysis。
- [ ] 主文同时报告 fixed threshold、nested CV 和 group validation。
- [ ] 报告 bootstrap CI，并承认当前 CI 跨 0 的限制。
- [ ] 增加至少 3 个典型曲线案例图。

### 文稿与图表

- [ ] 更新 `docs/thermal_rr_manuscript_draft.md` 中所有待补元数据段落。
- [ ] 使用 `rr_quality_residual_combined.png/.pdf` 作为主结果图候选。
- [ ] 检查所有图中文字在最终版面中可读。
- [ ] 正式投稿前核对目标期刊作者指南、图像格式、参考文献格式和数据可用性声明。

## 6. 当前最推荐的下一步

最优先不是继续调模型，而是补齐 `paper_metadata_template.csv` 中的关键字段。推荐最低补充集为：

1. `cow_id`
2. `collection_date`
3. `scene_id` 或 `camera_id`
4. `ambient_temperature_c`
5. `relative_humidity_percent`
6. `head_motion_score_0_3`
7. `occlusion_score_0_3`
8. `nostril_visibility_score_0_3`
9. `external_test_split`

当前可直接填写的安全模板是：

- `Dataset_new\72video\al_images\paper_repro_quality_residual_paper_assets\paper_metadata_annotation_fill_template.csv`

当前最短 Q2 元数据清单是：

- `Dataset_new\72video\al_images\paper_repro_quality_residual_paper_assets\paper_metadata_annotation_minimum_q2_checklist.md`

补完后按以下命令重跑：

```powershell
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\build_rr_metadata_annotation_pack.py
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\build_rr_metadata_annotation_dashboard.py
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\import_rr_metadata_annotations.py --annotation-csv Dataset_new\72video\al_images\paper_repro_quality_residual_paper_assets\paper_metadata_annotation_fill_template.csv
# 人工核对 import validation 后，再用下面命令正式写入模板：
# & 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\import_rr_metadata_annotations.py --annotation-csv Dataset_new\72video\al_images\paper_repro_quality_residual_paper_assets\paper_metadata_annotation_fill_template.csv --write-template
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\audit_rr_metadata_quality.py
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\rr_quality_residual_metadata_group_validation.py --group-columns cow_id
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\rr_quality_residual_metadata_group_validation.py --group-columns cow_id collection_date
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\build_rr_heat_stress_context.py
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\freeze_rr_external_method.py
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\rr_external_split_validation.py
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\audit_rr_submission_readiness.py
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\audit_rr_manuscript_claims.py
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\build_rr_paper_assets.py
```

如果真实分组验证仍保持 R2 高于默认流程，THI/ATHI 分析能提供合理生理解释，并且 readiness 审计不再报告 Q2 blocking 项，则稿件可以进入“二区以上投稿前完整初稿”阶段。

## 7. 需要再次核对的公开信息链接

正式投稿前请逐项核查以下页面或数据库：

- Computers and Electronics in Agriculture 官方期刊页：https://www.sciencedirect.com/journal/computers-and-electronics-in-agriculture
- Biosystems Engineering 官方期刊页：https://www.sciencedirect.com/journal/biosystems-engineering
- Journal of Thermal Biology 官方期刊页：https://www.sciencedirect.com/journal/journal-of-thermal-biology
- Journal of Dairy Science 官方期刊页：https://www.journalofdairyscience.org/
- SCImago Journal & Country Rank：https://www.scimagojr.com/
- Web of Science / Journal Citation Reports：以学校或机构订阅版本为准。
- 中科院期刊分区表：以当年最新版官方/机构可访问版本为准。
