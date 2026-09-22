# Thermal RR 创新实验复现说明

本文档记录当前从默认热红外呼吸频率流程到论文结果包的完整复现顺序。工作目录为 `E:\real\use_code\yoloV8`，推荐解释器为 `E:\real\anaconda\envs\plant_gpu\python.exe`。

## 1. 当前核心结论

默认流程在 73 个视频上达到 `RR R2 = 0.928992`、`MAE = 1.9713 bpm`、`RMSE = 3.3950 bpm`、`exact count = 49/73`。质量感知残差校正在固定阈值 out-of-fold 设置下达到 `RR R2 = 0.942885`、`MAE = 1.6187 bpm`、`RMSE = 3.0448 bpm`、`exact count = 53/73`。Nested threshold CV 和 leave-one-prefix-group-out 验证分别达到 `RR R2 = 0.937336` 和 `0.936263`。

信号共识补充实验已实现：在保留质量感知残差校正的基础上，仅当残差模型方向与 spectral/FFT/自相关中至少两个信号一致、且峰间隔变异达到保守阈值时才补充校正。固定 out-of-fold 指标达到 `RR R2 = 0.946345`、`MAE = 1.4955 bpm`、`RMSE = 2.9512 bpm`、`exact count = 55/73`；nested CV 达到 `RR R2 = 0.940796`、`exact count = 53/73`；prefix-group fixed threshold 达到 `RR R2 = 0.936263`、`exact count = 53/73`。该结果是新的内部候选精度扩展，仍需要真实元数据分组和外部验证。

Selective RR reporting 已实现：严格自动报告规则在 13/73 个视频上达到 `RR R2 = 0.999907`、`MAE = 0.0611 bpm`、`exact count = 13/13`；score 阈值 0.35 在 21/73 个视频上达到 `RR R2 = 0.987118`、`MAE = 0.4902 bpm`、`exact count = 19/21`。该结果只表示内部不确定性分层和人工复核优先级，不表示总体预测性能。

`truth-calibrated` 的 `RR R2 = 0.999655` 只能作为上限分析，不能写成主方法性能。

## 2. 一次性复现命令

按下面顺序运行，可以刷新当前主要结果、验证表、图表和论文结果包。

```powershell
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\build_rr_error_taxonomy.py
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\rr_quality_residual_corrector.py
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\rr_quality_residual_validation.py
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\rr_quality_residual_group_validation.py
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\rr_signal_consensus_validation.py
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\rr_selective_prediction_validation.py
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\rr_quality_residual_ablation.py
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\rr_quality_residual_figures.py
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\rr_quality_residual_case_figures.py
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\build_rr_heat_stress_context.py
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\build_rr_paper_assets.py
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\build_rr_metadata_annotation_pack.py
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\build_rr_metadata_annotation_dashboard.py
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\import_rr_metadata_annotations.py
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\audit_rr_metadata_quality.py
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\freeze_rr_external_method.py
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\rr_external_split_validation.py
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\audit_rr_submission_readiness.py
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\build_rr_literature_gap_table.py
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\build_rr_submission_gap_action_pack.py
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\build_rr_external_validation_sample_plan.py
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\build_rr_pseudo_external_gate_stress_test.py
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\audit_rr_manuscript_claims.py
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\build_rr_paper_assets.py
```

`audit_rr_manuscript_claims.py` 会检查 `docs\thermal_rr_manuscript_draft.md` 是否把 truth-calibrated、selective reporting、pseudo-external stress test、external validation 和 Q2 readiness 写在正确证据边界内；该审计通过也不替代真实元数据和外部验证。

第一次 `build_rr_paper_assets.py` 会准备 metadata template，`build_rr_metadata_annotation_pack.py` 会生成可填标注表、Q2 优先级分数和分批队列，`build_rr_metadata_annotation_dashboard.py` 会生成离线标注页面，`import_rr_metadata_annotations.py` 默认只做 dry-run 校验不写模板，`audit_rr_metadata_quality.py` 会检查元数据是否足以解锁 cow-level GroupKFold、THI/热应激解释、运动/遮挡分层和 external split claim，`freeze_rr_external_method.py` 会冻结外部验证前的方法脚本、阈值参数和关键结果资产并生成 method freeze id，`rr_external_split_validation.py` 会在 `external_test_split` 未填时拒绝生成外部指标，`build_rr_literature_gap_table.py` 会把本地与联网核对过的文献整理成 innovation gap 表，`build_rr_submission_gap_action_pack.py` 会把当前 Q2+ 阻塞项转成字段清单、视频优先队列和重跑命令，`build_rr_external_validation_sample_plan.py` 会把当前 bootstrap 不确定性转换为外部验证样本量和通过门槛，`build_rr_pseudo_external_gate_stress_test.py` 会用现有前缀组做内部 pseudo-external gate 压力测试，`audit_rr_manuscript_claims.py` 会审计稿件主张边界，最后一次 `build_rr_paper_assets.py` 用于把 annotation/import progress、priority batches、dashboard、metadata quality audit、method freeze manifest、external split readiness、submission readiness 审计结果、literature gap 表、submission gap action pack、external validation sample plan、pseudo-external stress test 和 manuscript claim audit 写入论文结果包和 `paper_assets_summary.md`。

元数据分组验证默认会在 `cow_id` 为空时拒绝运行。填完 `paper_metadata_template.csv` 后再运行：

```powershell
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\rr_quality_residual_metadata_group_validation.py --group-columns cow_id
```

如果只想测试链路，可以运行：

```powershell
& 'E:\real\anaconda\envs\plant_gpu\python.exe' scripts\rr_quality_residual_metadata_group_validation.py --allow-prefix-fallback
```

`--allow-prefix-fallback` 生成的是前缀回退结果，不能作为真实元数据 GroupKFold 写入论文。

## 3. 关键输出位置

默认流程指标：

`Dataset_new\72video\al_images\paper_repro_metrics.csv`

质量感知残差校正输出：

`Dataset_new\72video\al_images\paper_repro_quality_residual_predictions.csv`

`Dataset_new\72video\al_images\paper_repro_quality_residual_metrics.csv`

`Dataset_new\72video\al_images\paper_repro_quality_residual_validation_summary.csv`

特征组消融输出：

`Dataset_new\72video\al_images\paper_repro_quality_residual_ablation_metrics.csv`

`Dataset_new\72video\al_images\paper_repro_quality_residual_ablation_feature_groups.csv`

分组外推验证输出：

`Dataset_new\72video\al_images\paper_repro_quality_residual_group_metrics.csv`

`Dataset_new\72video\al_images\paper_repro_quality_residual_group_metrics_by_prefix.csv`

信号共识补充实验输出：

`Dataset_new\72video\al_images\paper_repro_signal_consensus_comparison_summary.csv`

`Dataset_new\72video\al_images\paper_repro_signal_consensus_predictions.csv`

`Dataset_new\72video\al_images\paper_repro_signal_consensus_nested_metrics.csv`

`Dataset_new\72video\al_images\paper_repro_signal_consensus_group_metrics.csv`

Selective RR reporting 输出：

`Dataset_new\72video\al_images\paper_repro_selective_rr_predictions.csv`

`Dataset_new\72video\al_images\paper_repro_selective_rr_metrics.csv`

`Dataset_new\72video\al_images\paper_repro_selective_rr_threshold_grid.csv`

`Dataset_new\72video\al_images\paper_repro_selective_rr_summary.md`

External split validation 输出：

`Dataset_new\72video\al_images\paper_repro_external_split_readiness.csv`

`Dataset_new\72video\al_images\paper_repro_external_split_metrics.csv`

`Dataset_new\72video\al_images\paper_repro_external_split_acceptance.csv`

`Dataset_new\72video\al_images\paper_repro_external_split_report.md`

论文图表：

`Dataset_new\72video\al_images\paper_repro_quality_residual_figures\`

代表性曲线案例图：

`Dataset_new\72video\al_images\paper_repro_quality_residual_figures\rr_representative_cases.png`

`Dataset_new\72video\al_images\paper_repro_quality_residual_case_examples.csv`

论文结果包：

`Dataset_new\72video\al_images\paper_repro_quality_residual_paper_assets\`

英文草稿：

`docs\thermal_rr_manuscript_draft.md`

创新方案：

`docs\thermal_rr_innovation_plan.md`

## 4. 论文主表建议

主文建议报告默认流程、质量感知固定阈值 out-of-fold、信号共识固定 out-of-fold、nested threshold CV、leave-one-prefix-group-out 训练组内选阈值，以及 selective automatic-report/manual-review triage。阈值网格最佳结果只能写入 sensitivity analysis。Truth-calibrated 只能写 upper-bound analysis。

当前主表数据来自：

`Dataset_new\72video\al_images\paper_repro_quality_residual_paper_assets\paper_main_results_table.csv`

特征组消融表来自：

`Dataset_new\72video\al_images\paper_repro_quality_residual_paper_assets\paper_feature_block_ablation_table.csv`

典型曲线案例表来自：

`Dataset_new\72video\al_images\paper_repro_quality_residual_paper_assets\paper_representative_case_table.csv`

投稿 readiness 审计：

`Dataset_new\72video\al_images\paper_repro_quality_residual_submission_readiness_summary.md`

`Dataset_new\72video\al_images\paper_repro_quality_residual_paper_assets\paper_submission_readiness_table.csv`

Manuscript claim audit:

`Dataset_new\72video\al_images\paper_repro_quality_residual_paper_assets\paper_manuscript_claim_audit.csv`

`Dataset_new\72video\al_images\paper_repro_quality_residual_paper_assets\paper_manuscript_claim_audit.md`

元数据标注包：

`Dataset_new\72video\al_images\paper_repro_metadata_annotation_sheet.csv`

`Dataset_new\72video\al_images\paper_repro_quality_residual_paper_assets\paper_metadata_annotation_codebook.md`

`Dataset_new\72video\al_images\paper_repro_quality_residual_paper_assets\paper_metadata_annotation_progress.csv`

`Dataset_new\72video\al_images\paper_repro_metadata_annotation_priority_summary.csv`

`Dataset_new\72video\al_images\paper_repro_quality_residual_paper_assets\paper_metadata_annotation_priority_summary.csv`

`Dataset_new\72video\al_images\paper_repro_metadata_annotation_dashboard.html`

`Dataset_new\72video\al_images\paper_repro_quality_residual_paper_assets\paper_metadata_annotation_dashboard.html`

Metadata quality and split leakage audit:

`Dataset_new\72video\al_images\paper_repro_quality_residual_paper_assets\paper_metadata_quality_audit.csv`

`Dataset_new\72video\al_images\paper_repro_quality_residual_paper_assets\paper_metadata_quality_field_status.csv`

`Dataset_new\72video\al_images\paper_repro_quality_residual_paper_assets\paper_metadata_split_leakage_audit.csv`

`Dataset_new\72video\al_images\paper_repro_quality_residual_paper_assets\paper_metadata_quality_audit.md`

Frozen external method manifest:

`Dataset_new\72video\al_images\paper_repro_quality_residual_paper_assets\paper_method_freeze_summary.csv`

`Dataset_new\72video\al_images\paper_repro_quality_residual_paper_assets\paper_method_freeze_manifest.csv`

`Dataset_new\72video\al_images\paper_repro_quality_residual_paper_assets\paper_method_freeze_parameters.csv`

`Dataset_new\72video\al_images\paper_repro_quality_residual_paper_assets\paper_method_freeze_report.md`

元数据导入校验：

`Dataset_new\72video\al_images\paper_repro_metadata_import_validation.md`

`Dataset_new\72video\al_images\paper_repro_metadata_import_validation.csv`

External validation protocol:

`docs\thermal_rr_external_validation_protocol.md`

Literature-grounded innovation matrix:

`docs\thermal_rr_literature_innovation_matrix.md`

## 5. 二区以上投稿前仍缺的证据

当前最大缺口不是代码，而是元数据和外部验证。`paper_metadata_template.csv` 里 `cow_id`、`collection_date`、`camera_id`、`scene_id`、`ambient_temperature_c`、`relative_humidity_percent`、`thi`、`athi`、`posture`、`head_motion_score_0_3`、`occlusion_score_0_3`、`nostril_visibility_score_0_3` 仍为空。补齐这些字段后，需要重跑真实 GroupKFold 和热应激上下文分析。

如果可以额外采集 20-50 个头部运动明显、遮挡明显或不同日期/场景的视频作为外部测试集，论文说服力会明显增强。

当前 `audit_rr_submission_readiness.py` 的判断是：内部稿件包 `ready`，但二区以上投稿 `not_ready`。主要阻塞项是 bootstrap 的 `delta RR R2` 置信区间仍跨 0、`cow_id` 真实分组为空、外部测试集为空、温湿度/THI 为空、头部运动/遮挡/鼻孔可见度评分为空。该 readiness 审计现在也会读取 `paper_metadata_quality_audit.csv` 和 `paper_metadata_split_leakage_audit.csv`，因此元数据 claim gates 或 external split 泄漏筛查未通过时，Q2+ 状态不会被误判为 ready。

填写元数据时优先使用 `paper_repro_metadata_annotation_dashboard.html` 或 `paper_repro_metadata_annotation_sheet.csv` 和 `paper_metadata_annotation_codebook.md`。标注表和 dashboard 包含每个视频的当前预测、错误类型、曲线图、复核图、三张采样帧路径、signal-consensus/selective review 状态、`q2_annotation_score` 和 `q2_annotation_batch`。先按 `q2_annotation_score` 降序补 `batch_1_residual_errors_and_representative_cases`，再补 `batch_3_high_uncertainty_manual_review`，最后补高置信对照和剩余视频。dashboard 导出的 `paper_repro_metadata_annotation_filled.csv` 可作为 `import_rr_metadata_annotations.py --annotation-csv ...` 输入；确认无严重错误后，再加 `--write-template` 写入 `paper_metadata_template.csv`，然后重跑 metadata GroupKFold、external split validation 和 readiness 审计。
