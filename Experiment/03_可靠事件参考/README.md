# 03 可靠事件参考

用途：把“整段数了多少次”补充为“每次呼吸发生在何时、哪一段看不清”，检验真实检出、伪峰与漏峰。

## 最新正式提交

2026-09-18进一步修订：[两窗状态修订归档](submissions/20260918_r2_status_v1/README.md)。两窗已由pending改为complete，分别16/10个确定事件；4345个事件及区间表原字节均未变化。当前林甸47 complete、2 partial；久福216 complete、8 partial、47 unobservable、pending=0。新的原始导出、差异与校验单独归档；下文2个pending仅指9月17日旧版。

2026-09-18完成处理：[9月17日桌面“标注”正式R2提交](submissions/20260917_r2_v2/README.md)，4345个真实用户导出事件（4340 confirmed、5 uncertain），区间表0行。林甸47 complete、2 partial；久福214 complete、8 partial、47 unobservable、2 pending。原三表/JSON已归档，标签不改写；完整窗口评分完成，部分窗口及不可观察时长分析未完成。两个pending是20240801T105204-919.MP4、20240806T095740-373.MP4，原状态保留。下文“事件仍为空”仅为旧阶段记录，不能当作当前状态。v1下载目录旧导出已SUPERSEDED。

**双盲仍暂停。** 本包提供单人、隐藏算法预测的逐事件标注和重复复核流程，不宣称双人一致性或传感器金标准。现有人工总次数不自动展开成均匀时间点，算法峰也不作为人工参考。

## 在哪里填写

2026-09-14新增：[本地逐事件标注页面R2](event_workspace/20260914_v3/index.html)，含320窗，不显示已有总次数或算法输出。可记录/删除/定位事件、标记不可观察区间、导出三表与JSON备份。林甸使用另外保存的兼容观看副本，49窗全部逐帧核对像素和时间戳；原长视频起点与观看片段0秒分开，原R1表不改。使用说明在同目录；v1/v2是未通过测试的开发页，不用于人工标注。

R2页面和软件测试不是已经产生人工事件。测试数据只在隔离浏览器中临时创建并清除；导出目录的事件表仍为空。林甸已有非盲历史不因本轮隐藏页面而消失；久福R2如已接触预测请如实登记，不能覆盖R1的原始声明。

| 范围 | 文件夹 | 要做什么 |
|---|---|---|
| 林甸49个已有MP4 | `annotations/20260909_v1/` | 内部事件诊断；含170333、ns210947等，但先不要看历史诊断文件 |
| 久福271个已冻结首30秒窗口 | `annotations/jiufu271_frozen_v1/` | 后续独立测试的参考；链接到原长视频，只观察表中起点和30秒范围 |

每个目录都有三个配套CSV：

1. **`annotation_windows.csv`**：填写`annotator`、`predictions_hidden`、`manual_breath_count`、`annotation_status`和备注。`video_path`可找到观看文件。`manual_breath_count`此处是本轮事件参考计数，**不会覆盖旧的49窗人工表**。
2. **`reference_events.csv`**：每次呼吸一行，填写事件时间（相对该观看窗口起点的秒数）、事件ID、置信度、标注轮次和人名。不能只填总数。
3. **`unobservable_intervals.csv`**：鼻孔/呼吸运动看不清的时间段及原因。看不清不填“0次呼吸”。

详细字段和操作步骤见 [ANNOTATION_PROTOCOL.md](ANNOTATION_PROTOCOL.md) 与 `field_dictionary.csv`。

2026-09-11更新：两份窗口总次数表已提交。林甸49条均有计数（39 completed、10 uncertain），久福157条complete有计数、114条unobservable无计数；两组逐事件表仍为空。原窗口CSV经表格软件保存为GB18030，新增中文备注均保留。总次数可单独分析，但不能生成事件F1。

用户最新更正：林甸未隐藏预测；久福原本没有已有计数和预测标注，人工计数未参考算法结果。派生表按该说明记录，不回写原始表，不宣称双观察者验证。

先看 [本次提交处理报告v2](submissions/20260911_v2/处理结果与下一步.md)。`submissions`保存原始字节快照、UTF-8副本、计数用途表和事件参考缺项；`../01_同口径消融/reference_updates/20260911_v2/`保存已有预测复算，不是重新推理。v1保留，v2增加冻结窗口、输入来源及提交专属声明的检查，指标完全相同。`annotation_declarations_20260911.json`保存本次用户确认并绑定原表哈希，不能无核对复用于其他提交。

## 其他文件用途

| 文件 | 用途 |
|---|---|
| `reference_tools.py` | 参考完整性检查；一对一事件匹配；输出TP、FP、FN、Precision/Recall/F1及时间误差 |
| `validation_20260909_v1/reference_validation.json` | 49窗初始模板检查：49窗、0完成、0事件，不产生准确率 |
| `validation_jiufu271_v1/reference_validation.json` | 271窗初始模板检查，状态同样为等待人工事件 |
| `prediction_events_template.csv` / `prediction_windows_template.csv` | 算法事件评估输入格式；空表不是实际预测 |
| `annotations/20260909_v1/historical_case_notes_not_blind.csv` | 用户历史病例备注，不是事件真值；为减少确认偏差，首次标注前不展示 |
| `annotations/20260909_v1/rgb_thermal_pair_candidates.csv` | 原扫描的RGB配对线索；最新久福窗口线索应以第二目录curated_v3为准 |

## 检查命令

当前提交包含旧版`completed/uncertain`状态且尚无事件，不直接按完整事件参考评分。`../process_annotation_submission.py`支持本次两种编码并保留原表；下面的严格事件命令应在另一个已规范化、逐事件完成的目录使用，不能把总次数自动展开为时间点。

```powershell
$env:PYTHONUTF8='1'
$py='E:\real\anaconda\envs\plant_gpu\python.exe'
& $py 'Experiment\03_可靠事件参考\reference_tools.py' validate --annotations 'Experiment\03_可靠事件参考\annotations\20260909_v1' --out 'Experiment\03_可靠事件参考\validation_new'
```

有真实事件参考且峰已映射到同一观看视频时间轴后：

```powershell
& $py 'Experiment\03_可靠事件参考\reference_tools.py' score --annotations '已完成的标注目录' --prediction-events '实际预测事件.csv' --prediction-windows '实际预测窗口状态.csv' --round R1 --out '新的事件评分目录'
```

每个输出目录用新名称。无完成参考会拒绝评分；名义帧号/8.7未经时间轴核验也会拒绝评分。
