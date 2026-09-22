# 冻结推理与独立测试协议

## 已固定

1. 271个久福原始录制，每个取首30秒。先前已用于开发的场次、同牛号不进入该集合。统计按牛号聚类，不把同牛不同日期/片段当完全独立观测。
2. 用户确认该候选范围没有YOLO训练、RR调参、RR预测查看经历。源文件与成员表有SHA256。
3. RR真值由可靠人工事件次数和同一窗口时长换算。无参考、不可观察、算法拒判各自保留，均不填0。
4. 保存目前30秒分支的信号处理策略；不使用73短片分支的目标峰数10重调规则。

## 2026-09-14限定迁移评价补充

N051在保持冻结271窗成员、YOLO/RF/信号及ROI参数的前提下，采用已做内部工程验证的raw时间采样完成代理信号迁移预测，目录`external_transfer/20260914_v2`。这不是把下列尚未确认的温标/可见性字段改成已验证：绝对温度与解剖可见性继续列为限制。新增代码、运行环境与处理策略锁见protocol_lock.json，全部预测先封存，再执行额外确定性验收，最后由独立的score_external_counts.py读取归档的单人总次数参考。旧frozen_rr_evaluation.py的完整事件参考通道及严格要求保留，不用总次数绕过事件F1检查。

此分支只允许“冻结RF伪彩代理信号的跨场总次数/RR评价”。研究方法可以有真实失败，不能因为失败而事后放宽时间规则、挑窗或换参数。旧版截至未推理时的描述保留在下节；当前状态以N051及新分支封存/验收为准。

## 原端到端测温/事件通道的待核对事项

- 核对271窗的实际媒体：是否为预期伪彩色热红外、首30秒完整解码、连续时间戳、镜像/旋转、图像几何和有效区域。
- 逐帧时间用真实时间戳建立映射，再确定对8.7Hz名义采样网格的处理规则。禁止直接用原MP4帧号除以8.7冒充实际时间；检测峰映回人工观看视频的秒数。
- 明确伪彩色色盘、固定/自动标尺及温度上下限。路径含18–36的11个候选仅是文件夹提示，不是逐帧测量。若标尺与RF训练语义不兼容，不能直接声称输出真实摄氏温度；先做不依赖呼吸标签的校准核验，并将处理规则固定。
- 确认ROI半径所在图像分辨率，左右关键点顺序与BGR通道，ROI是否混入标尺/文字区域。实际解码失败、不可用ROI按预先定义规则输出拒判，不能事后删除困难样本。
- 把输入适配器、模型绑定、依赖版本、预处理和拒判规则全部加入新的完整端到端冻结清单，在内部数据或不属于此测试集的技术校准样本上验证。

**当前没有宣称这些检查已完成，也没有生成271窗的RR预测。** 改动上述尚未冻结的技术细节时不得看该集合的呼吸率标签或用其预测性能选参。若已经这样做，应重新标记为开发集，另取外测。

## 推理输出接口

`prediction_windows.csv`每个已冻结window_id恰好一行，含：

`window_id,prediction_status,predicted_count,duration_seconds,method_manifest_sha256`

`prediction_status=ok`时次数为非负整数；`abstain`时次数留空。另存逐帧温度、ROI/质量来源、事件坐标与对应视频秒数。事件评估接口见第三个文件夹。

另提供 `inference_provenance.json`，至少含：`runner_path,runner_sha256,operator,started_at,finished_at,timestamp_validation_report,reference_counts_used=false,test_outcomes_used_for_tuning=false`。时间核验报告须为 `status=VALIDATED` 且列出全部 `window_ids`。这些声明必须有实际运行证据支撑，不是把字段改成VALIDATED就算完成。

## 封存后评分

```powershell
$py='E:\real\anaconda\envs\plant_gpu\python.exe'
# 下面的输入文件必须来自实际冻结推理；当前没有生成它们。
& $py 'Experiment\02_冻结独立测试\frozen_rr_evaluation.py' seal --release-dir 'Experiment\02_冻结独立测试\release_20260909_v1' --predictions '实际预测表.csv' --provenance '实际推理溯源.json'
& $py 'Experiment\02_冻结独立测试\frozen_rr_evaluation.py' score --release-dir 'Experiment\02_冻结独立测试\release_20260909_v1' --annotations 'Experiment\03_可靠事件参考\annotations\jiufu271_frozen_v1' --out '新的外测评分目录'
```

封存文件在同一release目录中只能排他创建一次；哈希变化、漏窗或混用方法会拒绝评分。哈希与操作者声明不是独立的盲法取证，不夸大工具能证明什么。

主表同时报告n冻结、n可靠参考、n可输出、n配对、拒判/失败/无参考原因及覆盖率。当前脚本输出RR点估计；正式稿还需按核实牛号给出聚类置信区间和预先定义的质量/运动分层。外测结果差时报告域偏移/失败，而不是用同一测试集继续优化后再称独立。
