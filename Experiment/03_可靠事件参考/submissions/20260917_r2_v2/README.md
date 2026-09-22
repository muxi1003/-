# 本次人工事件参考提交

来源：用户指定2026-09-17桌面“标注”目录三表及JSON。CSV与备份语义一致；12和12.0只按数值等价处理，原字节不修改。下载目录9月16日旧版停止评估、保留为superseded，不用于最终报告。

320窗、4345个事件（4340 confirmed、5 uncertain）、不可观察区间0行。
林甸47 complete、2 partial；久福214 complete、8 partial、47 unobservable、2 pending。

此前152窗问题属于错误选择的9月16日旧导出，不适用于新版。新版仅剩2窗pending，无事件与本轮备注，旧R1其中1条有备注，背景列单列而不回写R2。不将排除窗标成0或强制改状态。

- raw/：原字节副本及JSON；不得回写。
- lindian49/、jiufu271/：按牧场拆分三表及R1/R2次数比较。
- reference_validation.json：结构、声明、事件/次数一致性、元数据绑定检查。
- exclusion_reason_reconciliation.csv：所有排除窗口、原状态、新旧备注和用户声明。
- lindian_frame_time_bindings.csv：实际解码的观看视频逐帧秒数。
- analysis_policy_before_scoring.json：评分规则、原始文件哈希及用户说明。
- R1_R2_count_agreement.csv：仅总次数的一致性，不是两轮逐事件重复性。

原R1没有事件时刻，不能计算观察者内逐事件重测一致性。软件检查通过不证明人工相位本身准确。
