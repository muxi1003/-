# 长缺失计数保护候选：不采用为默认

日期：2026-09-14。独立候选运行完成；仅内部开发验证，没有久福预测。

## 规则与证据范围

沿用3帧短缺口、0.5关键点置信度和3帧事件支持半径。单侧融合检查所选侧；mean/min/max保守要求两个输入都有直接检测支持。仅允许内部不超过3帧的短缺口；长缺口或边界缺失导致整窗不报数值。被标记峰仅作诊断，不能把删峰后的部分次数当完整30秒呼吸次数。

这不是鼻孔可见性的人工真值：检测也可能错位；混合模式要求双侧可使门控过严。通过门控只说明满足本条代理规则，不证明呼吸计数准确。

## 同口径结果

| 分析集与方法 | 输出/参考窗口 | RR R² | MAE（次/分） |
|---|---:|---:|---:|
| all49_sensitivity / control_full | 49/49 | 0.888037 | 2.611234 |
| all49_sensitivity / control_same_output_subset | 18/49 | 0.908793 | 2.777314 |
| all49_sensitivity / guard_output_subset | 18/49 | 0.908793 | 2.777314 |
| completed39_primary / control_full | 39/39 | 0.913337 | 2.460418 |
| completed39_primary / control_same_output_subset | 15/39 | 0.927225 | 2.532891 |
| completed39_primary / guard_output_subset | 15/39 | 0.927225 | 2.532891 |

旧49窗控制回放49/49计数一致。全49候选只输出18窗，39 completed主分析只输出15窗。候选与旧方法在相同保留子集上计数和误差完全相同；不能把子集0.908793与全49的0.888037相减称提升。预设覆盖率不得降低条件未满足，决定DO_NOT_ADOPT_KEEP_DEFAULT。

新原始时间49窗仅描述覆盖：原37窗输出进一步降到9窗；上游12个拒绝保持，新增28个信号支持拒绝。此组不与旧人工计数评分。

## 病例与局限

- 170333：整窗被拒绝，但用户怀疑的第63帧峰没有被本规则标记，说明长缺失保护不能解释或修正该峰。
- ns210947：用户确认的82帧真峰未被标记、未被删除；整窗拒绝来自其他支持不足区段，不代表82帧错误。
- zs197000：保留25次输出，与旧计数一致。
- 共55个基线峰落在支持不足诊断范围，不能称55个伪峰，也没有据此生成55个事件真值。

![真实基线曲线与支持不足标记](E:/real/use_code/yoloV8/Experiment/02_冻结独立测试/signal_guard_validation/20260914_v1/case_support_diagnostics.png)

## 验证与下一步

推理和评分由两次独立命令运行；先保存/校验预测哈希，再读取20260911_v2林甸参考。参考文件与归档交付哈希一致；旧观看视频哈希与时长均绑定。58项软件测试通过，49窗支持掩码重放一致。单观察者非盲总次数不是事件参考，未报事件F1或独立外测精度。

同家族只读审查为WARN/provisional，详见REVIEW.md：seal未覆盖全部评分上下文；raw时间绑定依赖上游N048而非本轮独立解码验证。不能把确定性掩码重放通过扩大成完整科学验证或正式外测放行。

默认算法、模型和人工表未改。保留候选作为质量诊断，不采用如此严格的整窗拒绝策略。下一步可在连续可观察片段内单独检测、报告有效观测时长和部分次数；任何片段RR须明确口径，不能混称整窗RR，也不能按久福人工次数选规则。

## 文件用途

- internal_metrics.csv、internal_evaluated_rows.csv：同口径内部评分及逐窗参考。
- internal_predictions_before_scoring.csv、prediction_seal.json：评分前预测与哈希。
- raw_predictions_unscored.csv：新原始时间窗的覆盖诊断，不含人工评分。
- internal49/support/、gaps/、control_curves/：逐帧来源支持、缺口区间与原峰位置。
- known_case_checks.csv、case_support_diagnostics.png：病例诊断，不是新事件真值。
- decision.json、artifact_verification.json：不采用决定及确定性核验。
