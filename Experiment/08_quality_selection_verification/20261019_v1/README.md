# 质量选择核验交付入口

日期2026-10-19；计划上限10月23日，已提前完成。本轮仅复放既有缓存，不新增标注、训练、模型推理或模块。

**结论：证据仍不确定。保留候选，不改默认，不自动开始下一轮。**

1. [归因报告](E:/real/use_code/yoloV8/Experiment/08_quality_selection_verification/20261019_v1/REPORT.md)：主要对比、正负结果与停止条件。
2. [16组横向指标Excel](E:/real/use_code/yoloV8/Experiment/08_quality_selection_verification/20261019_v1/metric_comparison.xlsx)：各项指标的末行仅汇总82/69/13个约30秒完整参考窗，不混入旧73。
3. [六篇针对性查新](E:/real/use_code/yoloV8/Experiment/08_quality_selection_verification/20261019_v1/LITERATURE_COMPARISON.md)：5篇全文，1篇摘要/索引。
4. [论文写作段落](E:/real/use_code/yoloV8/Experiment/08_quality_selection_verification/20261019_v1/WRITING_DRAFT.md)：数据、方法、结果、讨论与局限。
5. [六例图文](E:/real/use_code/yoloV8/Experiment/08_quality_selection_verification/20261019_v1/CASES.md)：全部82窗中改善/退化最大各3例，非主要47窗的六个极值。
6. [最终主张边界](E:/real/use_code/yoloV8/Experiment/08_quality_selection_verification/20261019_v1/CLAIMS_FROM_RESULTS.md)：同家族临时审阅、证据支持与缺口。

## 复算与封存

`PROTOCOL.md`及核心实现、测试和输入已由`freeze.json`锁定。历史结果不覆盖，预测/统计阶段以独占写入拒绝覆写。不要直接在本目录重复运行`all`、`predict`、`evaluate`或生成器；需要再计算时建立另一个明确版本，保留原文件和协议修订。

只读核验命令，在项目根目录运行：

```powershell
& 'E:/real/anaconda/envs/plant_gpu/python.exe' 'E:/real/use_code/yoloV8/Experiment/08_quality_selection_verification/20261019_v1/run_verification.py' verify
& 'E:/real/anaconda/envs/plant_gpu/python.exe' -m unittest discover -s 'E:/real/use_code/yoloV8/Experiment/08_quality_selection_verification/20261019_v1' -p test_verification.py -v
```

`numeric_validation.json`记录逐曲线重数、汇总/区间重建；`supplemental_validation.json`记录评分选择、上游来源与图像检查；`FINAL_VALIDATION.json`与`COMPLETED.json`为最终验收标记。`delivery_manifest.json`列出本次新交付文件SHA256，项目总记录最新节点N100。

Excel是预计算快照，不含公式；未做Excel界面渲染。输入哈希与数值复算通过不等于独立生理验证或正式科研诚信审计。原始审阅留在项目`.aris/traces/novelty-check/20261019_quality_selection/`及对应`result-to-claim`目录。
