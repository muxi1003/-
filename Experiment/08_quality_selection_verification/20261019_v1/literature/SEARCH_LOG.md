# 检索范围和限制

检索日期：2026-10-19，Asia/Shanghai；最相关六篇进入literature_comparison.json。不是系统综述或穷尽查新，不因领域有近邻就否定具体贡献。

## 实际检索式

- Contactless Vital Signs Measurement System Using RGB-Thermal Image Sensors SQI
- Estimation of Respiratory Rate from Thermography Using Respiratory Likelihood Index
- Multi-camera infrared thermography for infant respiration monitoring signal quality
- Detection of Respiratory Rate of Dairy Cows Based on Infrared Thermography and Deep Learning 1939
- Adaptive Thermal Imaging Signal Analysis for Real-Time Non-Invasive Respiratory Rate Monitoring
- Respiratory rate detection of dairy cows 104154
- thermal respiration signal quality selection nostril waveform fusion peak interval variability missing data
- infrared respiratory rate SQI signal selection time domain prominence amplitude quality
- thermal respiration signal quality selection arxiv 2026 after:2026-04-19

目的分别为：候选热信号选择先例；规则性/缺失等质量依据的既有范围；牛头动测量和最近半年近邻工作。标题精确检索补充核对论文身份，不把检索命中当作全文核实。

## 来源与读取

Chen和Zhao为用户本地原PDF，经PyMuPDF实际取得14/15页文本。Negishi、Takahashi、Analia通过Europe PMC公开JATS接口取得全文，XML和派生文本及哈希保留。

Infant2020的Europe PMC两次公开接口请求失败，NCBI公开BioC接口未返回有效XML；保留首次错误、补充尝试和BioC记录。仅使用作者文章的公开索引摘要与可检索方法段落，不声称已完成全文查读。

原始抓取作者列表曾混入编辑，verified_bibliography.json已按JATS contrib-type=author区分；原抓取记录原样保留。Analia的JATS作者记录与网页索引呈现有差异，比较表使用标题及DOI，不以不一致作者列表下结论。

## 最近半年外围命中

- https://arxiv.org/abs/2606.17376 ：机器人多模态呼吸监测、SQI筛选，预印本，非牛双鼻孔时域四项评分。
- https://arxiv.org/abs/2608.14973 ：CORAL，自动通道选择及信号质量指示，预印本，非本轮完整方法的已证实相同实现。
- https://arxiv.org/abs/2610.11755 ：SR-TTA，射频多天线呼吸信号融合，预印本，模态不同。

外围命中仅列为竞争/相关方向，不扩展本轮六篇详细对照，不据预印本邻近性作ABANDON判断。没有对这些预印本执行完整代码复现或声称本方法优于它们。

## 判断原则

通用“质量评分后择优热呼吸信号”已有明确先例。可能的较窄区别是牛双鼻孔五候选、四项时域评分，以及共同补齐控制下的必要性/失效分析；是否有足够实质性，仍需结合当前实证与审查。未找到精确同一实现，不等于已经证明领域首次；看到类似SQI，也不等于证明整个研究无价值。
