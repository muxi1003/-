# 六篇原始研究的针对性对照
这是有限范围的方法差异核查，不是穷尽性系统综述或独立创新证明。五篇全文已取得，一篇仅公开摘要/方法索引，访问限制已保留。

## Respiratory rate detection of dairy cows based on infrared thermography in head movement scenarios (2025)
来源：[Journal of Thermal Biology](https://doi.org/10.1016/j.jtherbio.2025.104154)；读取等级：用户提供PDF全文，14页，文本已核对。
- 候选：左右鼻孔的归一化温度曲线。
- 评分：按鼻孔可见情况逐帧融合，双侧可见取归一化较大值；不是本项目的五曲线窗口评分。
- 决策与验证：逐帧双鼻孔融合，MAF和平滑后峰计数；246段牛脸热视频；人工观察RR；不同数据和模型，不能将其R²与本轮直接当公平增益。
- 本项目差异：固定温度输入下，对mean/max/min/left/right在整窗评分并选曲；本轮还将补齐方式与选曲因素分离。
- 不能推出：只证明与该文规则不同，不证明质量选择概念首次提出；缺原作者完整模型，非端到端完整复现。
- 定位：本地全文：温度曲线融合及峰检测方法、prominence=0.05段落；另核对项目原规则脚本。

## Detection of Respiratory Rate of Dairy Cows Based on Infrared Thermography and Deep Learning (2023)
来源：[Agriculture](https://doi.org/10.3390/agriculture13101939)；读取等级：用户提供PDF全文，15页，文本已核对。
- 候选：YOLOv8定位鼻部、Mask2Former等分割得到的鼻孔温度信号。
- 评分：温度信号Butterworth滤波及峰显著度抑制小波峰；不能将使用显著度本身说成新贡献。
- 决策与验证：分割区域提温、滤波后滑窗峰计数；81段红外视频；模型定位/分割与RR均有评价，报告相关系数R而非本项目identity-line R²。
- 本项目差异：本项目不以更换分割网络为贡献，而比较已有双鼻孔温度候选在整个窗口中的可靠性决策。
- 不能推出：滤波、显著度和峰计数是已有组件；本轮未证明比该端到端系统更优。
- 定位：本地全文：滤波、prominence、滑窗检测方法及摘要。

## Contactless Vital Signs Measurement System Using RGB-Thermal Image Sensors and Its Clinical Screening Test on Patients with Seasonal Influenza (2020)
来源：[Sensors](https://doi.org/10.3390/s20082171)；读取等级：开放JATS全文已取得，Section 2.5及3.2已核对。
- 候选：鼻与口各提取mean/min温度，形成四信号。
- 评分：鼻/口SQI基于PSD、ACR和CPSD所估RR的一致性，按0.85至1.15比例规则判断，非本项目四项时间域加权。
- 决策与验证：先选鼻或口温度信号，再由MUSIC估RR；研究包含15秒测量；健康与流感人群，对接触式参考设备评价；不是牛双鼻孔数据。
- 本项目差异：候选为左右鼻孔及三种融合曲线；以显著度、幅值、峰间隔变异和缺失率评分，最终计数采用冻结P0。
- 不能推出：已覆盖多热信号按质量选择这一通用思想，不能宣称本项目首次质量选曲；具体牛场适用性仍需证据。
- 定位：literature/negishi2020.xml：Section 2.5的四信号、两条SQI规则、MUSIC；Section 3.2。

## Estimation of Respiratory Rate from Thermography Using Respiratory Likelihood Index (2021)
来源：[Sensors](https://doi.org/10.3390/s21134406)；读取等级：开放JATS全文已取得，Sections 3.4与4.4已核对。
- 候选：YOLOv3面部框切为4×6子区域的温度信号。
- 评分：RQI结合频谱形状指数与时域/频域RR一致性的sigmoid项，选择最高RQI子区域。
- 决策与验证：候选区域质量评分后选RR，非逐帧Max融合；7名未用于训练的受试者，各15秒多呼吸频率；呼吸带参考，文中说明未实际验证救护车晃动。
- 本项目差异：不是网格区域定位，而是在现有左右鼻孔温度上比较五条时域候选与缺失补齐；没有复现其频谱RQI全系统。
- 不能推出：呼吸似然评分并择优已有先例；改变候选或指标不自动构成充分创新。
- 定位：literature/takahashi2021.xml：Sections 3.4.3、3.4.4，Eq.(13)，Section 4.4。

## Multi-camera infrared thermography for infant respiration monitoring (2020)
来源：[Biomedical Optics Express](https://pmc.ncbi.nlm.nih.gov/articles/PMC7510882/)；读取等级：公开摘要及可检索方法段落已核对；全文API/传输受限，不能称全文精读完成。
- 候选：多热相机合并图像平面的候选像素/区域。
- 评分：以三种特征选核心呼吸像素，再结合时间上高相关像素；文中对谱形状SQI选区有基线讨论。
- 决策与验证：区域/像素和多视角信号构成，采用窗口处理；7名婴儿、152分钟热记录，与胸阻抗参考比较；详细参数不从未取得全文处推测。
- 本项目差异：牛双鼻孔五条候选的固定窗时间域评分，不是多相机自动像素选区。
- 不能推出：说明质量选择与多热信号融合领域已有工作；不能据未核实全文断言其没有任何相同评分特征。
- 定位：出版作者原文的PMC索引摘要、Sections 1和2.1；所有失败的API取文记录保留于literature/。

## Adaptive Thermal Imaging Signal Analysis for Real-Time Non-Invasive Respiratory Rate Monitoring (2026)
来源：[Sensors](https://doi.org/10.3390/s26010278)；读取等级：开放JATS全文已取得，摘要、方法与结论已核对。
- 候选：YOLO定位、间帧Kalman保持的鼻孔ROI温度。
- 评分：median-MAD滞回相位检测、抑制闪烁、合并与IBI验证，不等同本项目整窗五候选评分。
- 决策与验证：时域呼吸相位/IBI判定，包含边缘部署与检测跳帧；10名健康受试者，六类实验条件，包括说话、偏航、姿态及距离；不能直接与本项目指标排名。
- 本项目差异：本轮固定P0，不新增相位器、Kalman或部署组件，仅分析选曲与补齐及四项评分的必要性。
- 不能推出：简单宣称运动稳健、MAD相位或轻量YOLO不构成独占贡献；本轮不能宣称已实现车载部署。
- 定位：literature/analia2026.xml：摘要、方法中跳帧与median-MAD相位，Section 6。

## 综合定位
不能以质量选曲概念或更换YOLO独占新颖性。可检验的窄差异是五条双鼻孔候选的时域/缺失联合决策及补齐控制归因；这不是已获确认的有效性或领域首次结论。最近半年预印本外围命中另见literature/SEARCH_LOG.md。
