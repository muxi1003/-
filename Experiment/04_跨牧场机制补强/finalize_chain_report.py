"""Package the diagnosed failure chain, qualified repair, and negative RR trial."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_transfer import HERE, BASES, read, write_csv, write_json, sha256

ROOT=HERE/'20260921_chain_v3'


def main():
    combined=ROOT/'combined_opt_in';combined.mkdir(exist_ok=False)
    bp=read(BASES['jiufu271']/'prediction_windows.csv')
    be=read(BASES['jiufu271']/'prediction_events.csv')
    pw=read(ROOT/'partial_timestamp/prediction_windows.csv').set_index('window_id')
    pe=read(ROOT/'partial_timestamp/prediction_events.csv')
    bp['event_output_status']=bp.prediction_status
    bp['partial_event_count']=np.nan
    bp['timestamp_supported_core_seconds']=np.nan
    for i,row in bp.iterrows():
        if row.window_id in pw.index:
            p=pw.loc[row.window_id]
            assert row.prediction_status=='abstain' and pd.isna(row.predicted_count)
            bp.loc[i,'event_output_status']=p.status
            bp.loc[i,'partial_event_count']=p.partial_event_count
            bp.loc[i,'timestamp_supported_core_seconds']=p.eligible_seconds
    assert len(bp)==271 and bp.predicted_count.notna().sum()==216
    events=pd.concat([be[['window_id','event_id','event_time_seconds']],pe[['window_id','event_id','event_time_seconds']]],ignore_index=True)
    events['output_scope']=np.where(events.window_id.isin(pw.index),'partial_timestamp_supported_events','original_frozen_output')
    write_csv(combined/'prediction_windows.csv',bp)
    write_csv(combined/'prediction_events.csv',events.sort_values(['window_id','event_time_seconds']))
    (combined/'README.md').write_text('''# 仅事件输出的可选合并结果

原216窗完整输出保持原样。53个原时间拒判窗增加partial_timestamp_supported_events；
prediction_status和predicted_count仍保留原整窗拒判与空值，不能把partial_event_count除30秒当整窗RR。
timestamp_supported_core_seconds仅表示时序支持核心时长，不保证整个片段鼻孔可见。
此结果是已查看外测数据后的回顾性候选，不替换原冻结结果，也不恢复缺失区间内的事件。
''',encoding='utf-8')
    a=read(ROOT/'all271_stage_audit.csv')
    paired=read(BASES['jiufu271']/'paired_count_results.csv').merge(a[['window_id','finite_signal_fraction']],on='window_id')
    paired['squared_error']=(paired.predicted_rr_bpm-paired.truth_rr_bpm)**2
    paired['band']=pd.cut(paired.finite_signal_fraction,[0,.25,.5,.75,1.00001],include_lowest=True)
    contributions=paired.groupby('band',observed=True).agg(windows=('window_id','size'),squared_error_sum=('squared_error','sum'))
    contributions['squared_error_percent']=100*contributions.squared_error_sum/contributions.squared_error_sum.sum()
    write_csv(ROOT/'RR_error_contributions.csv',contributions.reset_index())
    recovery=read(ROOT/'recovery_evaluation/metrics.csv')
    selector=read(ROOT/'observed_periodicity/evaluation/metrics.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axs=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    axs[0].bar(['Timestamp\nrejection','No pose\nall frames','Output\nmismatches'],[557,15,1134],color=['#c75c24','#ba4040','#087f8c'])
    axs[0].set(title='R2 full216: false negatives',ylabel='Reference events')
    for i,v in enumerate([557,15,1134]):axs[0].text(i,v+10,str(v),ha='center')
    strata=read(ROOT/'signal_coverage_strata.csv')
    axs[1].bar(['<=25%','25-50%','50-75%','>75%'],strata.rr_mae,color='#087f8c')
    axs[1].set(title='R2 paired173: available ROI signal',xlabel='Finite signal coverage',ylabel='RR MAE (breaths/min)')
    for i,r in strata.iterrows():axs[1].text(i,r.rr_mae+.2,f'n={int(r.windows)}',ha='center',fontsize=8)
    axs[2].bar(['Frozen','Partial events'],recovery[recovery.reference.eq('R2')].f1,color=['#74858d','#24865a'])
    axs[2].set(title='Same full216 event evaluation',ylabel='Event F1',ylim=(0,.7))
    for i,v in enumerate(recovery[recovery.reference.eq('R2')].f1):axs[2].text(i,v+.01,f'{v:.4f}',ha='center')
    fig.savefig(ROOT/'figures/diagnosis_summary.png',dpi=180);plt.close(fig)
    # Plot one predeclared reviewed time-gap case, without concatenating segment curves.
    wid='jiufu_4f7144680fca9d_first30'
    folder=ROOT/'partial_timestamp/windows'/wid
    spans=read(ROOT/'partial_timestamp/segments.csv')
    fig,axs=plt.subplots(2,1,figsize=(10,5.5),sharex=True,layout='constrained')
    for s in spans[spans.window_id.eq(wid)&spans.signal_status.eq('partial_events_only')].itertuples():
        c=read(folder/f'segment_{s.segment_id:02d}/curve.csv')
        axs[0].plot(c.target_time_seconds,c.smoothed_norm,label=f'Segment {s.segment_id+1}')
        axs[0].axvspan(s.core_start_seconds,s.core_stop_seconds,color='#24865a',alpha=.08)
    gaps=read(ROOT/'partial_timestamp/timestamp_gaps.csv')
    for g in gaps[gaps.window_id.eq(wid)].itertuples():
        axs[0].axvspan(g.start_seconds,g.end_seconds,color='#ba4040',alpha=.2)
    r3=read(HERE/'20260921_r3_v1/raw/reference_events.csv')
    truth=r3[r3.window_id.eq(wid)].event_time_seconds.to_numpy(float)
    pred=pe[pe.window_id.eq(wid)].event_time_seconds.to_numpy(float)
    axs[0].set(title='R3-14: independent segments, no curve through timestamp gap',ylabel='Normalized signal');axs[0].legend()
    axs[1].eventplot([truth,pred],lineoffsets=[1,0],linelengths=.6,colors=['#c75c24','#087f8c'])
    axs[1].set(yticks=[0,1],yticklabels=['Partial prediction','R3 reference'],xlabel='Original viewing time (s)',xlim=(0,30))
    fig.savefig(ROOT/'figures/R3-14_partial_recovery.png',dpi=180);plt.close(fig)
    report='''# 久福失败链路诊断与修复验证

2026-09-21开始、2026-09-22完成本轮评分。固定R2 20260918参考：271登记窗、216完整事件参考、173有输出计数配对；R3采用19:21新版的11个久福可用完整复核窗，独立报告。原参考、默认模型、封存预测均不改。

## 结论

能解决一部分，但完整RR的跨场精度尚未解决。本轮已落实“时间缺口两侧保留事件输出”，R2全216窗F1从0.492010提高到0.536676；173计数配对窗R²仍为0.419629，MAE仍为5.156069次/min。不能用片段事件冒充完整30秒计数，也不能把增加输出覆盖写成YOLO精度提升。

![全量证据概览](figures/diagnosis_summary.png)

## 一、原R2全量的损失发生在哪里

| 阶段 | 全271窗证据 | R2完整参考损失 | 结论范围 |
|---|---|---|---|
| 原视频/时间采样 | 51窗有超过0.5 s时间间隔，2窗末端无充分支持；先于YOLO拒判 | 其中42个完整参考窗贡献557 FN | 不是检测器未识别；整窗事件拒判丢弃了两侧可用片段 |
| YOLO定位 | 2窗261帧全部no_detection | 其中1个完整参考窗贡献15 FN | R3-01的原图能看到牛头，但没有鼻孔坐标；后续无法提温 |
| ROI与温度可用性 | 216个输出窗中，173个计数配对窗的有效温度比例中位数约68.97% | 低覆盖8窗贡献36.52%计数平方误差 | 这是关联与错误贡献，不是因果贡献比例 |
| 温度映射实现 | 29个可用ROI按原图、原坐标、原半径和冻结RF重算一致 | 未发现抽查点的通道/坐标计算错位 | 不等于绝对温度或解剖中心准确，也不代表所有帧均核验 |
| 曲线与峰值 | 长时间双侧缺失仍被线性连接；输出173窗仍有1028 FP、1134 FN | 时间/相位不匹配和真实波形损失并存 | 预测与参考未匹配不能一概称生理伪峰 |

55个拒判里53个源于时间采样，这比笼统归因于“定位困难”更准确。时间拒判对应557/1706个FN，但它不直接进入原173计数配对窗的R²。因此：修复时间拒判可以改善事件覆盖，却不能据此宣称解释或解决0.4196的计数R²。

## 二、为什么173窗RR R²仍然低

| 至少一侧有温度的帧比例 | 计数配对窗数 | RR MAE / 次·min⁻¹ | 该组占总平方误差 |
|---|---:|---:|---:|
| 不超过25% | 8 | 16.2500 | 36.52% |
| 25%至50% | 39 | 5.1795 | 18.46% |
| 50%至75% | 55 | 4.6182 | 19.21% |
| 超过75% | 71 | 4.3099 | 25.81% |

8个低覆盖窗口只占4.62%的配对样本，却贡献36.52%的平方误差，直接影响R²中的误差平方和。它们不是从评估中删掉就算“修好”：缺失信号中未被观察到的呼吸不能靠直线插值恢复。高覆盖组MAE仍有4.31次/min，说明缺失也不是唯一因素，坐标/ROI的解剖正确性、非呼吸运动热变化及峰相位仍需进一步证据。

## 三、真实窗口的链路证据

### R3-01：在检测阶段中断

261帧无检测，温度全空。固定0、10、20 s三帧分别试原640、1280分辨率及检测框阈值0.05，27次探针中的该窗9次均无检测。这个小实验不支持“仅调分辨率/阈值即可修好”；不能推广到全部视频，也不能凭置信度证明鼻孔位置准确。

![R3-01 原帧](figures/R3-01_frames.png)

### R3-10：采到了少量信号，却被当作整窗曲线处理

有效温度仅约27.6%帧，双侧最长连续缺失约5.86 s；189个原本双侧无温度的网格位置在融合后有数值。红色区域是原样本缺失，虚线为用于处理的填补值。这些连线不是恢复的呼吸波形。示例侧转、低头与部分头部出画时检测缺失，不能简单要求算法在不可见处继续数呼吸。

![R3-10 原帧与ROI](figures/R3-10_frames.png)

![R3-10 温度到峰值链路](figures/R3-10_signal_chain.png)

### R3-02：总数相同也可能相位/事件位置不同

有效温度约90.8%，多数抽查帧有坐标与ROI，但原R2总数12对12仅匹配2对。原温度、融合波形和两次参考并列后可见：不同局部极值与参考的对应不稳定，不能仅凭曲线周期明显或计数相同断言逐次呼吸准确。R3也改变了总数，不能将差异解释为纯固定延迟；本轮没有把任何参考统一平移。

![R3-02 温度到峰值链路](figures/R3-02_signal_chain.png)

## 四、已经实现并全量验证的可修部分

只处理原53个时间拒判窗。保留原观看时间和0.5 s缺口界限，不跨缺口插值或压缩时间；连续片段单独用原YOLO/RF/ROI和峰值规则处理。沿用既有片段策略：缺口邻接端留3帧余量，核心至少6 s。6 s不是从久福人工计数拟合的。原216窗输出逐字保持独立原结果，53窗只新增partial事件，完整次数/RR仍为空。

| 参考与集合 | 原F1 | 分段事件方案F1 | Precision变化 | Recall变化 |
|---|---:|---:|---|---|
| R2全部216完整参考窗 | 0.492010 | 0.536676 | 0.562925→0.551340 | 0.436964→0.522772 |
| R3复核11窗 | 0.362832 | 0.470149 | 0.493976→0.504000 | 0.286713→0.440559 |

R2 F1差值+0.044666，按144个牛号聚类重采样2000次的95%区间为[0.026379,0.065122]。新增260 TP，同时新增261 FP；不是所有新增峰都可靠。42个原拒判完整参考窗F1从0升高、其余窗不变，不能把“0窗恶化”解读成不存在误检代价。R3差值区间约[−0.000001,0.284913]，小样本没有稳定增益保证。622个新增事件对应全部53窗，其中521个进入R2完整参考评价，不能混用分母。

采用范围：作为可选的部分事件输出，不能替换完整窗口RR，也不是未接触测试集上的新独立验证。保留既有默认流程。实际合并文件位于combined_opt_in，原prediction_status/predicted_count与新增event_output_status/partial_event_count分别保存。

![R3-14 分段恢复示例](figures/R3-14_partial_recovery.png)

## 五、针对低R²的另一项对照没有通过采用门槛

仅将曲线选择改为“原始有效样本的Lomb–Scargle周期功率×有效比例”，去趋势后只用真实有限值，频带沿用冻结20–100次/min，模型/ROI/峰规则均不改。没有用人工次数生成预测，先封存271窗结果再评分；216个原有输出的基线峰位置逐一重放一致。

| 集合 | 原R² / MAE | 周期评分R² / MAE | 决定 |
|---|---|---|---|
| R2 173配对窗 | 0.419629 / 5.1561 | 0.443540 / 4.9480 | 不采用 |
| R3 7配对窗 | 0.346154 / 2.5714 | −0.346154 / 3.7143 | 计数退化 |

R2 MAE差值约−0.2081，95%区间[−0.6286,0.2000]跨0；事件F1仅0.492010→0.494579，差值区间也跨0。因此不能把小幅点估计改善当成功创新。结果保留于observed_periodicity，默认流程未替换，不继续对这批已查看结果搜参数。

## 六、能解决到什么程度

- 已解决并验证的部分：不再因一个时间缺口而丢弃整个窗口所有可处理事件；作为显式partial输出提高事件覆盖。
- 尚未解决：原173配对窗RR的稳定泛化精度，以及全部输出事件的生理相位/解剖正确性。
- 不能用算法凭空恢复：真实掉帧、鼻孔长时间出画期间的具体呼吸次数。需要重新采集、同步参考或明确报告缺测；不能线性插值后称观察到完整呼吸。
- 下一项有针对性的补强应是定位/ROI的跨场验证与训练数据覆盖，尤其侧转、低头及热像对比变化。需要鼻孔位置/可见性参考才能量化定位误差；目前这些视觉观察不构成完整检测精度评价。

## 七、交付与验证

入口脚本均在上级目录：diagnose_chain.py、chain_signal_details.py、recover_timestamp_segments.py、score_timestamp_recovery.py、probe_chain_pixels.py、test_observed_periodicity.py、score_periodicity.py。脚本默认拒绝覆盖输出，重跑需要新的版本目录，不覆盖人工数据。

all271_stage_audit.csv记录每窗断点；signal_stage_details.csv和signal_coverage_strata.csv记录温度缺失与误差；pixel_probe记录29个ROI重算和27次检测探针；partial_timestamp保存53窗片段、622事件与推理前协议、封存哈希；recovery_evaluation为同R2/R3集合评分；observed_periodicity为未采用RR候选；combined_opt_in是可用的合并事件输出。

18项既有及新增相关测试通过；53窗片段/事件/源文件哈希核验、216原R2逐窗指标对齐、216原输出峰重放、29 ROI像素重算通过。图使用真实原帧、缓存曲线和导出事件，未用生成波形代替数据。未独立验证辐射测温、鼻孔解剖坐标金标准或生理呼吸相位。原稿不覆盖，本报告可作为论文新增机制分析和补充实验，不能宣称已解决完整跨场RR。
'''
    (ROOT/'诊断报告.md').write_text(report,encoding='utf-8')
    for r in read(ROOT/'input_hashes.csv').itertuples():assert sha256(Path(r.path))==r.sha256
    original=read(BASES['jiufu271']/'prediction_windows.csv').set_index('window_id')
    pd.testing.assert_series_equal(bp.set_index('window_id').predicted_count,original.predicted_count)
    pd.testing.assert_series_equal(bp.set_index('window_id').prediction_status,original.prediction_status)
    write_json(ROOT/'delivery_verification.json',dict(status='DIAGNOSIS_AND_BOUNDED_REPAIRS_VERIFIED',
        scope='full271 stage audit; R2 full216 events and173 counts; separate R3 review11 events and7 counts',
        tests_passed=18,ROI_replay_count=29,localization_probes=27,partial_inference_windows=53,
        original_input_hashes_unchanged=True,combined_whole_window_counts_unchanged=True,
        original_models_and_references_unchanged=True,
        successful_scope='opt-in partial-event output only',full_RR_solution='not established',
        rejected='observed-only periodicity selector',date='2026-09-22',
        report_sha256=sha256(ROOT/'诊断报告.md'),
        combined_files={p.name:sha256(p) for p in combined.iterdir() if p.is_file()}))
    print('Packaged report and combined partial-event outputs:',ROOT)


if __name__=='__main__':main()
