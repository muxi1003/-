"""Real-image and signal evidence for temporally matched and unmatched peaks."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from analyze_transfer import read,write_csv,write_json
from audit_frozen_peak_evidence import ROOT,OUT,TRANSFER
from probe_pose_mechanism import load_bgr


def main():
    peaks=read(OUT/'emitted_peak_evidence.csv')
    refs=read(OUT/'reference_matching.csv')
    phase=read(ROOT/'phase_audit/reference_event_audit.csv')
    merged=refs.merge(phase[['window_id','event_id','classification']],on=['window_id','event_id'],validate='one_to_one')
    write_csv(OUT/'reference_loss_stage.csv',merged)
    stages=merged[merged.strict_visible_phase].groupby(['matching_status','classification']).size().reset_index(name='events')
    write_csv(OUT/'strict_reference_stage_summary.csv',stages)
    figdir=OUT/'figures';figdir.mkdir(exist_ok=False)
    cases=[('matched_direct',peaks[peaks.temporal_match.eq('TP')&peaks.all_contributors_direct]),
           ('unmatched_direct',peaks[peaks.temporal_match.eq('FP')&peaks.all_contributors_direct]),
           ('matched_with_interpolation',peaks[peaks.temporal_match.eq('TP')&peaks.any_temperature_interpolated])]
    chosen=[]
    for kind,subset in cases:
        p=subset.sort_values(['review_id','peak_seconds']).iloc[0]
        chosen.append(dict(case_type=kind,window_id=p.window_id,event_id=p.prediction_event_id))
        folder=TRANSFER/p.window_id;curve=read(folder/'curve.csv');temperature=read(folder/'temperatures.csv')
        index=int(round(p.peak_seconds*8.7));row=temperature.iloc[index]
        image=load_bgr(folder/'frames'/row.frame_name)[:,:,::-1]
        fig=plt.figure(figsize=(12,6),layout='constrained');gs=fig.add_gridspec(2,2,width_ratios=[1,2])
        aximage=fig.add_subplot(gs[:,0]);axraw=fig.add_subplot(gs[0,1]);axcurve=fig.add_subplot(gs[1,1])
        aximage.imshow(image);aximage.axis('off')
        for side,color in [('left','#20ccff'),('right','#ff9d24')]:
            if np.isfinite(row[side+'_x']) and np.isfinite(row[side+'_y']):
                radius=float(row.get('adaptive_roi_radius',20))
                aximage.add_patch(Circle((row[side+'_x'],row[side+'_y']),radius,
                    fill=False,edgecolor=color,linewidth=1.5,
                    linestyle='-' if row[side+'_source']=='detected' else '--'))
        aximage.set_title('Original frame + saved ROI\nsolid: detected; dashed: inferred',fontsize=9)
        t=curve.target_time_seconds.to_numpy(float);low=max(0,p.peak_seconds-2.5);high=min(t[-1],p.peak_seconds+2.5)
        mask=(t>=low)&(t<=high)
        for side,color in [('left','#1268a4'),('right','#cf6911')]:
            axraw.plot(t[mask],curve.loc[mask,side+'_temp_used'],color=color,alpha=.45,linestyle='--',label=side+' used')
            axraw.plot(t[mask],temperature.loc[mask,side+'_temp'],'.-',color=color,markersize=3,label=side+' raw')
        axraw.set(ylabel='RF temperature proxy (C)',title='Solid samples vs repaired series')
        axraw.legend(fontsize=7,ncol=2);axraw.grid(alpha=.2)
        axcurve.plot(t[mask],curve.loc[mask,'smoothed_norm'],color='#225c50',label='Frozen smoothed signal')
        emitted=peaks[peaks.window_id.eq(p.window_id)&peaks.peak_seconds.between(low,high)]
        for e in emitted.itertuples():
            j=int(round(e.peak_seconds*8.7))
            axcurve.plot(e.peak_seconds,curve.smoothed_norm.iloc[j],'o',color='#174778',markersize=5)
        localrefs=refs[refs.window_id.eq(p.window_id)&refs.reference_seconds.between(low,high)]
        for k,r in enumerate(localrefs.itertuples()):
            axcurve.axvline(r.reference_seconds,color='#b84366',alpha=.7,linestyle=':',label='R3 human event' if k==0 else None)
        for ax in [axraw,axcurve]:
            ax.axvline(p.peak_seconds,color='#111111',linestyle='--',linewidth=1,label='Audited peak' if ax is axcurve else None)
            ax.set_xlim(low,high)
        axcurve.set(xlabel='Window time (s)',ylabel='Normalized value',title='Event timing; not physiological proof')
        axcurve.legend(fontsize=7);axcurve.grid(alpha=.2)
        fig.suptitle(f'{p.review_id} / {p.prediction_event_id}: {kind}; fusion={p.selected_mode}',fontsize=12)
        fig.savefig(figdir/(kind+'.png'),dpi=160);plt.close(fig)
    write_csv(OUT/'figure_case_selection.csv',chosen)
    metrics=[]
    for scope,d,r in [('eligible11',peaks,refs),('strict9',peaks[peaks.strict_visible_phase],refs[refs.strict_visible_phase])]:
        for mode,selected in [('frozen_all',d),('diagnostic_keep_all_direct_only',d[d.all_contributors_direct])]:
            tp=int(selected.temporal_match.eq('TP').sum());fp=len(selected)-tp;fn=len(r)-tp
            metrics.append(dict(scope=scope,mode=mode,tp=tp,fp=fp,fn=fn,f1=2*tp/(2*tp+fp+fn),
                                matching='fixed original one-to-one pairs; no re-matching after subsetting'))
    write_csv(OUT/'fixed_matching_subset_diagnostic.csv',metrics)
    summary=read(OUT/'summary.csv')
    text='''# 实际输出峰的事件与温度来源核验

2026-09-23；N067实验附件。只审计已保存冻结输出，不改变时间、标签、阈值或呼吸次数。

## 1. 哪些峰可以报告为匹配人工呼气参考

在11个久福R3完整有效复核窗口中，共143个人工确认事件、83个冻结输出峰。保持原0.3秒容差和一对一最大匹配，41峰匹配、42峰未匹配、102个人工事件漏掉，F1=0.362832。

逐峰对应见`emitted_peak_evidence.csv`：每行给出视频、预测峰ID、峰时间、匹配的人工事件ID及时间差。`reference_matching.csv`保留全部143事件，包括FN。41峰只能称“匹配R3人工呼气参考”，不是独立生理仪器金标准；未匹配也可能包含人工参考时间误差，不自动断言每个峰都是生理假峰。

排除仅热模式相位等非严格窗口后，严格9窗有117事件、68输出峰，其中TP33、FP35、FN84。不能把这个小复核集当271窗全量。

## 2. 匹配与来源不是同一件事

每个输出峰已回查原始左右温度及实际融合方式，并验证保存曲线等于MAF3计算。对构成峰的3个时间样本，按实际归一化min/max/mean/单侧参与关系追踪来源，不能借另一未参与融合的鼻孔“检测成功”来证明当前峰可靠。min/max相等时保守保留两侧。

| 11窗输出峰类别 | 数量 | 全部参与温度均为直接检测 | 至少一个推测ROI | 至少一个插值温度 |
|---|---:|---:|---:|---:|
'''
    for row in summary[summary.scope.eq('eligible11')].itertuples():
        text+=f'| {row.temporal_match} | {row.peaks} | {row.all_contributors_direct} | {row.at_least_one_inferred_ROI} | {row.any_temperature_interpolated} |\n'
    text+='''
推测ROI和插值列可能重叠，不能相加作为总数。`raw_direct`仅表示代码中source=detected且原温度未改，不代表鼻孔位置已由人工逐帧证实。`raw_inferred`仍是从某个实际图像ROI读取的像素温度，但该ROI坐标来源为推测，不等于温度插值。

**关键发现：42个未匹配峰中，23个完全由直接检测温度构成。**所以即使去掉所有插值/推测ROI峰，也不能解决所有错误。另一方面，41个匹配峰中有12个并非全直接来源，一刀切也会删除匹配事件。

作为计数诊断，若固定原匹配关系只保留全直接峰，则剩29TP、23FP、114FN，F1=0.297436，低于原0.362832。这不是重跑算法或重新匹配后的新方法指标，也不是新的RR R²；它只说明“来源更严格”不能自动保证呼气检测更好。

## 3. 人工位置参考能验证到哪里

36张已标位置的图片只在0/10/20秒附近，不能把其椭圆传播到未标帧。本轮只允许与原保存图路径完全一致时引用人工位置参考；不以“时间差很小”冒充同图标注。

83个峰中仅3个的MAF3贡献帧有至少一帧落到已有人工位置参考，**没有任何一个峰的全部贡献帧都具备正确的人工解剖位置对应**。这不是说83个峰均错误，而是现有稀疏位置标注不能证明它们的全部温度都来自正确鼻孔。已有36帧仍有效，不需重复标注。

进一步区分漏检阶段见`reference_loss_stage.csv`和`strict_reference_stage_summary.csv`。它们把实际TP/FN与先前的“无可用ROI样本/附近有极值/附近无极值”关联，不新增或平移事件，不把最近极值距离当一对一匹配结果。

## 4. 完成边界

严格9窗的84个FN进一步按实际冻结流程追溯（`strict_FN_route_evidence.csv`）：47个来自整窗拒判，其中R3-01无有效温度16个、R3-03/R3-14时间长缺口共31个；其余37个中，11个参考附近没有有效ROI温度、20个有温度但没有原始极值邻近、4个原始极大值只在未选鼻孔、2个涉及所选信号/峰规则差异。这里按先发生的可观测流程条件分类，不把它们称为互斥生理原因。该表由trace_missed_peak_routes.py在初次报告后生成。

分段诊断在被拒判窗口中找到的极值不是冻结整窗预测，不能把47个拒判漏事件都归为数峰失败，或用恢复这些事件声称173个有输出RR配对问题已解决。4个未选鼻孔候选也不代表可无代价找回4次真实呼吸。

本轮已给出实际输出峰的逐事件对应和可核查温度来源。仍不能把所有温度极大值统一解释为呼气，也不能把检测坐标当作已证明的鼻孔解剖位置。需要区分有图像依据的R3人工相位、算法温度极值以及逐帧ROI正确性。

旧标注最终人工核准版本尚未确认，因此没有重训。当前没有成功的新定位或RR算法；已有负结果和原R2/R3保持不变。不要把补充证据表写成方法性能提升。

## 5. 实际图像与曲线示例

按review_id和峰时间，分别选择首个“匹配且全直接”“未匹配且全直接”“匹配但含插值”案例，不挑最漂亮的曲线。原图完整显示；ROI实线为直接检测、虚线为推测，显示的是保存坐标而非新人工真值。
'''
    for kind,_ in cases:text+=f'\n![{kind}](figures/{kind}.png)\n'
    text+='\n验证：7项来源逻辑测试通过；83个原峰和保存MAF3数值一致、41/42/102匹配总数与旧R3评分一致。\n'
    (OUT/'逐峰证据核验报告.md').write_text(text,encoding='utf-8')
    print(stages.to_string(index=False));print('Peak evidence report and three real-image signal panels written.')


if __name__=='__main__':main()
