"""Verify pooled RF traces and show actual peaks against unchanged R3 events."""
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from analyze_transfer import read,write_csv,write_json,sha256
from run_oof_nostril_signal import OUT,FROZEN,TRANSFER
from probe_pose_mechanism import load_bgr


def main():
    sys.path.insert(0,str(FROZEN/'scripts'))
    import paper_repro_rr as rr
    protocol=json.loads((OUT/'protocol_before_predictions.json').read_text())
    cfg=rr.ReproConfig(**protocol['signal_config'])
    windows=read(OUT/'prediction_windows.csv')
    corr=read(OUT/'event_correspondence.csv')
    checked=0;trace=[]
    for p in windows.itertuples():
        folder=OUT/p.method/p.window_id
        if not (folder/'temperatures.csv').exists():continue
        temperatures=read(folder/'temperatures.csv');detections=read(folder/'detections.csv')
        pooled=detections.groupby('frame_index').temperature.mean().reindex(range(261))
        np.testing.assert_allclose(pooled,temperatures.left_temp,equal_nan=True,atol=1e-10)
        assert temperatures.right_temp.isna().all()
        curve=read(folder/'curve.csv');replay,_=rr.fuse_temperature_curve(temperatures,cfg,truth_row=None)
        np.testing.assert_array_equal(curve.is_peak,replay.is_peak)
        checked+=1
        count=detections.groupby('frame_index').size().reindex(range(261),fill_value=0).to_numpy()
        change=np.r_[False,np.diff(count)!=0]
        selected=corr[corr.cohort.eq('R3_full11')&corr.method.eq(p.method)&corr.window_id.eq(p.window_id)&corr.status.isin(['TP','FP'])]
        for e in selected.itertuples():
            i=int(round(e.predicted_time*8.7));lo=max(0,i-3);hi=min(261,i+4)
            trace.append(dict(method=p.method,window_id=p.window_id,status=e.status,predicted_time=e.predicted_time,
                reference_time=e.reference_time,detections_at_peak=int(count[i]),
                near_detection_count_change=bool(change[lo:hi].any()),
                near_missing_temperature=bool(temperatures.left_temp.iloc[lo:hi].isna().any())))
    assert checked==18
    write_csv(OUT/'peak_continuity_audit.csv',trace)
    write_json(OUT/'verification.json',dict(status='PASS',replayed_curves=checked,pooled_temperature_recomputed=True,
        all18_peak_arrays_reproduced=True,original_R3_full11_reproduced=[41,42,102],
        scope='Saved detection temperatures pooled and signal calculation replayed; not independent physiological truth',
        default_changed=False))
    figures=OUT/'figures';figures.mkdir(exist_ok=False)
    chosen=[]
    for status in ['TP','FP']:
        g=corr[corr.cohort.eq('R3_full11')&corr.method.eq('unfrozen_direct')&corr.status.eq(status)].sort_values(['window_id','predicted_time'])
        if g.empty:continue
        e=g.iloc[0];i=int(round(e.predicted_time*8.7));folder=OUT/'unfrozen_direct'/e.window_id
        frame=load_bgr(TRANSFER/'windows'/e.window_id/'frames'/f'frame_{i:06d}.jpg')[:,:,::-1]
        detections=read(folder/'detections.csv');curve=read(folder/'curve.csv');t=np.arange(len(curve))/8.7
        fig=plt.figure(figsize=(13,7));gs=fig.add_gridspec(2,2,width_ratios=[.8,1.5])
        ax=fig.add_subplot(gs[:,0]);ax.imshow(frame)
        for d in detections[detections.frame_index.eq(i)].itertuples():ax.add_patch(Circle((d.x,d.y),20,fill=False,color='red',lw=1.5))
        ax.set_xlim(-.5,frame.shape[1]-.5);ax.set_ylim(frame.shape[0]-.5,-.5);ax.axis('off');ax.set_title(f'Predicted ROI at {e.predicted_time:.3f}s')
        ax1=fig.add_subplot(gs[0,1]);ax1.plot(t,curve.left_temp,'.-',label='Observed pooled ROI temperature',lw=.8,ms=2)
        ax1.set_ylabel('RF temperature (C)');ax1.legend(fontsize=8)
        ax2=fig.add_subplot(gs[1,1]);ax2.plot(t,curve.smoothed_norm,label='Smoothed normalized signal')
        peaks=curve.is_peak.to_numpy(bool);ax2.scatter(t[peaks],curve.smoothed_norm[peaks],c='black',s=12,label='Predicted peaks')
        refs=corr[corr.cohort.eq('R3_full11')&corr.method.eq('unfrozen_direct')&corr.window_id.eq(e.window_id)].reference_time.dropna().unique()
        for a in [ax1,ax2]:
            for value in refs:a.axvline(value,color='green',alpha=.3,lw=.8)
            a.axvline(e.predicted_time,color='red',lw=1);a.set_xlim(0,30);a.grid(alpha=.2)
        ax2.set_xlabel('Time (s)');ax2.set_ylabel('Normalized value');ax2.legend(fontsize=8)
        fig.suptitle(f'{status} example | {e.window_id} | green: unchanged human events',fontsize=10)
        fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(figures/f'{status}.png',dpi=160);plt.close(fig)
        chosen.append(dict(status=status,window_id=e.window_id,predicted_time=float(e.predicted_time)))
    write_json(OUT/'figure_selection.json',dict(rule='first window/time for TP and FP, no manual best-case selection',examples=chosen))
    summary=read(OUT/'metrics.csv')
    lines=['# 折外连续鼻孔温度与呼气事件对照','',
        '本轮不采用：点定位改善未稳定转为呼吸率改善。默认模型、人工事件、RF模型和既有RR输出均未改。','',
        '## 数据与比较口径','',
        '12个既有复核视频，每视频用未训练过该牛的折外模型。9窗沿用原261帧/30秒时间网格运行，3窗仍按原时间戳规则拒判。R3完整窗口参考11窗；严格可见呼气相位参考9窗。R3-11因既有参考冲突未纳入完整事件评分，不因新预测修改参考。',
        '冻结与解冻直接检测器使用相同的r20、BGR随机森林、>20℃像素保留、当帧可用鼻孔ROI温度均值及同一平滑/数峰规则。其温度流无解剖左右身份，不能冒充原左右质量融合。原流程列为背景基线，只有两个直接检测分支是相同连续处理协议。','',
        '| 参考 | 方法 | 事件窗 | 有输出窗 | 共同计数窗 | RR MAE | RR R² | TP/FP/FN | 事件F1 |',
        '|---|---|---:|---:|---:|---:|---:|---|---:|']
    for r in summary.itertuples():lines.append(f'| {r.cohort} | {r.method} | {r.event_windows} | {r.output_windows} | {r.common_count_pairs} | {r.rr_mae_common:.4f} | {r.rr_r2_common:.4f} | {r.tp}/{r.fp}/{r.fn} | {r.f1:.4f} |')
    lines+=['','RR只比较三方法均有输出的同窗，拒判不补0次；事件召回纳入所有合格参考窗口，无输出就是没有预测事件。不能将这些7窗/6窗R²称为173配对或全271指标。事件匹配固定0.3秒、一对一；时间匹配只是与既有人工参考一致，不是独立生理真值。',
        '', '## 失败仍发生在哪一步','',
        'R3-01在解冻分支仅12/261帧有有效温度，最终2个峰；该窗有16个人工确认事件。原来整窗无信号现在产生输出，不等于已经可靠恢复呼吸。',
        '事件F1和次数误差并不等价：本轮事件匹配略改善，但新增/丢失峰的数量关系仍使共同窗口RR误差增大。检测数量切换及温度缺失邻域可在peak_continuity_audit.csv逐峰查看；这些是关联，不作为已证实的伪峰成因。',
        '均值池化避免了无序检测框按置信度交换产生的身份跳变，但在两个鼻孔变一个鼻孔时仍可能产生温度基线变化；原缺失修复也不能恢复不存在的呼吸信息。后续须区分鼻孔身份、连续可观测性及真实温度周期，不能凭单帧命中率宣称问题解决。',
        '', '## 真实预测与温度曲线','']
    for e in chosen:lines += [f'### {e["status"]}示例',f'![{e["status"]}](figures/{e["status"]}.png)','']
    lines+=['## 文件与验证','',
        '- prediction_windows.csv / prediction_events.csv：两个候选的全部状态与事件，含拒判。',
        '- event_correspondence.csv：每个TP、FP、FN与预测/参考时间。',
        '- common_count_pairs.csv：共同支持集RR计算明细。',
        '- 各方法/窗口目录：逐帧检测框、ROI温度、池化温度及曲线。',
        '- verification.json：18条候选曲线均值与峰数组回放通过；原流程R3十一窗41TP/42FP/102FN复现。']
    (OUT/'折外连续信号对照报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(OUT/'adoption_decision.json',dict(disposition='DO_NOT_ADOPT_CONTINUOUS_BRANCH',
        reason='RR MAE on identical output windows worsened despite event F1 gain',
        default_changed=False,not_all271=True,goal_complete=False))
    print('PASS:18 traces and peak arrays replayed. Continuous branch not adopted.')


if __name__=='__main__':main()
