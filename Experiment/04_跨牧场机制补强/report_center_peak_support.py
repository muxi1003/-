"""Pair rejected frozen peaks with existing R3 event review and real curves."""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd

from analyze_transfer import HERE, HOLDOUT, read, sha256, write_csv, write_json
from audit_center_peak_support import OUT


TRANSFER=HOLDOUT/'external_transfer/20260914_v2/windows'
EVIDENCE=HERE/'20260922_pose_phase_v1/frozen_peak_evidence_v1'


def case_figure(row, output):
    folder=TRANSFER/row.window_id
    curve=read(folder/'curve.csv')
    temps=read(folder/'temperatures.csv')
    index=int(row.frame_index)
    side=row.selected_mode if row.selected_mode in ['left','right'] else 'left'
    t=np.arange(len(curve))/8.7
    source=folder/'frames'/str(temps.iloc[index].frame_name)
    image=cv2.imdecode(np.fromfile(source,np.uint8),cv2.IMREAD_COLOR)
    assert image is not None
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,6),gridspec_kw={'width_ratios':[.8,1.4]})
    ax1.imshow(image)
    x,y=temps.iloc[index][[side+'_x',side+'_y']]
    observed=temps.iloc[index][side+'_source']=='detected'
    if np.isfinite(x) and np.isfinite(y):
        ax1.add_patch(Circle((x,y),20,fill=False,color='lime' if observed else 'orange',lw=1.8))
        ax1.plot(x,y,'+',color='lime' if observed else 'orange')
    ax1.set_xlim(-.5,image.shape[1]-.5);ax1.set_ylim(image.shape[0]-.5,-.5);ax1.axis('off')
    ax1.set_title(f'Frame {index}; {side} source={temps.iloc[index][side+"_source"]}')
    raw=temps[side+'_temp'].where(temps[side+'_source'].eq('detected'))
    ax2.plot(t,curve[side+'_temp_used'],color='#d98418',lw=1,label=f'{side} temperature used')
    ax2.scatter(t[raw.notna()],raw.dropna(),s=7,color='#157f6b',label=f'{side} direct RF')
    ax2b=ax2.twinx()
    ax2b.plot(t,curve.smoothed_norm,color='#225eaa',lw=1,label='smoothed fused signal')
    ax2b.set_ylabel('Normalized signal')
    ax2.axvline(float(row.event_time_seconds),color='#d12b2b',lw=1.5,label='rejected peak')
    if np.isfinite(row.reference_seconds):
        ax2.axvline(float(row.reference_seconds),color='green',alpha=.7,lw=1.5,label='R3 reviewed event')
    ax2.set_xlim(max(0,float(row.event_time_seconds)-2),min(30,float(row.event_time_seconds)+2))
    ax2.set_xlabel('Time (s)');ax2.set_ylabel('Temperature (C)');ax2.grid(alpha=.25)
    ax2.legend(loc='best',fontsize=8)
    fig.suptitle(f'{row.temporal_match}: {row.review_id}, {row.event_id}; ellipse position not asserted',fontsize=11)
    fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(output,dpi=170);plt.close(fig)


def main():
    seal=json.loads((OUT/'prediction_seal.json').read_text(encoding='utf-8'))
    for name,digest in seal['files'].items():assert sha256(OUT/name)==digest
    support=read(OUT/'peak_support.csv')
    evidence=read(EVIDENCE/'emitted_peak_evidence.csv')
    joined=support.merge(evidence,left_on=['window_id','event_id'],
                         right_on=['window_id','prediction_event_id'],how='inner',validate='one_to_one')
    assert len(joined)==83
    removed=joined[~joined.retained]
    counts=removed.groupby('temporal_match').size().to_dict()
    assert len(removed)==22 and counts=={'FP':17,'TP':5}
    write_csv(OUT/'R3_peak_reference_crosscheck.csv',joined)
    cases=[];figures=OUT/'figures';figures.mkdir(exist_ok=False)
    for status in ['TP','FP']:
        sample=removed[removed.temporal_match.eq(status)].sort_values(['window_id','frame_index']).iloc[0]
        case_figure(sample,figures/f'rejected_{status}.png')
        cases.append(dict(status=status,review_id=sample.review_id,window_id=sample.window_id,
                          event_id=sample.event_id,frame_index=int(sample.frame_index),
                          peak_seconds=float(sample.event_time_seconds),reference_seconds=float(sample.reference_seconds)
                          if np.isfinite(sample.reference_seconds) else None))
    write_json(OUT/'figure_selection.json',dict(rule='first rejected TP and first rejected FP sorted by window/frame',cases=cases))
    summary=read(OUT/'metrics.csv').set_index(['cohort','method'])
    b=summary.loc[('R2_complete216','original')];c=summary.loc[('R2_complete216','center_support')]
    status='DO_NOT_ADOPT' if c.f1<b.f1 or c.rr_mae>b.rr_mae else 'REQUIRES_MORE_VALIDATION'
    write_json(OUT/'adoption_decision.json',dict(disposition=status,
        reason='Same-reference event F1 decreased and 173-pair RR MAE increased',
        removed_peaks_all271=int(seal['original_peaks']-seal['kept_peaks']),
        removed_matched_R3_TP=5,removed_unmatched_R3_FP=17,
        candidate_generation_script_modified_after_seal='Scoring block moved to separate script after an argument error; source hash in protocol identifies executed pre-split version',
        no_reference_or_default_change=True,goal_complete=False))
    text=['# 峰位真实测温支持：271窗固定规则对照','',
      '规则：已选融合信号的鼻孔在峰帧直接测温，或同一鼻孔在峰帧前后各1帧都直接测温，则保留原峰；其余删去。只筛选原峰，不新增峰、不移动时间，不改变时间戳拒判。',
      '该规则来自可观测性假设，数据已用于前期诊断，属于回顾性实验。预测事件清单封存之后才读取R2/R3人工事件。',
      '', '| R2，216事件参考窗 / 173 RR配对窗 | 原流程 | 峰位支持规则 |',
      '|---|---:|---:|',
      f'| 事件TP / FP / FN | {int(b.tp)} / {int(b.fp)} / {int(b.fn)} | {int(c.tp)} / {int(c.fp)} / {int(c.fn)} |',
      f'| 事件F1 | {b.f1:.6f} | {c.f1:.6f} |',
      f'| RR MAE (次/分钟) | {b.rr_mae:.6f} | {c.rr_mae:.6f} |',
      f'| RR R² | {b.rr_r2:.6f} | {c.rr_r2:.6f} |',
      '', f'271窗共有{int(seal["original_peaks"])}个原始峰，删去{int(seal["original_peaks"]-seal["kept_peaks"])}个；55个原拒判窗口保持拒判。R2完整216窗中删去442个峰，其中事件TP净减少174、FP净减少268。误报减少，但漏检增加且同窗RR误差增大。',
      'R3完整11窗的83个原峰中删去22个：原先匹配人工事件的5个也被删去。因此不能把“峰帧缺测”直接判为伪呼气。',
      '', '## 原图和曲线示例','',
      '红线为被筛掉的原峰，绿线为R3已有人工呼气时间。绿色ROI圈表示该帧直接检测，橙色圈表示可用坐标非直接测温；无圈表示没有可显示坐标。圆圈均来自算法坐标，不是人工鼻孔真值。']
    for case in cases:
        text.extend(['',f'### 被删{case["status"]}：{case["review_id"]}，{case["peak_seconds"]:.3f}秒',
                    f'![rejected {case["status"]}](figures/rejected_{case["status"]}.png)'])
    text.extend(['','## 判断','',
      '默认流程保留。峰位附近真实测温是必要的信号来源信息，但不是足以确定呼气相位的条件：时间匹配R3也受人工可见性影响，只有少数输出峰具有同帧人工鼻孔区域核对。',
      'peak_support.csv保留271窗每个原峰的保留决定；R3_peak_reference_crosscheck.csv标出已人工复核事件的TP/FP；event_window_results.csv和rr_pairs.csv保留同口径复算。',
      '评分入口在生成事件清单封存后移到score_center_peak_support.py，原运行协议中的脚本哈希对应分离评分前的执行版本。候选筛选逻辑未改变。'])
    (OUT/'峰位真实测温支持对照报告.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print('PASS:',len(joined),'R3 peaks crossed,',len(removed),'removed;',status)


if __name__=='__main__':main()
