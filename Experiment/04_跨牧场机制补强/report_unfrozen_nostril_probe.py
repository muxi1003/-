"""Compare the fixed frozen/unfrozen protocols on identical cow-held-out frames."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from analyze_transfer import read,write_csv,write_json
from train_nostril_unfrozen_probe import ROOT,OUT,CONTROL,REF
from probe_pose_mechanism import load_bgr
from evaluate_nostril_reference import match_regions,outside_fraction


def main():
    old=read(CONTROL/'evaluation/per_frame.csv')
    new=read(OUT/'evaluation/per_frame.csv')
    a=old[old.method.eq('direct_nostril_oof')].set_index('frame_id')
    b=new[new.method.eq('direct_nostril_oof')].set_index('frame_id')
    assert set(a.index)==set(b.index) and len(a)==36
    change=pd.DataFrame(dict(frame_id=a.index,cow_id=a.cow_id.values,
        old_correct=a.correct.values,new_correct=b.loc[a.index].correct.values,
        delta_correct=(b.loc[a.index].correct-a.correct).values,
        old_wrong=a.false_or_off_target.values,new_wrong=b.loc[a.index].false_or_off_target.values))
    write_csv(OUT/'freeze_paired_changes.csv',change)
    cow=change.groupby('cow_id')[['old_correct','new_correct','old_wrong','new_wrong']].sum()
    expected=a.groupby('cow_id').expected.sum().reindex(cow.index).to_numpy()
    rng=np.random.default_rng(20260923);draw=rng.integers(0,len(cow),size=(5000,len(cow)))
    totals=cow.to_numpy()[draw].sum(axis=1);denom=expected[draw].sum(axis=1)
    recall_delta=(totals[:,1]-totals[:,0])/denom
    intervals=dict(cows=len(cow),resamples=5000,seed=20260923,
        recall_change=float((b.correct.sum()-a.correct.sum())/a.expected.sum()),
        cluster_bootstrap_recall_change_CI95=np.quantile(recall_delta,[.025,.975]).tolist(),
        limitation='Paired resampling of12 selected cows; not independent external confirmation or causal isolation of unarchived initialization randomness')
    write_json(OUT/'freeze_uncertainty.json',intervals)
    ref=json.loads(REF.read_text(encoding='utf-8'))['records']
    predictions={v:read(folder/'out_of_fold_predictions.csv') for v,folder in [('frozen',CONTROL),('unfrozen',OUT)]}
    roi=[]
    for variant,pred in predictions.items():
        for fid in a.index:
            regions=[e for e in ref[fid]['regions'].values() if e is not None]
            pp=pred[pred.frame_id.eq(fid)]
            points=pp[['x','y']].to_numpy(float)
            for i,j,d in match_regions(points,regions):
                roi.append(dict(variant=variant,frame_id=fid,human_region=j,center_inside=d<=1,
                    outside_r20=outside_fraction(points[i],20,regions[j],1080,1440)))
    write_csv(OUT/'freeze_ROI_pairs.csv',roi)
    common=pd.DataFrame(roi).pivot(index=['frame_id','human_region'],columns='variant',values='outside_r20').dropna()
    write_json(OUT/'freeze_ROI_summary.json',dict(paired_regions=len(common),
        frozen_median=float(common.frozen.median()),unfrozen_median=float(common.unfrozen.median()),
        median_paired_change=float((common.unfrozen-common.frozen).median()),
        restriction='Fixed r20 diagnostic; matched-only subset, approximate anatomical ellipses, not absolute background truth'))
    figures=OUT/'figures';figures.mkdir(exist_ok=False)
    cases=[]
    for kind,ordered in [('gain',change.sort_values(['delta_correct','frame_id'],ascending=[False,True])),
                         ('loss',change.sort_values(['delta_correct','frame_id']))]:
        row=ordered.iloc[0]
        if (kind=='gain' and row.delta_correct<=0) or (kind=='loss' and row.delta_correct>=0):continue
        fid=row.frame_id;image=load_bgr(ROOT/'frames'/f'{fid}.png')[:,:,::-1]
        fig,axs=plt.subplots(1,2,figsize=(10,7.4))
        for ax,variant in zip(axs,['frozen','unfrozen']):
            ax.imshow(image)
            for e in ref[fid]['regions'].values():
                if e:ax.add_patch(Ellipse((e['cx'],e['cy']),2*e['rx'],2*e['ry'],fill=False,color='lime',lw=1.4))
            for p in predictions[variant][predictions[variant].frame_id.eq(fid)].itertuples():
                ax.add_patch(Circle((p.x,p.y),20,fill=False,color='red',lw=1.3))
                ax.plot(p.x,p.y,'r+',ms=6)
            ax.set_xlim(-.5,image.shape[1]-.5);ax.set_ylim(image.shape[0]-.5,-.5);ax.axis('off');ax.set_title(variant)
        fig.suptitle(fid+' | green: human ellipse; red: predicted center and r20',fontsize=10)
        fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(figures/f'{kind}.png',dpi=170);plt.close(fig)
        cases.append(dict(kind=kind,frame_id=fid,delta_correct=int(row.delta_correct)))
    write_json(OUT/'figure_selection.json',dict(rule='largest gain/loss in correct count then lexical ID',cases=cases))
    summaries=[]
    for label,folder,method in [('original_pose',CONTROL,'original_pose'),('frozen_direct200',CONTROL,'direct_nostril_oof'),
                              ('unfrozen_direct200',OUT,'direct_nostril_oof'),('unfrozen_fallback',OUT,'no_keypoint_fallback_oof')]:
        s=read(folder/'evaluation/summary.csv').set_index('method').loc[method].to_dict();s['variant']=label;summaries.append(s)
    write_csv(OUT/'comparison_summary.csv',summaries)
    fit=read(OUT/'training_fit_diagnostic/summary.csv')
    lines=['# 解除骨干冻结：按牛三折鼻孔检测对照','',
        '36帧、12牛、50人工鼻孔区域；每折8牛24帧训练，4牛12帧评分。冻结对照与新对照均200轮last.pt，conf=0.25，iou=0.7，max_det=2，imgsz=640。',
        '', '训练配置只改freeze=10→0；沿用相同初始权重文件与trainer随机种子。但历史对照未归档trainer之前的随机头初始化，因此不是逐参数初始状态完全相同的因果实验。久福已被多轮查看，只作回顾性适配证据。','',
        '| 方法 | 预测点 | 正确点/50 | 精确率 | 召回率 | 点F1 |', '|---|---:|---:|---:|---:|---:|']
    for s in summaries:lines.append(f'| {s["variant"]} | {s["predicted_points"]} | {s["correct_points"]} | {s["precision"]:.4f} | {s["recall"]:.4f} | {s["point_f1"]:.4f} |')
    lines+=['',f'相对冻结检测器：正确点净变化{int(change.delta_correct.sum()):+d}；{int((change.delta_correct>0).sum())}帧改善、{int((change.delta_correct<0).sum())}帧恶化。',
        f'12牛配对bootstrap召回变化95%区间：{intervals["cluster_bootstrap_recall_change_CI95"]}。不能把重复查看的12牛当成新外测。',
        '', '## 训练拟合与折外差距','']
    for r in fit[fit.threshold.eq(.25)].itertuples():
        lines.append(f'- {r.subset}：{r.correct}/{r.expected}命中，预测{r.predicted}点，精确率{r.precision:.4f}。')
    lines+=['','训练集按帧-模型对统计，同一帧会出现在另外两折模型训练中，不是新增72个独立帧。0.001阈值只保留为诊断，不用于主结果或选阈值。',
        '', '## 原图案例','']
    for c in cases:lines += [f'### {c["kind"]}：{c["frame_id"]}',f'![{c["kind"]}](figures/{c["kind"]}.png)','']
    lines+=['## 边界','', '鼻孔框由人工椭圆外接矩形定义，检测框中心作为预测点；不是分割网络。ROI仍是r20诊断，未将人工椭圆用于自动温度提取。',
        '本轮没有改变默认模型、RF温度映射、呼气事件参考或RR结果。evaluation/adoption_decision.json仅沿用兜底组合门槛，直接检测器与兜底组合必须分别判断。即使直接点定位改善，也仍需连续温度信号及呼吸事件验证，不能直接部署。',
        f'共同{len(common)}个有分配点的人工区域中，r20 ROI越界比例中位数：冻结{common.frozen.median():.6f}、解冻{common.unfrozen.median():.6f}，逐对差中位数{(common.unfrozen-common.frozen).median():.6f}。条件子集不代表全部鼻孔的ROI准确性。']
    (OUT/'解除冻结对照报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(pd.DataFrame(summaries).to_string(index=False));print(intervals)


if __name__=='__main__':main()
