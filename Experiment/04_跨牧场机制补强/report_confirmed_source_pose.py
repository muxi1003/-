"""Evidence report using complete real frames and sealed predictions."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, Rectangle
from analyze_transfer import read, write_json
from train_confirmed_source_pose import OUT
from probe_pose_mechanism import ROOT, load_bgr
from score_confirmed_source_pose import high_points


def main():
    summary=read(OUT/'anatomy_summary.csv').set_index('variant')
    source=read(OUT/'source_validation_summary.csv').set_index('variant')
    detail=read(OUT/'anatomy_frame_results.csv')
    ref=json.loads((ROOT/'anatomy_reference_142648_v1/reference_original.json').read_text(encoding='utf-8'))['records']
    predictions={v:{p['frame_id']:p for p in json.loads((OUT/f'{v}_predictions.json').read_text(encoding='utf-8'))}
                 for v in ['baseline','candidate']}
    figures=OUT/'figures';figures.mkdir(exist_ok=False)
    diff=detail.pivot(index='frame_id',columns='variant',values='correct')
    diff['change']=diff.candidate-diff.baseline
    diff=diff.sort_values(['change','frame_id'],ascending=[False,True])
    examples=[]
    for category,ids in [('largest_gain',diff[diff.change>0].index[:1]),
                         ('largest_loss',diff.sort_values(['change','frame_id'])[lambda d:d.change<0].index[:1]),
                         ('unchanged_first',diff[diff.change==0].sort_index().index[:1])]:
        for fid in ids:
            fig,axs=plt.subplots(1,2,figsize=(10,7.4))
            for ax,variant in zip(axs,['baseline','candidate']):
                p=predictions[variant][fid]
                im=load_bgr(p['image_path'])[:,:,::-1]
                ax.imshow(im)
                for e in ref[fid]['regions'].values():
                    if e:
                        ax.add_patch(Ellipse((e['cx'],e['cy']),2*e['rx'],2*e['ry'],fill=False,color='lime',lw=1.5))
                if p['boxes']:
                    b=p['boxes'][int(np.argmax(p['box_conf']))]
                    ax.add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,color='cyan',lw=1))
                for x,y in high_points(p):
                    ax.add_patch(Circle((x,y),20,fill=False,color='red',lw=1.3))
                    ax.plot(x,y,'r+',ms=7)
                count=detail[(detail.variant==variant)&(detail.frame_id==fid)].iloc[0]
                ax.set_title(f'{variant}: {count.correct}/{count.expected} regions; {count.predicted} points')
                ax.axis('off')
            fig.suptitle(fid+' | green: human ellipse; red: predicted ROI r=20; cyan: predicted nose',fontsize=10)
            fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(figures/f'{category}.png',dpi=180);plt.close(fig)
            examples.append(dict(category=category,frame_id=fid,correct_point_change=int(diff.loc[fid,'change'])))
    write_json(OUT/'figure_selection.json',dict(rule='Largest per-frame correct-point gain/loss, then lexical ID; first unchanged ID',examples=examples))
    losses=read(OUT/'runs/fold0/results.csv')
    fig,axs=plt.subplots(1,3,figsize=(12,3.3))
    for ax,col in zip(axs,['train/box_loss','train/pose_loss','train/kobj_loss']):
        ax.plot(losses.epoch,losses[col]);ax.set_title(col);ax.set_xlabel('Epoch');ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(figures/'training_losses.png',dpi=160);plt.close(fig)
    b=summary.loc['baseline'];c=summary.loc['candidate']
    paired=json.loads((OUT/'paired_ROI_summary.json').read_text())
    location_promising=bool(c.correct_points>b.correct_points and c.point_precision>=b.point_precision and c.visible_no_box<=b.visible_no_box)
    decision='定位候选值得进一步回顾性验证，尚不替换默认RR流程' if location_promising else '不采用为默认定位模型；未达到同时改善召回并保持精度的条件'
    write_json(OUT/'decision.json',dict(location_screen_pass=location_promising,
        gate='correct points strictly improve, point precision no worse, visible no-box frames no greater',
        default_adopted=False,RR_validated=False,retrospective=True))
    lines=['# 最终JSON源域分组训练与久福定位对照','',
           '2026-09-23；固定第0折，1942训练帧/573验证帧，40轮last.pt。久福未参与本轮训练或权重选择，但已用于先前多轮诊断，不是新独立外测。',
           '', '## 1. 已核验鼻孔上的定位结果','',
           '36个人工复核帧，其中30帧鼻孔可见，共50个近似鼻孔椭圆。框阈值0.25、关键点阈值0.5、推理尺寸640；最大置信鼻部框，左右点无序一对一匹配。','',
           '| 指标 | 原冻结Pose | 本轮源域训练Pose |','|---|---:|---:|']
    for title,col in [('预测点数','predicted_points'),('落入人工区域的点数','correct_points'),('点召回率','point_recall'),
                      ('点精确率','point_precision'),('点F1','point_F1'),('可见但无鼻部框帧','visible_no_box'),
                      ('可见但无正确点帧','visible_no_correct'),('分配ROI的越界比例中位数','median_ROI_outside_all_assigned')]:
        lines.append(f'| {title} | {b[col]:.6g} | {c[col]:.6g} |')
    lines+=['','注意：点F1不是呼吸事件F1；ROI越出近似人工椭圆不等于已证明都是背景。不同模型匹配样本不同，不能只比较各自中位数。',
            f'共同有分配点的{paired["paired_regions"]}个人工区域：原模型越界比例中位数{paired["baseline_median"]}，候选{paired["candidate_median"]}；逐对变化中位数{paired["median_paired_change"]}。这仅是条件子集。',
            '', '## 2. 源域内部验证','', '| 指标 | 原冻结Pose | 候选 |','|---|---:|---:|']
    for title,col in [('鼻部框召回(IoU>=0.5)','box_recall_iou50'),('鼻孔PCK@0.1(全标注点分母)','pck10_all_annotated')]:
        lines.append(f'| {title} | {source.loc["baseline",col]:.6f} | {source.loc["candidate",col]:.6f} |')
    lines+=['', 'PCK按原标注左右语义匹配；关键点置信>=0.5、鼻框IoU>=0.5且点误差<=真实鼻框对角线的10%。这与久福椭圆内命中率不是同一指标。原冻结模型可能见过源域验证视频，因此该对比只辅助诊断，不能冒充公平独立训练消融。只完成预先固定一折，不能说已完成五折交叉验证。',
            '', '## 3. 可视化','', '选择规则固定为逐帧正确点改善最大、恶化最大及首个不变样本，完整案例表为anatomy_frame_results.csv，不隐藏恶化样本。']
    for e in examples:lines+=['',f'### {e["category"]}: {e["frame_id"]}',f'![{e["category"]}](figures/{e["category"]}.png)']
    lines+=['','## 4. 采用决定与未完成项','',decision+'。',
            '本轮同时改变初始化、源域训练协议和数据划分，不能把变化单独归因于浮点转换或某一增强。没有增加注意力、没有训练鼻孔分割网络、没有用人工ROI替换自动结果。',
            '原RR输出、温度映射、人工呼气时间与计数均未改。定位改善即使成立，也不能证明全部温度极值对应呼气；该问题还需连续ROI温度和事件链验证。',
            '', '![training losses](figures/training_losses.png)']
    (OUT/'源域训练与久福定位对照.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(decision)


if __name__=='__main__':main()
