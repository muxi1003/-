"""Evidence report; completion of scientific objective remains unproven."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_transfer import HERE,HOLDOUT,read,write_json,sha256

ROOT=HERE/'20260922_pose_phase_v1'


def main():
    figdir=ROOT/'figures'
    audit=read(HERE/'20260921_r3_v1/annotation_audit.csv')
    refs=read(HERE/'20260921_r3_v1/raw/reference_events.csv')
    peaks=read(ROOT/'phase_audit/extremum_candidates.csv')
    for rid in ['R3-02','R3-10']:
        row=audit[audit.review_id.eq(rid)].iloc[0];wid=row.window_id
        folder=HOLDOUT/'external_transfer/20260914_v2/windows'/wid
        temps=read(folder/'temperatures.csv');times=read(folder/'map.csv').target_time_seconds.to_numpy(float)
        reference=refs[refs.window_id.eq(wid)&refs.confidence.eq('confirmed')].event_time_seconds.to_numpy(float)
        fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[2,2,1]})
        for ax,side,color in zip(axes[:2],['left','right'],['#087f8c','#b95823']):
            ax.plot(times,temps[side+'_temp'],color=color,lw=1.3,label='original finite RF ROI samples')
            for kind,marker,c in [('maximum','^','#2f6a47'),('minimum','v','#9d3a65')]:
                subset=peaks[peaks.window_id.eq(wid)&peaks.side.eq(side)&peaks.kind.eq(kind)]
                tt=subset.event_time_seconds.to_numpy(float)
                indices=np.argmin(abs(times[:,None]-tt[None,:]),axis=0) if len(tt) else []
                ax.scatter(tt,temps[side+'_temp'].iloc[indices],marker=marker,color=c,s=35,label=kind)
            for t in reference:ax.axvline(t,color='#79868a',lw=.6,alpha=.55)
            ax.set(ylabel=f'{side} RF proxy / degC');ax.legend(loc='upper right',fontsize=8)
        axes[0].set_title(f'{rid}: extrema from MAF3 within finite runs; missing samples NOT interpolated\nGray vertical lines: existing R3 reference, no time shift; proximity is not physiological proof',fontsize=11)
        axes[2].eventplot([reference],lineoffsets=[0],linelengths=.7,colors=['#2f4554'])
        axes[2].set(yticks=[0],yticklabels=['R3 expiration'],xlabel='Original viewing time (s)',xlim=(0,30))
        fig.savefig(figdir/f'{rid}_finite_extrema_vs_R3.png',dpi=160);plt.close(fig)
    report='''# 鼻孔定位、ROI与呼气相位：本轮机制核验

2026-09-22。目标仍在进行中，不能以完成诊断工具代替完成解剖定位和生理对应验证。

## 一、做了什么，哪些证据仍缺

沿用已有12个久福R3复核窗，各固定0/5/10/15/20/25秒附近原图，共72帧。固定七种输入，504次YOLO11n-Pose推理；坐标反变换回原图，RF不使用变换后的颜色。另用既有R3参考，核验温度极大/极小值的对应，不移动参考时间、不跨缺失插值。原模型、R2/R3标注、冻结预测和完整RR保持不变。

| 目标 | 本轮证据 | 是否解决 |
|---|---|---|
| 鼻孔可见时漏检 | 固定抽样与7种变换，候选坐标全部保存；已有呼气标注不包含鼻孔位置 | 尚未解决；需要区分鼻孔真的可辨认、模型没检测、检测到错误位置 |
| ROI是否偏离鼻孔 | 原图坐标反变换测试通过；评分脚本可比较原保存ROI与人工椭圆 | 尚无实际人工位置参考，不能报告偏离率 |
| 哪些极值对应呼气 | 已输出逐R3事件的左右极大/极小时间差及原始有限温度支持 | 完成时间对应诊断，但单观察者R3及热像不足以独立证明生理相位 |

## 二、简单的几何/对比度修复未成立

| 单变量输入 | 有检测的帧 / 72 | 高置信关键点数（不等于正确点） |
|---|---:|---:|
| 原图基线 | 34 | 51 |
| 中央75%裁剪 | 32 | 47 |
| LAB亮度CLAHE | 33 | 48 |
| 灰度三通道 | 21 | 39 |
| 下方75%裁剪 | 30 | 42 |
| 旋转-30° | 11 | 13 |
| 旋转+30° | 6 | 8 |

冻结训练配置degrees=0支持检查旋转敏感性，但不能据此证明训练姿态覆盖不足就是主要原因。当前所有单变量变换均未提高总体检出覆盖；没有采用任一候选，没有根据人工次数选择最佳变换。少数帧获得新检测也不等于恢复鼻孔，必须核验具体位置。R3-01在-30°变换下仅5秒帧出现2个超过0.5的关键点，其余固定抽样多数仍失败。

![原图与反变换预测坐标](figures/R3-01_t05_pose_comparison.png)

上图圈为20像素诊断圆，不是人工区域，也不表示采用新坐标。新增坐标需要人工解剖确认后才能判断正确与否。

## 三、极大值比极小值更接近参考，但并非每个极大值都是真呼气

计算规则固定为有限连续段内MAF3、distance=5、prominence=0.035；左右分别归一化，绝不跨空值连接。极值时间为三点均值的中心时刻，不引入平移。候选分别按极大/极小和左右侧做0.3秒容差的一对一匹配。这里只诊断，不作为新的完整窗口RR预测。

11个可用久福R3窗共143个确认事件；严格可见呼气相位集合为9窗117事件，排除热极值作为相位依据的2窗。R3-11完整状态与不可见备注冲突，继续排除完整评分。

| 每个参考事件附近的情况 | 11窗143事件 | 严格9窗117事件 |
|---|---:|---:|
| 附近没有有效ROI温度样本 | 38 | 34 |
| 有温度，但0.3秒内没有候选极值 | 37 | 28 |
| 仅有极大值接近 | 46 | 36 |
| 仅有极小值接近 | 5 | 5 |
| 极大与极小均接近，不能单凭邻近判定 | 17 | 14 |

这些类别是逐事件邻近描述，左右联合可能含歧义，不能相加冒充一对一TP。无ROI样本表示算法信号缺失，不表示人一定看不到鼻孔。原始观看时间与温度采样格点匹配均保留。

11窗一对一对照：左侧极大值F1=0.401709、极小值0.127273；右侧极大值0.424779、极小值0.126697。极大值较接近参考不等于已验证每个峰为呼气，也不能用左右最佳一侧的事后选择作为主结果。温标、解剖ROI与R3重复参考的局限保留。

![R3-02原始温度与极值对照](figures/R3-02_finite_extrema_vs_R3.png)

![R3-10缺失导致无法追踪完整呼吸](figures/R3-10_finite_extrema_vs_R3.png)

## 四、下一步已经准备好，但不代填人工参考

打开index.html，默认优先36帧，全部72帧。页面完全不加载算法预测坐标，拖动标记A/B鼻孔椭圆，选择两侧/单侧可辨认、不可辨认或不确定。无需重新数呼吸。完整操作见README.md。导出JSON后才能运行evaluate_nostril_reference.py，空白文件会拒绝评分。

评分将分别报告：可见帧无预测比例、可见鼻孔无正确坐标比例、无序一对一关键点到人工椭圆的归一化距离、圆ROI位于人工椭圆外的像素比例。算法探针用20像素诊断半径；frozen_roi用实际保存的自适应半径和温度可用坐标，两者分开报告。人工椭圆是近似解剖区域，不是像素分割金标准；标注未完成、不可辨认和不确定不填0、不强行匹配。

## 五、采用边界与验证

本轮不采用旋转、裁剪、灰度或CLAHE作为新默认算法。72固定抽样也不是271窗全部帧，检出数不能当检测AP或全量RR。后续需要先根据实际解剖参考定位错误类型，再决定补训练、改变ROI或建立有界可见性规则。已查看的久福数据只能用于回顾性研究，不能再次宣称未接触外测。

15项新增相关单元测试覆盖反变换、缺口不跨越、MAF时间对齐、极值极性、无序匹配、ROI区域面积、状态冲突、待完成和合成数据保护。独立无头Edge浏览器测试覆盖IME物理键、编辑区保护、坐标换算、撤销、导出和导入，桌面/手机截图已检查。QA目录中合成JSON明确标记synthetic_test_only，评分器拒绝将其用于研究。

剩余阻点是实际鼻孔位置/可见性参考及生理对应证据，不是代码运行失败。本轮有新实验及工具交付，目标保持active；收到真实标注后按已固定规则评分，不用预测值代填标注。
'''
    (ROOT/'机制核验报告.md').write_text(report,encoding='utf-8')
    write_json(ROOT/'progress_verification.json',dict(status='PROGRESS_OBJECTIVE_NOT_COMPLETE',
        inferences=504,fixed_frames=72,diagnostic_variants=7,strict_R3_windows=9,strict_R3_events=117,
        actual_anatomical_labels_received=False,default_changed=False,
        scientific_gaps=['anatomical_location_visibility_reference','independent_physiological_phase_validation'],
        report_sha256=sha256(ROOT/'机制核验报告.md'),
        reviewed_outputs=['probe_verification.json','phase_audit/event_classification_summary.csv','qa/ui_verification.json'],
        human_reference_entry='index.html'))
    print('Report written; objective NOT complete')


if __name__=='__main__':main()
