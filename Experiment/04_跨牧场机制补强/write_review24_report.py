"""Generate an evidence supplement and a versioned revision, preserving the old draft."""
from pathlib import Path
import re
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE/'20260921_r3_v1'
OLD=HERE/'20260921_v1'


def main():
    audit=pd.read_csv(ROOT/'annotation_audit.csv')
    ev=pd.read_csv(ROOT/'event_metrics.csv')
    rr=pd.read_csv(ROOT/'rr_metrics.csv')
    delta=pd.read_csv(ROOT/'reference_sensitivity.csv')
    shift=pd.read_csv(ROOT/'r2_r3_matched_times.csv')
    issues=pd.read_csv(ROOT/'clarifications.csv')
    assert len(audit)==24 and audit.eligible_full_window.sum()==23
    assert audit.confirmed_events.sum()==448
    scope='eligible_full_window'
    def em(c,v,strict=False,method='frozen_baseline'):
        s='strict_visible_phase' if strict else scope
        return ev[(ev.scope==s)&(ev.cohort==c)&(ev.version==v)&(ev.method==method)&ev.tolerance_seconds.eq(.3)].iloc[0]
    labels={'lindian49':'林甸','jiufu271':'久福'}
    table=['| 队列 / 配对参考窗 | R2事件F1 | R3事件F1 | R3−R2差值（95%区间） | R3 TP / FP / FN |',
           '|---|---:|---:|---|---|']
    for c,label in labels.items():
        a,b=em(c,'R2'),em(c,'R3')
        d=delta[(delta.scope==scope)&(delta.cohort==c)].iloc[0]
        table.append(f'| {label} / {int(b.windows)} | {a.f1:.6f} | {b.f1:.6f} | {d.r3_minus_r2_f1:+.6f} [{d.ci_low:.6f}, {d.ci_high:.6f}] | {int(b.tp)} / {int(b.fp)} / {int(b.fn)} |')
    event_table='\n'.join(table)
    rt=['| 队列 / 有输出配对窗 | R2 RR R² | R3 RR R² | R2 MAE | R3 MAE |',
        '|---|---:|---:|---:|---:|']
    for c,label in labels.items():
        a=rr[(rr.scope==scope)&(rr.cohort==c)&(rr.version=='R2')].iloc[0]
        b=rr[(rr.scope==scope)&(rr.cohort==c)&(rr.version=='R3')].iloc[0]
        rt.append(f'| {label} / {int(b.paired_windows)} | {a.rr_r2:.6f} | {b.rr_r2:.6f} | {a.rr_mae_bpm:.4f} | {b.rr_mae_bpm:.4f} |')
    rr_table='\n'.join(rt)
    visible='；'.join(f"{labels[c]}{int(em(c,'R3',True).windows)}窗F1 {em(c,'R2',True).f1:.6f}→{em(c,'R3',True).f1:.6f}" for c in labels)
    timing='；'.join(f"{label}在0.30 s内可匹配事件的R2−R3时间差中位数为{shift[(shift.cohort==c)&shift.tolerance_seconds.eq(.3)].r2_minus_r3_seconds.median():.4f} s" for c,label in labels.items())
    result=f'''### 4.9 R3暂停逐帧复核：参考敏感性不能解释全部失效

2026年9月21日19:21重新导出的R3包含24窗、448个确认事件，CSV与JSON一致。R3-04已补齐19次；R3-11虽标为complete并记录2次，但备注明确4 s后至结尾鼻子在拍摄范围外，且逐段不可观察表为空，因此仅保留其原始记录，不将2次当作完整30 s参考。完整窗口比较使用其余23窗：林甸12窗、久福11窗。它们在R2和R3下使用完全相同的冻结预测、观看时间和0.30 s主容差；没有重新推理、平移参考或调阈值。

{event_table}

表8 同一复核子集、同一预测下的R2/R3事件评价。差值及2000次配对窗口重采样区间衡量参考敏感性，不是算法增益；每窗为不同原视频（林甸）或不同牛号（久福）。久福复核有意包含拒判，样本量小，不能替换47/216窗主事件结果。

![图6 同窗冻结预测对R2与R3的容差敏感性](figures/01_reference_sensitivity.png)

图6 两条线仅改变人工参考，预测不变；竖线为预先保留的0.30 s主容差。更宽容差更容易匹配，不作为提高主成绩的方法。

林甸8/12窗、久福11/11窗的参考总数发生变化，平均绝对变化分别为1.2500次和1.5455次；因此这不是仅改变时间而保持总数不变的纯延迟试验。{timing}。这些时间差只来自容差内匹配成功的截断子集，受周期错配与选择影响，不能估计普遍反应延迟，也不能据此平移全部事件。林甸参考变化关联的F1差值区间不跨0，久福区间跨0；不能将久福的所有失效归因于播放按键标注。

{rr_table}

表9 复核子集的计数评价，MAE单位为次/min。只纳入同一批有输出窗口：林甸12、久福7。它不是原47/173窗计数结果；久福7窗R²对小样本和参考范围非常敏感。事件F1升高而计数MAE增大，说明更新事件时序不必然令总次数评价更好。

R3中R3-06与R3-22仅标明热极值；R3-12窗级写热极值、22个事件级仍写可辨认呼气，字段不一致，原值均保留。主描述性分析允许这3窗，但不把其事件当作已确认生理呼气；更严格地要求窗级与所有事件均为visible_expiration后，{visible}。该敏感性结果仍有明显差异，不支持“全部差距只来自这3窗相位依据”的解释。visible_expiration本身仍是同一观察者的选择，不是同步生理仪器验证。

久福11窗中7窗有输出、4窗拒判。R3参考的102个FN中，59个来自拒判，43个来自有输出窗；有输出7窗为TP41、FP42、FN43，条件F1为0.491018，不能代替包含拒判的0.362832。4个拒判窗仍标出了59个确认事件，其中3窗声明可辨认呼气、1窗仅能辨认热极值。这是“算法拒判不等于完全无法人工辨认”的窗口级证据，而非逐帧左右鼻孔均可见的证明。

![图7 原参考与复核事件的真实时间栅格](figures/02_event_rasters.png)

图7 分别取每个队列参考计数绝对变化最大的窗；并列时按window_id排序。图只作差异说明，不是随机代表样本。每条短线为导出或冻结事件，没有生成模拟波形。

此前已否决的来源门控使用完全相同保留/剔除事件，在R3下林甸F1为0.812183、久福为0.230769，对应冻结基线为0.810631和0.362832。林甸局部小幅升高不足以改变采用决定，久福仍退化；继续不采用，不选择复核样本重调门控。结合R3可辨认事件与算法拒判并存，后续应优先审查信号提取或质量门限为何失去可利用信息，而非简单把非直接检测一律视为无效呼吸。

逐段不可观察区间仍为0行，R3-11的自然语言备注只提供粗略范围，不能代替逐帧区间，更不能补造左右侧可见性标签。该轮已完成重复事件参考的主要分析，但尚未完成原计划中的真实可见性定量验证。

'''
    report=f'''# 24窗R3复核结果与论文修订说明

来源：C:/Users/muxi/Desktop/实验/24标注，采用2026-09-21 19:21新版。448个事件，原件已按字节归档于raw；旧R2及算法均未改。

## 核心判断

暂停逐帧参考使内部事件一致性明显变化，但外部问题仍然存在；标注方式是评价敏感因素，却不是目前能确认的唯一跨场失效原因。来源门控仍有负结果，不能包装成成功创新。

{event_table}

完整参考主分析为林甸12窗、久福11窗，后者只有7窗有算法输出，4窗拒判仍纳入FN。此表不与47/216窗整体F1混用；F1变化不是算法提升。

## 具体证据

1. R3下久福TP41、FP42、FN102；FN中59来自拒判，43在有输出窗。人工可确认事件和算法拒判并存，说明缺失的是算法测量覆盖，未必是所有可观察呼吸信息。
2. 林甸8/12、久福11/11窗总数变化。事件时间与事件数均变化，不能将F1差值解释为纯反应延迟效应。
3. {timing}。仅成功匹配子集，不做统一时间偏移。
4. 严格相位子集：{visible}。排除热极值/字段冲突后，差距仍在；不是独立生理相位验证。
5. 被否决的局部门控在R3久福F1仍降至0.230769，继续不采用。没有用R3优化预测。

{rr_table}

计数配对仅12/7窗；MAE单位次/min。事件一致性与计数误差不是同一指标。

## 保留的问题

- R3-11：按导出仍是complete，但备注4 s后鼻子在范围外。暂排除完整窗口评分，不替用户改状态或填区间；其2个事件仍保留。
- R3-12：窗级thermal_extremum_only、22事件级visible_expiration。保留原字段；仅纳入混合依据描述性集合，不纳入严格相位集合。
- 不可观察表0行，不等于全程可观察。当前不能计算来源门控对真实不可观察区间的灵敏度或特异度。
- 本轮复核为同观察者重复标注、有意抽取拒判窗，不是双盲、第二观察者或新的独立外测。

## 文件用途

| 文件 | 用途 |
|---|---|
| raw/ | 四份新版导出原件，不改字段 |
| annotation_audit.csv / clarifications.csv | 24窗完整性、相位及冲突记录 |
| count_changes.csv | 每窗R2/R3次数变化及冻结输出 |
| events_by_window.csv / event_metrics.csv | 同窗口事件评分、容差敏感性与旧候选复核 |
| rr_metrics.csv / reference_sensitivity.csv | 计数指标与小样本配对区间 |
| r2_r3_reference_matching.csv / r2_r3_matched_times.csv | 参考间匹配及截断时间差，不用于平移 |
| figures/ | 实际事件栅格和容差曲线 |
| manuscript/ | R3更新后的完整中文稿与英文摘要，旧稿保留 |
| input_hashes.csv / verification.json | 来源锁定、CSV/JSON一致性与核验结果 |

完成边界：已完成可用23窗的R3分析与稿件修订；真实可见性区间和成功的新方法贡献仍未获得，不追加未经证据支持的性能声明。
'''
    (ROOT/'复核结果与下一步.md').write_text(report,encoding='utf-8')
    text=(OLD/'中文论文初稿.md').read_text(encoding='utf-8')
    def replace(old,new):
        nonlocal text
        assert text.count(old)==1,old[:80]
        text=text.replace(old,new)
    replace('中文研究初稿 / 2026-09-21 / 供导师讨论，尚非投稿终稿','中文研究初稿 / 2026-09-21 R3更新版 / 供导师讨论，尚非投稿终稿')
    replace('24窗暂停逐帧复核尚待人工完成。','新增24窗暂停逐帧导出已完成；其中23窗用于同预测参考敏感性分析，剩余1窗有完整状态与不可观察备注冲突。区间级可见性证据仍不足。')
    replace('后续核心补强应以独立保存的暂停逐帧相位与可见性复核约束事件判定，而非仅增加网络模块或提高拒判率。',
            '新增R3同观察者暂停逐帧复核中，23个可用窗口的同预测事件F1在林甸12窗由0.692180变为0.810631，在久福11窗由0.304933变为0.362832；该参考变化未消除跨场差距，也不是算法提升。真实可见性区间仍需核查，而非仅增加网络模块或提高拒判率。')
    replace('A separate paused, framewise review is required before observation-aware event decision rules can be claimed as a validated methodological contribution.',
            'A subsequent same-observer, paused framewise review provided 23 eligible paired windows: frozen-prediction F1 changed from 0.692180 to 0.810631 on 12 internal windows and from 0.304933 to 0.362832 on 11 external windows. These changes reflect reference sensitivity, not algorithmic improvement, and did not remove the cross-farm gap. Interval-level observability and independent physiological phase remain unvalidated.')
    replace('后续补充复核是同一观察者重复标注，不写成双人一致性验证。',
            'R3补充复核是同一观察者重复标注，不写成双人一致性验证。复核成员按固定散列顺序选取林甸12个不同原视频、久福8个输出和4个拒判的不同牛号；未用预测误差或原次数抽样。新页面不预填事件与次数，要求暂停后按解码PTS记录，并保留窗级/事件级相位依据。2026年9月21日收到24窗448事件；所有时间在对应观看窗内且与解码PTS一致至毫秒导出精度，原件独立归档。纳入与排除依据见4.9节；不按新参考选取预测参数。')
    replace('## 5 讨论',result+'## 5 讨论')
    replace('但当前简单来源约束已经被负结果否定；不能将它包装成有效新方法。24窗复核将检验来源不直接的帧是否真的不可观察，以及R2时间误差是否大到影响事件评价。只有在这些证据明确后，才决定是否值得把二值来源门控改为左右侧真实支持驱动的局部判定。方法改进仍需同输入对照，且不得以牺牲大量召回来换取precision后单方面宣称更优。',
            '但当前简单来源约束已经被负结果否定；不能将它包装成有效新方法。R3复核已表明参考改变影响内部事件评价，但未消除外部差距；人工可辨认事件与算法拒判并存，提示不能把来源非直接一律视为不可测。由于逐段可见性仍为空，尚不能定量确认来源标志对真实可见性的对应关系。方法改进仍需同输入对照，不得仅靠牺牲召回来提高precision。当前可报告的是事件参考敏感性和错误来源分解，而不是成功的新门控算法。')
    replace('投稿前必须完成当前24窗人工复核，在已完成的缓存层统一对照基础上核查上游检测与真实可见性，再评估一项有证据支持的事件方法。',
            '本轮24窗复核已经导出并完成23窗主要分析；投稿前仍需澄清R3-11完整状态及R3-12相位字段，补足必要的可观察区间。在已完成的缓存层统一对照基础上核查上游检测与真实可见性，再评估一项有证据支持的事件方法。')
    replace('人工事件时间敏感性与两队列处理差异限制了因果归因。',
            'R3重复标注在同预测下改变了事件评价，却未消除跨场差距；其参考计数也变化，不能把全部差异归因于按键反应延迟。两队列处理差异与区间级可见性缺失仍限制因果归因。')
    replace('待补A1：24窗R3暂停逐帧事件、相位依据及不可观察区间，原R2保留。R3不是替换全部320窗参考的授权。',
            '待补A1：24窗R3事件导出已完成（448事件），23窗完成主要分析；需澄清R3-11状态/区间、R3-12窗级与事件级相位冲突。不可观察区间仍未填写，不能称可见性验证完成。原R2保留，R3不是替换全部320窗参考的授权。')
    # Relocate only image links; retain the archived figures byte-for-byte.
    text=re.sub(r'(!\[[^\]]*\]\()(analysis_v3/[^)]+)(\))',r'\1../../20260921_v1/\2\3',text)
    text=text.replace('](figures/','](../figures/')
    dest=ROOT/'manuscript';dest.mkdir(exist_ok=False)
    (dest/'中文论文初稿.md').write_text(text,encoding='utf-8')
    claims=pd.DataFrame([
        ['参考敏感性','同预测下林甸12窗F1变化+0.118452','reference_sensitivity.csv','不是算法改善或普遍反应延迟'],
        ['外部问题仍在','久福11窗R3 F1 0.362832；拒判FN59、输出FN43','events_by_window.csv','不是216窗全量结果'],
        ['门控继续否决','同R3久福局部门控F1 0.230769','event_metrics.csv','不因林甸小子集略升而采用'],
        ['可见性证据不足','0区间、R3-11备注、R3-12字段冲突','clarifications.csv','不能构造解剖可见性金标准'],
    ],columns=['claim','evidence','file','limit'])
    claims.to_csv(ROOT/'R3主张与证据.csv',index=False,encoding='utf-8-sig')
    print('Created report and versioned manuscript:',dest)


if __name__=='__main__':main()
