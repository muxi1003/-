"""Create a fresh, prediction-hidden R3 review from the existing annotation UI."""
from pathlib import Path
import hashlib
import json
import sys
import re
import cv2
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from experiment_common import REFERENCE, HOLDOUT, write_csv, write_json, write_text, sha256


def main():
    out=HERE/'20260921_v1/review24'
    out.mkdir(parents=True,exist_ok=False)
    source=REFERENCE/'submissions/20260918_r2_status_v1/raw/annotation_windows.csv'
    original=pd.read_csv(source,dtype=str,keep_default_na=False,encoding='utf-8-sig')
    pred=pd.read_csv(HOLDOUT/'event_evaluation/20260918_r2_status_v1/prediction_windows.csv',dtype=str,keep_default_na=False)
    statuses=pred.set_index('window_id').prediction_status.to_dict()
    def rank(w):return hashlib.sha256(('20260921:'+w).encode()).hexdigest()
    candidates=original[original.annotation_status.eq('complete')].copy()
    candidates['selection_hash']=candidates.window_id.map(rank)
    selected=[];groups=set()
    for cohort,status,n in [('lindian49','ok',12),('jiufu271','ok',8),('jiufu271','abstain',4)]:
        pool=candidates[candidates.cohort.eq(cohort)].sort_values('selection_hash')
        picked=[]
        for _,row in pool.iterrows():
            if cohort=='jiufu271' and statuses[row.window_id]!=status:continue
            group=row.source_path if cohort=='lindian49' else row.source_path.replace('\\','/').split('/')[-1].rsplit('-',1)[-1].rsplit('.',1)[0]
            if (cohort,group) in groups:continue
            groups.add((cohort,group));record=row.to_dict();record['sampling_stratum']=cohort+'_'+status;record['sampling_group']=group
            picked.append(record)
            if len(picked)==n:break
        assert len(picked)==n, 'Insufficient distinct groups; do not silently relax sampling'
        selected+=picked
    selected=sorted(selected,key=lambda x:rank('display:'+x['window_id']))
    mapping=[];initial=[];times={};video_checks=[]
    cols=list(original.columns)+['phase_basis','review_procedure']
    for i,row in enumerate(selected,1):
        rid=f'R3-{i:02d}'
        mapping.append(dict(review_id=rid,window_id=row['window_id'],cohort=row['cohort'],
                            sampling_stratum=row['sampling_stratum'],sampling_group=row['sampling_group'],
                            selection_hash=row['selection_hash']))
        w={k:row[k] for k in original.columns}
        w.update(video_id=rid,annotation_round='R3',annotator='',annotation_status='pending',
                 manual_breath_count='',predictions_hidden='',reference_notes='',
                 phase_basis='',review_procedure='paused_frame_review',cohort='review24')
        initial.append(w)
        cap=cv2.VideoCapture(w['browser_video_path']);tt=[]
        start=float(w['view_start_seconds']);end=start+float(w['duration_seconds'])
        try:
            assert cap.isOpened(),w['browser_video_path']
            while True:
                ok,frame=cap.read()
                if not ok:break
                t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                if t>=end:break
                if t>=start:tt.append(t-start)
        finally:cap.release()
        assert len(tt)>1 and np.all(np.diff(tt)>0), 'Invalid frame PTS'
        times[w['window_id']]=tt
        video_checks.append(dict(review_id=rid,window_id=w['window_id'],frames=len(tt),
            first_pts=tt[0],last_pts=tt[-1],duration_seconds=float(w['duration_seconds']),
            source_sha256=sha256(Path(w['browser_video_path'])),
            pts_source='OpenCV FFMPEG decoded presentation timestamps; not independent ffprobe'))
        print(f'{rid}: {len(tt)} decoded frames',flush=True)
    template=REFERENCE/'event_workspace/20260914_v3/index.html'
    html=template.read_text(encoding='utf-8')
    html,n=re.subn(r'const initial=.*?;\n',lambda _: 'const initial='+json.dumps(initial,ensure_ascii=False)+';\n',html,count=1)
    assert n==1
    html=html.replace('R2','R3').replace('20260915.2','20260921.1')
    html=html.replace('cow-rr-event-R3-20260914-v1','cow-rr-review24-R3-20260921-v1')
    html=html.replace('<option value="lindian49">林甸49窗</option><option value="jiufu271">久福271窗</option>','')
    html=html.replace("$('path').textContent=x.browser_video_path+' | 播放起点 '+x.view_start_seconds+' s | 时长 '+x.duration_seconds+' s';", "$('path').textContent=x.video_id+' | '+Number(x.duration_seconds).toFixed(3)+' s';")
    html=html.replace("$('history').textContent=x.cohort==='lindian49'?'林甸：存在历史算法接触，不能宣称原始盲法。':'久福：本轮接触情况请如实登记。';", "$('history').textContent='R3 · 独立保存';$('phase').value=x.phase_basis||'';")
    html=html.replace('function time(){return', 'function legacyTime(){return')
    html=html.replace("function ready(){", "function ready(){if(!$('video').paused||$('video').seeking){msg('先暂停，并等待画面定位完成。');return false}if(!$('phase').value){msg('请先选择相位依据。');return false}")
    html=html.replace("notes:''});invalidateCompletion()", "notes:'',phase_basis:w().phase_basis,marking_mode:'paused_frame_review'});invalidateCompletion()")
    html=html.replace("'confidence','annotator','notes']);", "'confidence','annotator','notes','phase_basis','marking_mode']);")
    html=html.replace('<div id="clock">', '<div class="row"><button id="previous-frame" title="上一帧">&#9664;|</button><button id="next-frame" title="下一帧">|&#9654;</button><select id="speed" aria-label="播放速度"><option value="0.25">0.25x</option><option value="0.5">0.5x</option><option value="1" selected>1x</option></select></div><div id="clock">')
    html=html.replace('<label>本轮备注', '<label>相位依据<select id="phase"><option value="">待确认</option><option value="visible_expiration">可辨认呼气过程</option><option value="thermal_extremum_only">仅能辨认热变化极值</option><option value="uncertain_phase">相位不能确认</option></select></label><label>本轮备注')
    html=html.replace("const physical={Space:", "const physical={ArrowLeft:'prevframe',ArrowRight:'nextframe',Space:")
    html=html.replace("[' ','q','w','e','a','d'].includes(key)", "[' ','q','w','e','a','d','prevframe','nextframe'].includes(key)")
    html=html.replace("if(key===' '){", "if(key==='prevframe'||key==='nextframe'){frameStep(key==='prevframe'?-1:1);}else if(key===' '){")
    extension=r'''
const framePTS=__PTS__;
function frameIndex(){
 const a=framePTS[current],t=legacyTime();let lo=0,hi=a.length;
 while(lo<hi){const m=(lo+hi)>>1;if(a[m]<=t+1e-6)lo=m+1;else hi=m;}
 return Math.max(0,Math.min(a.length-1,lo-1));
}
function time(){const a=framePTS[current];return a[frameIndex()];}
function frameStep(direction){
 const v=$('video');if(!v.readyState||v.seeking)return;
 v.pause();const a=framePTS[current],i=Math.max(0,Math.min(a.length-1,frameIndex()+direction));
 const end=i+1<a.length?a[i+1]:Number(w().duration_seconds);
 v.currentTime=Number(w().view_start_seconds)+a[i]+Math.max(0,(end-a[i])/2);
}
$('previous-frame').onclick=()=>frameStep(-1);
$('next-frame').onclick=()=>frameStep(1);
$('speed').onchange=()=>{$('video').playbackRate=Number($('speed').value)};
$('phase').onchange=()=>{w().phase_basis=$('phase').value;invalidateCompletion();save()};
$('video').addEventListener('seeked',()=>{$('clock').textContent=time().toFixed(3)+' s | frame '+frameIndex()});
// Keep the end guard on continuous video time, not the last quantized frame time.
$('video').addEventListener('timeupdate',()=>{if(legacyTime()>=Number(w().duration_seconds))$('video').pause()});
'''.replace('__PTS__',json.dumps(times))
    html=html.replace('list();load();\n</script>',extension+'\nlist();load();\n</script>')
    assert 'const framePTS=' in html and "phase_basis:w().phase_basis" in html
    write_text(out/'index.html',html)
    write_csv(out/'annotation_windows.csv',pd.DataFrame(initial)[cols])
    write_csv(out/'selection_key_DO_NOT_VIEW_DURING_REVIEW.csv',mapping)
    write_json(out/'decoded_frame_pts.json',times)
    write_csv(out/'viewing_time_verification.csv',video_checks)
    write_json(out/'selection_protocol.json',dict(seed='SHA256(20260921:window_id)',
        counts={'lindian_complete_distinct_sources':12,'jiufu_complete_output_distinct_cows':8,'jiufu_complete_abstain_distinct_cows':4},
        prior_counts_or_errors_used_in_selection=False,previous_events_loaded=False,
        source_windows_sha256=sha256(source),template_sha256=sha256(template),
        purpose='phase and visibility mechanism review; deliberately enrich abstentions; not prevalence sample',
        repeat_observer_not_second_independent_observer=True,original_references_unchanged=True))
    print('Review24 prepared without existing events or counts.',flush=True)


if __name__=='__main__':main()
