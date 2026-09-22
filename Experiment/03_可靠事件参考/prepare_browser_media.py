"""Lossless H.264 viewing copies of original CFR49, with time/pixel equivalence checks."""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_common import HOLDOUT, REFERENCE, sha256, write_csv, write_json
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools_deps"))
import imageio_ffmpeg


def compare(source, target):
    a,b=cv2.VideoCapture(str(source)),cv2.VideoCapture(str(target))
    n=0; error=0.; pixel=0
    try:
        while True:
            oka,fa=a.read(); okb,fb=b.read()
            if oka!=okb:
                raise ValueError("Viewing-copy frame count changed")
            if not oka:
                break
            if fa.shape!=fb.shape:
                raise ValueError("Viewing-copy geometry changed")
            error=max(error,abs(a.get(cv2.CAP_PROP_POS_MSEC)-b.get(cv2.CAP_PROP_POS_MSEC))/1000)
            pixel=max(pixel,int(np.abs(fa.astype(np.int16)-fb.astype(np.int16)).max()))
            n+=1
    finally:
        a.release();b.release()
    if n==0 or error>.001 or pixel>1:
        raise ValueError(f"Viewing-copy equivalence failed: n={n}, timestamp_delta={error}, max_pixel_delta={pixel}")
    return n,error,pixel


def main():
    out=REFERENCE/'event_workspace/browser_media_v1'
    out.mkdir(parents=True,exist_ok=False)
    source_manifest=HOLDOUT/'adapter_validation/20260912_v2/source_manifest.csv'
    sources=pd.read_csv(source_manifest,dtype=str,keep_default_na=False)
    executable=Path(imageio_ffmpeg.get_ffmpeg_exe())
    results=[]
    for i,row in enumerate(sources.itertuples(),1):
        source=Path(row.video_path)
        if sha256(source)!=row.video_sha256:
            raise ValueError("Original viewing video changed")
        target=out/f'{row.video_id}_browser_lossless.mp4'
        command=[str(executable),'-nostdin','-v','error','-n','-i',str(source),'-map','0:v:0','-an','-c:v','libx264','-preset','fast','-qp','0','-pix_fmt','yuv420p','-threads','2','-fps_mode','passthrough','-movflags','+faststart',str(target)]
        subprocess.run(command,check=True)
        n,error,pixel=compare(source,target)
        results.append(dict(video_id=row.video_id,source_path=str(source),source_sha256=row.video_sha256,browser_video_path=str(target.resolve()),browser_sha256=sha256(target),frames=n,max_timestamp_delta_seconds=error,max_decoded_pixel_delta=pixel))
        print(f'Viewing copy {i}/49 verified',flush=True)
    write_csv(out/'viewing_copy_manifest.csv',results)
    write_json(out/'verification.json',dict(status='PASS',windows=len(results),lossless_yuv_encoding=True,all_frames_compared=True,ffmpeg_sha256=sha256(executable),source_manifest_sha256=sha256(source_manifest),annotation_counts_changed=False))


if __name__=='__main__':
    main()
