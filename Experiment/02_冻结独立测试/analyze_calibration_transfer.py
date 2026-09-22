"""Describe calibration transfer bias; never apply a fitted correction or train a model."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment_common import *


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    root = REPO/"temperature_extraction/getRandomForestRegress"
    model_path = HOLDOUT/"method_snapshots/20260909_v1/weights/clf_model_RGB_20240906.pkl"
    model = joblib.load(model_path)
    rows, color_tables = [], []
    fig, axes = plt.subplots(1,2,figsize=(11,4.8),layout="constrained")
    for ax,(image_name,table_name,label) in zip(axes,[("1.png","1.csv","Jiufu 2024 calibration still"),("20230810T152315.JPG","20230810T152315.csv","Lindian 2023 calibration still")]):
        image = cv2.imread(str(root/image_name))
        truth = pd.read_csv(root/table_name,header=None).to_numpy(float)
        if truth.shape != image.shape[:2]: raise ValueError("Image/matrix shape mismatch")
        colors, inverse, counts = np.unique(image.reshape(-1,3),axis=0,return_inverse=True,return_counts=True)
        y = truth.ravel()
        p = model.predict(colors)[inverse]
        bias = float(np.mean(p-y))
        centered = p-y-bias
        coefficient = np.polyfit(y,p,1)
        correlation2 = float(np.corrcoef(y,p)[0,1]**2)
        rows.append({"image":image_name,"temperature_r2":float(1-np.sum((p-y)**2)/np.sum((y-y.mean())**2)),
                     "mae_celsius":float(np.mean(abs(p-y))),"bias_pred_minus_true_celsius":bias,
                     "rmse_after_removing_mean_bias_diagnostic_only":float(np.sqrt(np.mean(centered**2))),
                     "pearson_squared_not_r2":correlation2,"diagnostic_slope":float(coefficient[0]),"diagnostic_intercept":float(coefficient[1]),
                     "correction_applied":False,"model_retrained":False,"rf_sha256":sha256(model_path)})
        averages = np.bincount(inverse,weights=y)/counts
        codes = colors[:,0].astype(np.int64)*65536+colors[:,1].astype(np.int64)*256+colors[:,2]
        color_tables.append(pd.DataFrame({"bgr_code":codes,"mean_true_temperature":averages,"pixels":counts}))
        sample = np.arange(0,len(y),64)
        ax.scatter(y[sample],p[sample],s=3,alpha=.25,color="#087e8b")
        limits=[min(y.min(),p.min())-1,max(y.max(),p.max())+1]
        ax.plot(limits,limits,color="#c2533b",linestyle="--",label="Identity")
        ax.set(xlabel="Exported pixel temperature (C)",ylabel="Frozen RF prediction (C)",title=label,xlim=limits,ylim=limits)
        ax.legend();ax.grid(alpha=.2)
    shared = color_tables[0].merge(color_tables[1],on="bgr_code",suffixes=("_2024","_2023"),validate="one_to_one")
    shared["target_2023_minus_2024_celsius"] = shared.mean_true_temperature_2023-shared.mean_true_temperature_2024
    write_csv(args.out/"calibration_transfer_diagnostics.csv",rows)
    write_csv(args.out/"identical_bgr_target_comparison.csv",shared)
    write_json(args.out/"shared_color_summary.json",{"exact_common_bgr_colors":len(shared),
               "median_target_shift_2023_minus_2024_celsius":float(shared.target_2023_minus_2024_celsius.median()),
               "diagnostic_only":True,"no_video_rr_labels_used":True,"no_mapping_modified":True})
    fig.savefig(args.out/"paired_still_temperature_transfer.png",dpi=180)
    plt.close(fig)
    print(pd.DataFrame(rows).drop(columns="rf_sha256").to_string(index=False))
    print('Shared colors:',len(shared),'median target shift:',shared.target_2023_minus_2024_celsius.median())


if __name__ == "__main__":
    main()
