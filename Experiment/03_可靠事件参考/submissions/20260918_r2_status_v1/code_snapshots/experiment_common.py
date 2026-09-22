"""Local, versioned I/O and measurement statistics for the three evidence blocks."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DATA = REPO / "Dataset_new" / "72video"
ASSETS = DATA / "al_images" / "paper_repro_quality_residual_paper_assets"
ABLATION = ROOT / "01_同口径消融"
HOLDOUT = ROOT / "02_冻结独立测试"
REFERENCE = ROOT / "03_可靠事件参考"
PYTHON = Path("E:/real/anaconda/envs/plant_gpu/python.exe")
RAW_ROOTS = {
    "jiufu": REPO.parent / "久福牧场面部热红外数据及牛脸RGB录像数据_344",
    "lindian": REPO.parent / "林甸红外视频",
}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mts"}


def stamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2 ** 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def write_csv(path: Path, data, columns=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data, columns=columns)
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        table.to_csv(handle, index=False)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)


def is_true(value) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def measurements(frame: pd.DataFrame) -> dict:
    y = frame["truth_rr_bpm"].to_numpy(float)
    p = frame["predicted_rr_bpm"].to_numpy(float)
    counts = frame["truth_count"].to_numpy(float)
    estimated = frame["predicted_count"].to_numpy(float)
    valid = np.isfinite(y) & np.isfinite(p) & np.isfinite(counts) & np.isfinite(estimated)
    if not valid.all():
        raise ValueError("Nonfinite predictions/references cannot be silently dropped")
    error = p - y
    ce = np.abs(estimated - counts)
    sst = float(np.sum((y - y.mean()) ** 2))
    sd = float(error.std(ddof=1)) if len(error) > 1 else np.nan
    return {
        "n": len(y), "rr_r2": 1 - float(np.sum(error ** 2)) / sst if sst > 0 else np.nan,
        "rr_mae_bpm": float(np.abs(error).mean()),
        "rr_rmse_bpm": float(np.sqrt(np.mean(error ** 2))),
        "count_mae": float(ce.mean()), "exact_count": int((ce == 0).sum()),
        "within_one_count": int((ce <= 1).sum()),
        "mean_count_accuracy_percent": float(np.maximum(0, 1 - ce[counts > 0] / counts[counts > 0]).mean() * 100) if np.any(counts > 0) else np.nan,
        "bias_bpm": float(error.mean()), "loa_low_bpm": float(error.mean() - 1.96 * sd),
        "loa_high_bpm": float(error.mean() + 1.96 * sd),
    }


def paired_cluster_bootstrap(frame: pd.DataFrame, draws=2000, seed=20260909) -> dict:
    """Resample whole source groups; preserve all paired windows in a group."""
    groups = frame.groupby("cluster_id", sort=True)
    arrays = [g["paired_abs_error_delta"].to_numpy(float) for _, g in groups]
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        chosen = rng.integers(0, len(arrays), size=len(arrays))
        samples.append(np.concatenate([arrays[k] for k in chosen]).mean())
    return {"delta_mae_bpm": float(frame["paired_abs_error_delta"].mean()),
            "delta_mae_ci95_low": float(np.quantile(samples, .025)),
            "delta_mae_ci95_high": float(np.quantile(samples, .975)),
            "bootstrap_clusters": len(arrays), "bootstrap_draws": draws, "bootstrap_seed": seed}
