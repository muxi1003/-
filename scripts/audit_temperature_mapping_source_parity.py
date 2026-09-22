from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    assets = repo / "Dataset_new" / "72video" / "al_images" / "paper_repro_quality_residual_paper_assets"
    parser = argparse.ArgumentParser(
        description=(
            "Audit source-parity of the random-forest pseudo-color temperature mapping. "
            "It verifies the calibration image in both BGR and channel-reversed order, then "
            "replays stored nostril ROI temperatures without changing any RR artifact."
        )
    )
    parser.add_argument(
        "--temperature-model",
        type=Path,
        default=repo
        / "temperature_extraction"
        / "getRandomForestRegress"
        / "clf_model_RGB_20240906.pkl",
    )
    parser.add_argument(
        "--calibration-image",
        type=Path,
        default=repo / "temperature_extraction" / "getRandomForestRegress" / "1.png",
    )
    parser.add_argument(
        "--calibration-temperature-csv",
        type=Path,
        default=repo / "temperature_extraction" / "getRandomForestRegress" / "1.csv",
    )
    parser.add_argument("--internal-root", type=Path, default=repo / "Dataset_new" / "72video" / "al_images")
    parser.add_argument(
        "--external-root",
        type=Path,
        default=repo / "Dataset_new" / "72video" / "external_al_images_single_reference",
    )
    parser.add_argument("--internal-prefix", default="paper_repro")
    parser.add_argument("--external-prefix", default="external_repro_single_reference")
    parser.add_argument("--output-dir", type=Path, default=assets)
    parser.add_argument("--radius", type=int, default=20)
    parser.add_argument("--min-temp", type=float, default=20.0)
    parser.add_argument("--videos-per-cohort", type=int, default=8)
    parser.add_argument("--frames-per-video", type=int, default=8)
    return parser.parse_args()


def r2_score(truth: np.ndarray, predicted: np.ndarray) -> float:
    valid = np.isfinite(truth) & np.isfinite(predicted)
    if valid.sum() < 2:
        return math.nan
    actual = truth[valid]
    estimate = predicted[valid]
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    return float(1.0 - np.sum((actual - estimate) ** 2) / denominator) if denominator > 0 else math.nan


def metric_values(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(truth) & np.isfinite(predicted)
    if valid.sum() == 0:
        return {"samples": 0, "r2": math.nan, "mae_c": math.nan, "rmse_c": math.nan, "max_abs_error_c": math.nan}
    error = predicted[valid] - truth[valid]
    return {
        "samples": int(valid.sum()),
        "r2": r2_score(truth, predicted),
        "mae_c": float(np.mean(np.abs(error))),
        "rmse_c": float(np.sqrt(np.mean(error**2))),
        "max_abs_error_c": float(np.max(np.abs(error))),
    }


def circle_temperature(
    image_bgr: np.ndarray,
    model: object,
    x: float,
    y: float,
    radius: int,
    min_temp: float,
) -> float:
    height, width = image_bgr.shape[:2]
    center_x, center_y = int(round(x)), int(round(y))
    if not (0 <= center_x < width and 0 <= center_y < height):
        return math.nan
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), int(radius), 255, -1)
    pixels = image_bgr[mask == 255]
    if not len(pixels):
        return math.nan
    prediction = np.asarray(model.predict(pixels.reshape(-1, 3)), dtype=float)
    prediction = prediction[prediction > float(min_temp)]
    return float(np.mean(prediction)) if len(prediction) else math.nan


def calibration_audit(image_path: Path, temperature_csv: Path, model: object) -> pd.DataFrame:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read calibration image: {image_path}")
    truth = pd.read_csv(temperature_csv, header=None).to_numpy(dtype=float).reshape(-1)
    pixels_bgr = image.reshape(-1, 3)
    if len(truth) != len(pixels_bgr):
        raise ValueError(
            f"Calibration image pixels ({len(pixels_bgr)}) do not match temperatures ({len(truth)})"
        )
    rows = []
    for input_order, pixels in [
        ("opencv_bgr_source_parity", pixels_bgr),
        ("channel_reversed_rgb_negative_control", pixels_bgr[:, ::-1]),
    ]:
        predicted = np.asarray(model.predict(pixels), dtype=float)
        rows.append({"audit": "calibration_image", "input_order": input_order, **metric_values(truth, predicted)})
    return pd.DataFrame(rows)


def sample_indices(length: int, maximum: int) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=int)
    return np.unique(np.linspace(0, length - 1, min(length, maximum)).round().astype(int))


def replay_cohort(
    root: Path,
    prefix: str,
    cohort: str,
    model: object,
    radius: int,
    min_temp: float,
    videos_per_cohort: int,
    frames_per_video: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    temperature_paths = sorted(root.glob(f"*/{prefix}_temperatures.csv"))[: int(videos_per_cohort)]
    for temperature_path in temperature_paths:
        video_dir = temperature_path.parent
        data = pd.read_csv(temperature_path)
        image_paths = {path.name: path for path in video_dir.glob("*.jpg")}
        for row_index in sample_indices(len(data), int(frames_per_video)):
            row = data.iloc[int(row_index)]
            image_path = image_paths.get(str(row.get("frame_name", "")))
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR) if image_path else None
            if image is None:
                continue
            for side in ["left", "right"]:
                source_temperature = pd.to_numeric(pd.Series([row.get(f"{side}_temp")]), errors="coerce").iloc[0]
                x = pd.to_numeric(pd.Series([row.get(f"{side}_x")]), errors="coerce").iloc[0]
                y = pd.to_numeric(pd.Series([row.get(f"{side}_y")]), errors="coerce").iloc[0]
                if not (np.isfinite(source_temperature) and np.isfinite(x) and np.isfinite(y)):
                    continue
                replayed = circle_temperature(image, model, float(x), float(y), radius, min_temp)
                rows.append(
                    {
                        "cohort": cohort,
                        "video_id": video_dir.name,
                        "frame_name": str(row.get("frame_name", "")),
                        "side": side,
                        "stored_temperature_c": float(source_temperature),
                        "replayed_temperature_c": replayed,
                        "abs_error_c": abs(float(replayed) - float(source_temperature)) if np.isfinite(replayed) else math.nan,
                        "temperature_source": str(row.get(f"{side}_source", "")),
                    }
                )
    return pd.DataFrame(rows)


def replay_summary(replay: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort, group in replay.groupby("cohort", dropna=False):
        values = metric_values(
            group["stored_temperature_c"].to_numpy(float),
            group["replayed_temperature_c"].to_numpy(float),
        )
        rows.append({"audit": "stored_roi_replay", "cohort": str(cohort), **values})
    return pd.DataFrame(rows)


def write_report(path: Path, calibration: pd.DataFrame, replay: pd.DataFrame, args: argparse.Namespace) -> None:
    direct = calibration[calibration["input_order"].eq("opencv_bgr_source_parity")].iloc[0]
    reversed_order = calibration[calibration["input_order"].eq("channel_reversed_rgb_negative_control")].iloc[0]
    lines = [
        "# Temperature Mapping Source-Parity Audit",
        "",
        "The original random-forest training script reads pseudo-color images with OpenCV and "
        "passes the resulting BGR triplets directly to the model despite the historical `RGB` "
        "file name. This audit verifies that contract before replaying stored nostril ROIs.",
        "",
        "## Calibration Image",
        "",
        f"- Direct OpenCV BGR: R2=`{direct['r2']:.6f}`, MAE=`{direct['mae_c']:.6f}` C, RMSE=`{direct['rmse_c']:.6f}` C.",
        f"- Channel-reversed negative control: R2=`{reversed_order['r2']:.6f}`, MAE=`{reversed_order['mae_c']:.6f}` C, RMSE=`{reversed_order['rmse_c']:.6f}` C.",
        "",
        "## Stored ROI Replay",
        "",
    ]
    for row in replay.itertuples(index=False):
        lines.append(
            f"- {row.cohort}: samples=`{int(row.samples)}`, R2=`{row.r2:.6f}`, "
            f"MAE=`{row.mae_c:.8f}` C, max absolute error=`{row.max_abs_error_c:.8f}` C."
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "The current pipeline is source-parity only when it passes OpenCV BGR pixels to the "
            "same pkl and uses the recorded circular ROI/minimum-temperature rule. This audit "
            "does not prove that the pseudo-color calibration transfers across cameras or farms; "
            "that remains a separate external generalization question.",
            "",
            f"Replay settings: radius=`{args.radius}` px, min_temp=`{args.min_temp:.3f}` C, "
            f"videos per cohort=`{args.videos_per_cohort}`, frames per video=`{args.frames_per_video}`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    model = joblib.load(args.temperature_model)
    calibration = calibration_audit(args.calibration_image, args.calibration_temperature_csv, model)
    internal = replay_cohort(
        args.internal_root,
        args.internal_prefix,
        "internal_development_sample",
        model,
        args.radius,
        args.min_temp,
        args.videos_per_cohort,
        args.frames_per_video,
    )
    external = replay_cohort(
        args.external_root,
        args.external_prefix,
        "external_provisional_sample",
        model,
        args.radius,
        args.min_temp,
        args.videos_per_cohort,
        args.frames_per_video,
    )
    replay = pd.concat([internal, external], ignore_index=True)
    summary = replay_summary(replay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = args.output_dir / "paper_temperature_mapping_source_parity_calibration.csv"
    replay_path = args.output_dir / "paper_temperature_mapping_source_parity_roi_replay.csv"
    summary_path = args.output_dir / "paper_temperature_mapping_source_parity_summary.csv"
    report_path = args.output_dir / "paper_temperature_mapping_source_parity_report.md"
    calibration.to_csv(calibration_path, index=False)
    replay.to_csv(replay_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_report(report_path, calibration, summary, args)
    print(f"Saved calibration audit: {calibration_path}")
    print(f"Saved ROI replay audit: {replay_path}")
    print(f"Saved source-parity report: {report_path}")
    print(calibration.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
