from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.models.yolo.pose.train import PoseTrainer
from ultralytics.nn.modules import CBAM
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import get_flops, get_num_params


DEFAULT_DATA = ROOT / "Dataset_new" / "72video" / "dataset.yaml"
DEFAULT_WEIGHTS = ROOT / "yolo11n-pose.pt"
P2_CONFIG = ROOT / "models" / "yolo11n-pose-p2.yaml"
DEFAULT_PROJECT = ROOT / "runs" / "pose_attention"
ATTENTION_LAYERS = (16, 19, 22)
ATTENTION_LEVELS = {16: "P3/8", 19: "P4/16", 22: "P5/32"}
VARIANT_LAYERS = {
    "baseline": (),
    "cbam-p3": (16,),
    "cbam-p345": ATTENTION_LAYERS,
    "p2": (),
}
VARIANTS = tuple(VARIANT_LAYERS)


def _pose_model(model: Any) -> PoseModel:
    """Return the Ultralytics PoseModel held by a YOLO wrapper or checkpoint model."""
    if isinstance(model, YOLO):
        model = model.model
    if not isinstance(model, PoseModel):
        raise TypeError(f"Expected PoseModel, got {type(model).__name__}")
    return model


def _output_channels(layer: nn.Module) -> int:
    """Read the output width of a YOLO11 C3k2 feature block."""
    cv2 = getattr(layer, "cv2", None)
    conv = getattr(cv2, "conv", None)
    channels = getattr(conv, "out_channels", None)
    if not isinstance(channels, int):
        raise TypeError(f"Cannot infer output channels from {type(layer).__name__}")
    return channels


def has_cbam(model: Any, layer_indices: Iterable[int] = ATTENTION_LAYERS) -> bool:
    """Return True when every requested feature layer already contains CBAM."""
    pose_model = _pose_model(model)
    for index in layer_indices:
        layer = pose_model.model[index]
        if not isinstance(layer, nn.Sequential) or not any(isinstance(item, CBAM) for item in layer):
            return False
    return True


def cbam_layers(model: Any) -> tuple[int, ...]:
    """Return the YOLO feature-layer indices containing CBAM."""
    pose_model = _pose_model(model)
    return tuple(
        index
        for index in ATTENTION_LAYERS
        if isinstance(pose_model.model[index], nn.Sequential)
        and any(isinstance(item, CBAM) for item in pose_model.model[index])
    )


def inject_cbam(
    model: Any,
    layer_indices: Iterable[int] = ATTENTION_LAYERS,
) -> PoseModel:
    """Append CBAM to the P3/P4/P5 blocks without changing YOLO layer indices."""
    pose_model = _pose_model(model)
    indices = tuple(layer_indices)

    for index in indices:
        layer = pose_model.model[index]
        if isinstance(layer, nn.Sequential) and any(isinstance(item, CBAM) for item in layer):
            continue

        channels = _output_channels(layer)
        reference = next(layer.parameters())
        attention = CBAM(channels, kernel_size=7)
        nn.init.zeros_(attention.channel_attention.fc.weight)
        nn.init.zeros_(attention.channel_attention.fc.bias)
        nn.init.zeros_(attention.spatial_attention.cv1.weight)

        # Both zero-logit sigmoid gates initially equal 0.5. A trainable
        # depthwise 4x calibration therefore makes the complete CBAM branch an
        # exact identity at initialization and preserves pretrained features.
        calibration = nn.Conv2d(channels, channels, 1, groups=channels, bias=False)
        nn.init.constant_(calibration.weight, 4.0)
        attention.to(device=reference.device, dtype=reference.dtype)
        calibration.to(device=reference.device, dtype=reference.dtype)
        wrapped = nn.Sequential(layer, attention, calibration)

        # BaseModel._predict_once uses these parser annotations on each top-level layer.
        wrapped.i = layer.i
        wrapped.f = layer.f
        wrapped.type = f"{layer.type}+CBAM"
        wrapped.np = sum(parameter.numel() for parameter in wrapped.parameters())
        pose_model.model[index] = wrapped

    pose_model.cbam_feature_layers = indices
    pose_model.cbam_feature_levels = tuple(ATTENTION_LEVELS[index] for index in indices)
    return pose_model


class CBAMPoseTrainer(PoseTrainer):
    """Pose trainer that loads pretrained weights before adding trainable CBAM blocks."""

    attention_layers = ATTENTION_LAYERS

    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ) -> PoseModel:
        model = PoseModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            data_kpt_shape=self.data["kpt_shape"],
            verbose=verbose and RANK == -1,
        )

        # A resumed CBAM checkpoint already has wrapped state-dict keys. Fresh
        # YOLO11 weights use the original keys and must be loaded first.
        source_has_cbam = False
        if isinstance(weights, PoseModel):
            source_has_cbam = has_cbam(weights, self.attention_layers)
        if source_has_cbam:
            inject_cbam(model, self.attention_layers)
        if weights:
            model.load(weights)
        if not source_has_cbam:
            inject_cbam(model, self.attention_layers)
        return model


def _torch_device(device: str) -> torch.device:
    if device.lower() == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    index = device.split(",", maxsplit=1)[0]
    index = index.removeprefix("cuda:")
    return torch.device(f"cuda:{index}")


def build_variant(weights: Path, variant: str) -> YOLO:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {VARIANTS}")
    if variant == "p2" and weights.resolve() == DEFAULT_WEIGHTS.resolve():
        model = YOLO(str(P2_CONFIG))
        model.load(str(weights))
    else:
        model = YOLO(str(weights))
    expected = VARIANT_LAYERS[variant]
    present = cbam_layers(model)
    if present and present != expected:
        raise ValueError(
            f"Checkpoint has CBAM layers {present}, but variant {variant!r} expects {expected}"
        )
    if expected and not present:
        inject_cbam(model, expected)
    return model


def profile_model(
    model: YOLO,
    variant: str,
    imgsz: int,
    device: str,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    """Measure batch-1 PyTorch latency and structural cost on one device."""
    core = _pose_model(model)
    target = _torch_device(device)
    core.to(target).eval()
    sample = torch.rand(1, 3, imgsz, imgsz, device=target)

    if target.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(target)

    timings: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup):
            core(sample)
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        for _ in range(iterations):
            started = time.perf_counter()
            core(sample)
            if target.type == "cuda":
                torch.cuda.synchronize(target)
            timings.append((time.perf_counter() - started) * 1000.0)

    peak_vram_mb = (
        torch.cuda.max_memory_allocated(target) / (1024.0**2) if target.type == "cuda" else 0.0
    )
    return {
        "variant": variant,
        "imgsz": imgsz,
        "device": str(target),
        "parameters": get_num_params(core),
        "gflops": get_flops(core, imgsz),
        "latency_mean_ms": statistics.fmean(timings),
        "latency_median_ms": statistics.median(timings),
        "latency_p95_ms": sorted(timings)[max(0, int(len(timings) * 0.95) - 1)],
        "fps_batch1": 1000.0 / statistics.fmean(timings),
        "peak_vram_mb": peak_vram_mb,
        "cbam_layers": ",".join(map(str, VARIANT_LAYERS[variant])),
    }


def metric_values(metrics: Any) -> dict[str, float]:
    return {
        "box_precision": float(metrics.box.mp),
        "box_recall": float(metrics.box.mr),
        "box_map50": float(metrics.box.map50),
        "box_map50_95": float(metrics.box.map),
        "pose_precision": float(metrics.pose.mp),
        "pose_recall": float(metrics.pose.mr),
        "pose_map50": float(metrics.pose.map50),
        "pose_map50_95": float(metrics.pose.map),
        "val_preprocess_ms": float(metrics.speed.get("preprocess", 0.0)),
        "val_inference_ms": float(metrics.speed.get("inference", 0.0)),
        "val_postprocess_ms": float(metrics.speed.get("postprocess", 0.0)),
    }


def write_rows(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_profile(args: argparse.Namespace) -> None:
    rows = []
    for variant in VARIANTS:
        model = build_variant(args.weights, variant)
        rows.append(
            profile_model(model, variant, args.imgsz, args.device, args.warmup, args.iterations)
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_rows(rows, args.output)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"Wrote profile comparison: {args.output}")


def _training_kwargs(args: argparse.Namespace, variant: str) -> dict[str, Any]:
    name = args.name or f"yolo11n_pose_{variant.replace('-', '_')}"
    return {
        "data": str(args.data),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "project": str(args.project),
        "name": name,
        "exist_ok": args.exist_ok,
        "patience": args.patience,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "deterministic": True,
        "fraction": args.fraction,
        "close_mosaic": args.close_mosaic,
        "amp": True,
        "plots": not args.smoke,
        "verbose": True,
    }


def run_train(args: argparse.Namespace) -> None:
    if args.smoke:
        args.epochs = 1
        args.imgsz = min(args.imgsz, 320)
        args.batch = min(args.batch, 4)
        args.workers = 0
        args.fraction = min(args.fraction, 0.02)
        args.close_mosaic = 0

    if args.variant == "p2":
        model = YOLO(str(P2_CONFIG))
        model.load(str(args.weights))
    else:
        model = YOLO(str(args.weights))
    attention_layers = VARIANT_LAYERS[args.variant]
    trainer = CBAMPoseTrainer if attention_layers else None
    if trainer:
        CBAMPoseTrainer.attention_layers = attention_layers
    model.train(trainer=trainer, **_training_kwargs(args, args.variant))

    run_dir = Path(model.trainer.save_dir)
    best = Path(model.trainer.best)
    metrics = model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(run_dir),
        name="final_val",
        exist_ok=True,
        plots=not args.smoke,
    )
    row = {
        "variant": args.variant,
        "initial_weights": str(args.weights),
        "best_weights": str(best),
        "data": str(args.data),
        "epochs": args.epochs,
        "fraction": args.fraction,
        "seed": args.seed,
        **metric_values(metrics),
        **profile_model(model, args.variant, args.imgsz, args.device, args.warmup, args.iterations),
    }
    output = run_dir / "attention_experiment_metrics.csv"
    write_rows([row], output)
    print(json.dumps(row, ensure_ascii=False, indent=2))
    print(f"Wrote training metrics: {output}")


def _validate_checkpoint(
    weights: Path,
    variant: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model = build_variant(weights, variant)
    metrics = model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.output.parent / "validation"),
        name=variant,
        exist_ok=True,
        plots=False,
    )
    return {
        "variant": variant,
        "weights": str(weights),
        **metric_values(metrics),
        **profile_model(model, variant, args.imgsz, args.device, args.warmup, args.iterations),
    }


def run_compare(args: argparse.Namespace) -> None:
    rows = [
        _validate_checkpoint(args.baseline_weights, "baseline", args),
        _validate_checkpoint(args.attention_weights, args.attention_variant, args),
    ]
    baseline, attention = rows
    attention["delta_pose_map50"] = attention["pose_map50"] - baseline["pose_map50"]
    attention["delta_pose_map50_95"] = attention["pose_map50_95"] - baseline["pose_map50_95"]
    attention["delta_box_map50_95"] = attention["box_map50_95"] - baseline["box_map50_95"]
    attention["delta_val_inference_ms"] = (
        attention["val_inference_ms"] - baseline["val_inference_ms"]
    )
    attention["delta_latency_ms"] = attention["latency_mean_ms"] - baseline["latency_mean_ms"]
    attention["delta_parameters"] = attention["parameters"] - baseline["parameters"]
    attention["delta_gflops"] = attention["gflops"] - baseline["gflops"]
    write_rows(rows, args.output)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"Wrote checkpoint comparison: {args.output}")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and benchmark YOLO11n-Pose with P3/P4/P5 CBAM attention."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile = subparsers.add_parser("profile", help="Compare untrained structural/runtime overhead.")
    profile.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    profile.add_argument(
        "--output", type=Path, default=DEFAULT_PROJECT / "profile_comparison.csv"
    )
    _add_common(profile)
    profile.set_defaults(func=run_profile)

    train = subparsers.add_parser("train", help="Train one fair-ablation variant.")
    train.add_argument("--variant", choices=VARIANTS, required=True)
    train.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    train.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    train.add_argument("--name")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--patience", type=int, default=30)
    train.add_argument("--optimizer", default="auto")
    train.add_argument("--lr0", type=float, default=0.001)
    train.add_argument("--weight-decay", type=float, default=0.0005)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--fraction", type=float, default=1.0)
    train.add_argument("--close-mosaic", type=int, default=10)
    train.add_argument("--exist-ok", action="store_true")
    train.add_argument("--smoke", action="store_true")
    _add_common(train)
    train.set_defaults(func=run_train)

    compare = subparsers.add_parser("compare", help="Validate and profile two trained checkpoints.")
    compare.add_argument("--baseline-weights", type=Path, required=True)
    compare.add_argument("--attention-weights", type=Path, required=True)
    compare.add_argument(
        "--attention-variant",
        choices=tuple(variant for variant in VARIANTS if variant != "baseline"),
        default="cbam-p345",
    )
    compare.add_argument(
        "--output", type=Path, default=DEFAULT_PROJECT / "trained_checkpoint_comparison.csv"
    )
    _add_common(compare)
    compare.set_defaults(func=run_compare)

    args = parser.parse_args()
    for attribute in ("data", "weights", "baseline_weights", "attention_weights"):
        value = getattr(args, attribute, None)
        if value is not None and not value.exists():
            parser.error(f"{attribute.replace('_', '-')} does not exist: {value}")
    return args


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
