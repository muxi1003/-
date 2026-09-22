from pathlib import Path

import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.nn.modules import CBAM

from scripts.yolo11_pose_attention_experiment import ATTENTION_LAYERS, has_cbam, inject_cbam


ROOT = Path(__file__).resolve().parents[1]


def test_cbam_injection_is_shape_preserving_and_idempotent():
    model = YOLO(str(ROOT / "yolo11n-pose.pt"))
    baseline_parameters = sum(parameter.numel() for parameter in model.model.parameters())
    sample = torch.rand(1, 3, 64, 64)
    model.model.eval()
    with torch.inference_mode():
        baseline_output = model.model(sample)[0]

    inject_cbam(model)
    first_parameters = sum(parameter.numel() for parameter in model.model.parameters())
    inject_cbam(model)
    second_parameters = sum(parameter.numel() for parameter in model.model.parameters())

    assert has_cbam(model)
    assert first_parameters == second_parameters
    assert 0 < first_parameters - baseline_parameters < 100_000
    for index in ATTENTION_LAYERS:
        layer = model.model.model[index]
        assert isinstance(layer, nn.Sequential)
        assert sum(isinstance(item, CBAM) for item in layer) == 1

    model.model.eval()
    with torch.inference_mode():
        attention_output = model.model(sample)[0]
    assert torch.allclose(attention_output, baseline_output, atol=1e-5, rtol=1e-5)
