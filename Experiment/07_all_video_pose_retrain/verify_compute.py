"""Seeded local CUDA and YOLO forward witness; does not train or download weights."""
import json
import os
import platform

from review_gate import HERE, sha256

os.environ["YOLO_CONFIG_DIR"] = str(HERE / "ultralytics_config")

import torch
import ultralytics
from ultralytics import YOLO


def main():
    torch.manual_seed(20260929)
    weights = HERE.parents[1] / "yolo11n-pose.pt"
    assert sha256(weights) == "869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0"
    model = YOLO(str(weights)).model.eval().cuda()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 640, 640, device="cuda"))[0]
    assert output.numel() > 0 and torch.isfinite(output).all()
    print("WITNESS", tuple(output.shape), torch.cuda.get_device_name(), flush=True)
    print(json.dumps({"python": platform.python_version(), "torch": torch.__version__,
                      "ultralytics": ultralytics.__version__, "cuda": torch.version.cuda}), flush=True)


if __name__ == "__main__":
    main()
