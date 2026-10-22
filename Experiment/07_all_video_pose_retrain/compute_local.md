# Existing Local Training Environment

Snapshot date: 2026-10-16, Asia/Shanghai. Reuse the existing environment; do not install or update packages.

- Python: `E:/real/anaconda/envs/plant_gpu/python.exe`, 3.10.20
- Torch: `2.12.0.dev20260322+cu128`, CUDA 12.8
- Ultralytics: `8.4.40`
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB
- Pretrained weight: project `yolo11n-pose.pt`, SHA-256 `869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0`
- YOLO config is isolated under this experiment, not the global roaming profile.

From `E:/real/use_code/yoloV8`, execute this read-only forward probe verbatim:

```powershell
& 'E:\real\anaconda\envs\plant_gpu\python.exe' 'Experiment\07_all_video_pose_retrain\verify_compute.py'
```

Success requires exit code 0 plus a `WITNESS` line with a nonempty finite tensor and the GPU name. This probe is not training, model accuracy validation, or a guarantee that a full epoch fits memory. Training startup and the first epoch must be checked separately.

2026-10-16 validation: initial probe caught a missing `f` in the copied checksum (63 rather than 64 hex characters). The actual weight was not modified. After correcting the documented and executable checksums, independent verbatim replay exited 0 with `WITNESS (1, 56, 8400) NVIDIA GeForce RTX 5060 Laptop GPU`.
