"""Canonical BGR input matching the existing quality-95 JPEG frame workflow."""
from __future__ import annotations

import cv2
import numpy as np


FRAME_POLICY = {
    "version": "raw_frame_candidate_20260912_v1",
    "decoder": "OpenCV_FFMPEG_default_orientation",
    "input_channels": "BGR",
    "geometry": "native_decoded_dimensions; no_crop_flip_or_resize_before_YOLO",
    "legacy_frame_equivalence": "JPEG_quality_95_encode_decode_in_memory",
    "temperature_input_channels": "BGR",
    "roi_coordinates": "YOLO_keypoints_in_native_image_pixels",
    "note": "preserves_legacy_JPEG_preprocessing_not_native_radiometric_temperature",
}


def canonical_bgr(frame):
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected uint8 BGR frame")
    ok, payload = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("JPEG preprocessing failed")
    restored = cv2.imdecode(payload, cv2.IMREAD_COLOR)
    if restored is None or restored.shape != frame.shape:
        raise ValueError("JPEG roundtrip changed native geometry")
    return restored
