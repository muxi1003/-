# Lindian Long-Video Fixed-Window Protocol

The 73 labelled `all_can_use` clips are short selections from the Lindian raw-video archive. `build_lindian_long_video_manifest.py` maps each selection to its sibling timestamped raw source by SHA-256 content matching. A raw-video centre window is retained only as a diagnostic baseline because it can contain no visible nostril.

`build_lindian_anchored_fixed_window_manifest.py` uses the short clip's first, middle, and final frame to locate the same sequence in the raw video by thumbnail error. It then centres the 30-second window on that matched sequence. The matcher uses image content, timestamps, and a fixed error threshold, never RR or breath-count labels. This is appropriate for retrospective fixed-window analysis, but it is not an external generalization protocol because the short clip had already been selected from the same raw recording.

The generated `lindian_fixed30_window_annotation_template.csv` intentionally separates `short_clip_reference_rr_bpm_not_window_truth` from the blank fixed-window manual labels. The former is provenance only and must not be used to calculate fixed-window RR R-squared, MAE, or RMSE.

`extract_lindian_fixed_windows.py` exports every selected window at the raw video frame rate. Centre-window diagnostics live in `Dataset_new/72video/lindian_fixed30_frames/`; anchored windows live in `Dataset_new/72video/lindian_anchored30_frames/`. The extraction report records the requested and actual start frame, source fps, and exact frame count.

After manual review, record the observed 30-second breath count. For a complete 30-second window, manual RR is twice the count. Only rows with a completed window-specific count and an explicit inclusion decision may be used for fixed-window RR evaluation.

Before exporting all anchored windows for temperature mapping, apply a YOLO bilateral-visibility quality gate. The source anchor proves that the labelled short clip is present but does not prove 30 seconds of continuous nostril visibility. In the `bs1479` smoke test, the raw-video centre window had 32/260 frames with either nostril detected, the anchored window improved to 90/260, while the original 79-frame selected clip had 79/79. Thus neither the centre nor anchored 30-second cohort may be treated as ready for RR evaluation until a non-truth quality screen defines which windows are suitable.

The current 1 Hz screen with YOLO keypoint confidence >= 0.50 and bilateral-valid fraction >= 0.70 retains 49/73 anchored windows. `lindian_anchored30_quality_gated_annotation_template.csv` contains only those 49 candidates. Its short-clip RR column is context only; the 30-second breath count remains blank and must be manually recorded before any fixed-window RR metric is calculated.
