# Lindian 190588 source-time 30-second pilot

Date: 2026-09-24. This is an isolated development pilot. It does not replace the frozen 47-window comparison or create independent test data.

## Input and procedure

- Source: `E:/real/use_code/林甸红外视频/0805/B1号舍泌乳牛下午0805时间405挤奶之后进食/20230805T164945n190588/20230805T164945n190588.MP4`
- Source SHA-256: `48a355a2850dc5637b1310a5c4441cd787db43bd030c68de7a0d17081f4c0c79`
- Cutter: `scripts/cut_lindian_contiguous_30s.py`; nonoverlapping source-PTS bins of 30 seconds. A bin is skipped if an internal frame interval exceeds 0.5 seconds or either edge lacks more than 0.25 seconds of coverage.
- Encoded clips are CFR MP4s of approximately 30 seconds for playback. `frame_time_map.csv` preserves every output frame's original source frame and PTS. Do not infer source time from the playback frame number alone.
- The MP4s are viewing copies. YOLO/RF temperature inference must use frames decoded from the original source and the saved frame map; a second lossy encode can change the pseudo-color pixels used by the RF mapper.
- Source metadata: 2987 frames, 8.6451 fps; the last decoded frame is at 345.010 seconds. Source PTS gaps: 70.218 to 72.795 seconds (2.577 seconds) and 250.172 to 252.518 seconds (2.347 seconds).

## Outputs

- Nine playable clips: `000_030`, `030_060`, `090_120`, `120_150`, `150_180`, `180_210`, `210_240`, `270_300`, `300_330`.
- Two skipped bins: `060_090` and `240_270`, each crossing a source PTS gap. Full disposition is in `segments.csv`.
- All nine clips were fully decoded after encoding; decoded frame counts matched `segments.csv` and their playback durations were within 0.05 seconds of 30 seconds.
- `annotation_windows_template.csv` has nine pending, count-free rows. No new respiratory ground truth has been entered.
- `index.html` is a separate offline annotation page for the nine clips. It uses browser-playable VP8 WebM copies from `../190588_browser_vp8_v1/`, encoded directly from the original source frames. The MP4 and WebM sets share the same source-frame windows and each has its own `frame_time_map.csv`.
- The page has a unique localStorage key and annotation round `L190588-P1`; it does not read or overwrite the older R2 page. Browser storage is not the deliverable: export the three CSVs and a JSON backup after annotating, and retain the backup outside browser cache. This page has no automatic write-back to project reference files.

## Exploratory image screen

`scripts/screen_lindian_contiguous_windows.py` sampled 30 frames per clip with the existing YOLO11n-Pose checkpoint (SHA-256 `12C3AD363A00E008AB6D8B859AC6134DBD6582BBFD9CC72AFEE6884CCD304A09`). The full per-clip table is `screen_yolo_1hz.csv`.

| Source-time bin | Bilateral keypoints at confidence >= 0.5 |
| --- | ---: |
| 000-030 | 12/30 |
| 030-060 | 20/30 |
| 090-120 | 17/30 |
| 120-150 | 18/30 |
| 150-180 | 17/30 |
| 180-210 | 21/30 |
| 210-240 | 30/30 |
| 270-300 | 22/30 |
| 300-330 | 27/30 |

The ROI near-white fraction is only a palette-image proxy, not a validated radiometric saturation measure. Preview frames from `210_240`, `270_300`, and `300_330` still show large bright areas around the muzzle. High keypoint visibility does not establish an observable breathing-temperature signal. Full playback and manual observability review are required before designating a clip usable.

## Existing window caveat

The earlier `190588_anchored30s.mp4` has a saved source start of 63.389 seconds and a playback duration of 29.959 seconds. The corresponding frame-index span (548-806) covers source PTS approximately 62.365-94.253 seconds and crosses the 2.577-second gap. Its old labels and metrics remain unchanged; new source-time clips must not silently replace it.

## Evaluation boundary

No RR MAE, R-squared, exact-count accuracy, or event F1 can be calculated on these clips yet. They have no independent manual counts or event timestamps. All nine derive from one original recording and must remain in the same cow/source-video group. The next step is manual full-clip observability and event annotation, followed by a frozen same-input method comparison on eligible windows. Do not select clips by agreement with model predictions.

The annotation page records timestamps on each 30-second viewing copy. Before event-level scoring against source-frame predictions, map those timestamps through the corresponding `frame_time_map.csv` rather than treating viewing-copy frame numbers as original source PTS.
