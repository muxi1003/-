# Lindian expansion, first source-separated annotation batch

The dated `0805`-`0810` folders were inventoried without treating their archived, renamed, or short derivatives as independent original recordings. `source_inventory.csv` contains 293 timestamp-named candidate MP4s: 73 paths used by the prior matched-input snapshot, 63 additional recordings with a cow ID present in that snapshot, 35 with no filename cow ID, 2 in an explicitly uncertain-identity directory, 1 with short/invalid video metadata, and 119 remaining candidate recordings with a new filename cow ID.

`selected_sources.csv` freezes a deterministic path-hash sample of up to two new-cow sources per date. The first batch has 8 sources: two each from August 7, 8, 9, and 10. August 5 had no remaining new-cow source under this rule; the two August 6 sources were under a directory labelled `无法分辨` and were excluded. File-name IDs have **not** been independently verified against farm records. No algorithm prediction or artificial respiratory label was used to choose these sources.

All 8 selected source-file SHA-256 digests were compared with the 73 prior matched raw-source files; there were zero byte-identical overlaps. This does not exclude repackaged copies or overlap with YOLO training images.

`annotation_batch_v2` has one first technically valid 30-second source-time window per selected recording. A source window is accepted only with <=0.5 s largest internal frame gap and <=0.25 s edge distance. Video `20230809T082528n170320` uses source seconds 30-60; the others use 0-30. All 8 decoded viewing copies passed exact frame-count checks, and all loaded as about 30-second playable media in headless Edge.

This is a **development annotation batch**, not an unseen end-to-end validation cohort. The project YOLO checkpoint's training-source provenance has not been audited. The webpage contains no predicted respiratory events; annotators still need to declare any prior algorithm exposure. Existing 47-window and 73-video result tables remain unchanged. Do not calculate or report RR/event accuracy for this batch until human event references are exported and validated.

The first inventory `20260924_expansion_v1` is retained for audit but its selection is superseded. `annotation_batch_v1` has relative paths and is not an annotation deliverable. Use `annotation_batch_v2/index.html` only.
