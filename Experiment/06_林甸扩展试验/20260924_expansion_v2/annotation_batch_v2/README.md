# Human respiratory-event annotation

Open `index.html` in a local browser. It contains 8 approximately 30-second videos from 8 distinct filename cow IDs. If a video is blocked by local-file restrictions, use the `选择对应视频` control to select the exact matching `.webm` under `media/`.

For each window, enter the annotator and whether algorithm predictions have actually been hidden from this annotation round. Mark each reliably visible expiration peak at its video playback time. Record uncertain events as uncertain, and record any unobservable interval with a reason. A complete window requires all events confirmed and an integer manual count; severe exposure or obscured respiration should remain partial or unobservable rather than being treated as zero breaths. Do not transfer old predictions or count annotations into this page.

When done, export all three CSV tables and the JSON backup into a separate folder. The page saves progress only in its browser storage; that is not a durable project reference. Keep the export files together and provide their directory path for source-frame time binding, one-to-one event matching, and same-input method scoring.

The `.webm` files are viewing copies. YOLO/BGR-RF temperature inference must use the original MP4 frames and `frame_time_map.csv`, not the viewing-copy pixels. The source start is recorded in `segments.csv`; the browser timeline starts at zero in every clip.
