"""Synthetic tests for submission preservation and count-only evidence guards."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import process_annotation_submission as submission


class SubmissionTests(unittest.TestCase):
    def test_declaration_is_bound_to_current_submission(self):
        declaration = {"jiufu271": {"predictions_hidden": True, "user_statement": "test confirmation", "annotation_windows_sha256": "a"}}
        self.assertTrue(submission.declaration_for("jiufu271", declaration, "a")["predictions_hidden"])
        with self.assertRaises(ValueError):
            submission.declaration_for("jiufu271", declaration, "b")

    def test_blinding_not_inferred_from_cohort(self):
        declaration = {"jiufu271": {"predictions_hidden": False, "user_statement": "test correction", "annotation_windows_sha256": "a"}}
        self.assertFalse(submission.declaration_for("jiufu271", declaration, "a")["predictions_hidden"])
        declaration["jiufu271"]["predictions_hidden"] = "false"
        with self.assertRaises(ValueError):
            submission.declaration_for("jiufu271", declaration, "a")

    def test_internal_source_start_cannot_drift(self):
        w = pd.DataFrame([dict(window_id="a_anchored_30s", video_id="a", duration_seconds="30", source_start_seconds="6", source_path="E:/a.mp4")])
        e = pd.DataFrame([dict(video_id="a", duration_seconds="30", start_seconds="5", raw_source_path="E:/a.mp4")])
        with self.assertRaisesRegex(ValueError, "internal source start"):
            submission.check_identity(w, e, False)

    def test_gb18030_and_extra_notes_preserved(self):
        source = "window_id,备注,Unnamed: 2\nw1,看不清,\n"
        table, encoding = submission.decode_table(source.encode("gb18030"))
        self.assertEqual(encoding, "gb18030")
        self.assertEqual(table.loc[0, "备注"], "看不清")
        self.assertEqual(table.loc[0, "Unnamed: 2"], "")

    def test_blank_and_invalid_are_not_zero(self):
        values = pd.Series(["", "0", "2", "-1", "1.5", "nan", "inf", "x"])
        self.assertEqual(submission.valid_counts(values).tolist(), [False, True, True, False, False, False, False, False])

    def test_frozen_duration_cannot_drift(self):
        w = pd.DataFrame([dict(window_id="a_anchored_30s", video_id="a", duration_seconds="29")])
        e = pd.DataFrame([dict(video_id="a", duration_seconds="30")])
        with self.assertRaisesRegex(ValueError, "duration"):
            submission.check_identity(w, e, False)

    def test_unknown_window_cannot_enter(self):
        w = pd.DataFrame([dict(window_id="b_anchored_30s", video_id="b", duration_seconds="30")])
        e = pd.DataFrame([dict(video_id="a", duration_seconds="30")])
        with self.assertRaisesRegex(ValueError, "membership"):
            submission.check_identity(w, e, False)

    def test_external_source_start_cannot_drift(self):
        w = pd.DataFrame([dict(window_id="w", source_path="E:/a.mp4", source_start_seconds="1", duration_seconds="30")])
        e = pd.DataFrame([dict(window_id="w", source_path="E:\\a.mp4", start_seconds="0", duration_seconds="30")])
        with self.assertRaisesRegex(ValueError, "source start"):
            submission.check_identity(w, e, True)

    def test_external_paths_normalized_without_rewriting(self):
        w = pd.DataFrame([dict(window_id="w", source_path="E:/a.mp4", source_start_seconds="0", duration_seconds="30")])
        e = pd.DataFrame([dict(window_id="w", source_path="E:\\a.mp4", start_seconds="0", duration_seconds="30")])
        self.assertEqual(len(submission.check_identity(w, e, True)), 1)
        self.assertEqual(w.iloc[0].source_path, "E:/a.mp4")

    def test_archive_exact_bytes(self):
        with tempfile.TemporaryDirectory(dir=submission.ROOT) as name:
            root = Path(name)
            source = root / "source"
            source.mkdir()
            for filename in ("annotation_windows.csv", "reference_events.csv", "unobservable_intervals.csv"):
                (source / filename).write_bytes("window_id,备注\nw,看不清\n".encode("gb18030"))
            tables, manifest = submission.archive_group(source, root / "snapshot")
            for entry in manifest:
                self.assertEqual(Path(entry["source"]).read_bytes(), Path(entry["snapshot"]).read_bytes())
            self.assertEqual(tables["annotation_windows.csv"].loc[0, "备注"], "看不清")
            with self.assertRaises(FileExistsError):
                submission.archive_group(source, root / "snapshot")


if __name__ == "__main__":
    unittest.main()
