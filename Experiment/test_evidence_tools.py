"""Synthetic guard tests; these are NOT animal-data experimental results."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

from experiment_common import *


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


ref = module("reference_tools", REFERENCE / "reference_tools.py")
holdout = module("holdout_tools", HOLDOUT / "holdout_tools.py")
import sys
sys.path.insert(0, str(HOLDOUT))
frozen = module("frozen_rr_evaluation", HOLDOUT / "frozen_rr_evaluation.py")


def reference_fixture():
    window = {"window_id": "synthetic", "annotation_round": "R1", "duration_seconds": "30",
              "annotation_status": "complete", "manual_breath_count": "1", "annotator": "test",
              "predictions_hidden": "yes", "event_definition": "expiration_peak"}
    event = {"window_id": "synthetic", "annotation_round": "R1", "event_id": "e1",
             "event_time_seconds": "1.0", "event_start_seconds": "", "event_end_seconds": "",
             "event_type": "expiration_peak", "confidence": "confirmed", "annotator": "test", "notes": ""}
    intervals = pd.DataFrame(columns=["window_id", "annotation_round", "start_seconds", "end_seconds", "reason", "annotator"])
    return pd.DataFrame([window]), pd.DataFrame([event]), intervals


class EvidenceTests(unittest.TestCase):
    def test_one_to_one_prevents_double_credit(self):
        matched, fp, fn = ref.match_events([1., 1.1], [1.05])
        self.assertEqual((len(matched), len(fp), len(fn)), (1, 1, 0))

    def test_assignment_not_greedy_nearest(self):
        matched, fp, fn = ref.match_events([.29, .60], [0., .30], .31)
        self.assertEqual((len(matched), len(fp), len(fn)), (2, 0, 0))

    def test_empty_event_arrays(self):
        self.assertEqual(ref.match_events([], [1.]), ([], [], [0]))
        self.assertEqual(ref.match_events([], []), ([], [], []))

    def test_valid_reference(self):
        self.assertEqual(ref.validate_tables(*reference_fixture()), [])

    def test_blank_count_is_not_zero(self):
        w, e, i = reference_fixture()
        w.loc[0, "manual_breath_count"] = ""
        self.assertTrue(any("integer_manual" in x for x in ref.validate_tables(w, e, i)))

    def test_explicit_zero_valid(self):
        w, e, i = reference_fixture()
        w.loc[0, "manual_breath_count"] = "0"
        self.assertEqual(ref.validate_tables(w, e.iloc[:0], i), [])

    def test_boundary_event_rejected(self):
        w, e, i = reference_fixture()
        e.loc[0, "event_time_seconds"] = "30"
        self.assertTrue(any("outside" in x for x in ref.validate_tables(w, e, i)))

    def test_missing_reference_visibility_blocks_complete(self):
        w, e, i = reference_fixture()
        w.loc[0, "predictions_hidden"] = ""
        self.assertTrue(any("hidden" in x for x in ref.validate_tables(w, e, i)))

    def test_duplicate_event_rejected(self):
        w, e, i = reference_fixture()
        self.assertIn("duplicate_event_id", ref.validate_tables(w, pd.concat([e, e]), i))

    def test_occluded_event_rejected(self):
        w, e, i = reference_fixture()
        i.loc[0] = ["synthetic", "R1", "0.5", "2", "nose_out_of_view", "test"]
        self.assertTrue(any("unobservable" in x for x in ref.validate_tables(w, e, i)))

    def test_yolo_only_not_independent(self):
        problems = holdout.review_blockers({"yolo_training_exposure": "no"})
        self.assertTrue(any("rr_tuning" in p for p in problems))
        self.assertTrue(any("rr_result" in p for p in problems))

    def test_confirmed_exposure_still_requires_identity(self):
        row = {"yolo_training_exposure": "no", "rr_tuning_exposure": "no", "rr_result_viewing_exposure": "no"}
        self.assertTrue(any("cow" in p for p in holdout.review_blockers(row)))

    def test_changed_method_hash_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as name:
            path = Path(name)
            write_text(path / "method.py", "x=1\n")
            write_json(path / "freeze_manifest.json", {"files": [{"snapshot_relative_path": "method.py", "sha256": "wrong"}]})
            with self.assertRaises(ValueError):
                holdout.verify_method(path, sha256(path / "freeze_manifest.json"))

    def test_missing_predictions_not_silently_dropped(self):
        windows = pd.DataFrame([{"window_id": "w1", "duration_seconds": "30"}, {"window_id": "w2", "duration_seconds": "30"}])
        pred = pd.DataFrame([{"window_id": "w1", "duration_seconds": "30", "prediction_status": "ok", "predicted_count": "10", "method_manifest_sha256": "h"}])
        with self.assertRaises(ValueError):
            frozen.validate_predictions(pred, windows, "h")

    def test_abstention_is_not_zero(self):
        windows = pd.DataFrame([{"window_id": "w1", "duration_seconds": "30"}])
        pred = pd.DataFrame([{"window_id": "w1", "duration_seconds": "30", "prediction_status": "abstain", "predicted_count": "0", "method_manifest_sha256": "h"}])
        with self.assertRaises(ValueError):
            frozen.validate_predictions(pred, windows, "h")
        pred.loc[0, "predicted_count"] = ""
        frozen.validate_predictions(pred, windows, "h")

    def test_metrics_not_pearson_squared(self):
        frame = pd.DataFrame({"truth_rr_bpm": [10., 20., 30.], "predicted_rr_bpm": [20., 30., 40.],
                              "truth_count": [5., 10., 15.], "predicted_count": [10., 15., 20.]})
        self.assertAlmostEqual(measurements(frame)["rr_r2"], -.5)
        self.assertAlmostEqual(np.corrcoef(frame.truth_rr_bpm, frame.predicted_rr_bpm)[0, 1] ** 2, 1.)

    def test_complete_event_score_end_to_end(self):
        w, e, i = reference_fixture()
        with tempfile.TemporaryDirectory(dir=ROOT) as name:
            path = Path(name)
            forms = path / "reference"
            write_csv(forms / "annotation_windows.csv", w)
            write_csv(forms / "reference_events.csv", e)
            write_csv(forms / "unobservable_intervals.csv", i)
            write_csv(path / "events.csv", [{"window_id": "synthetic", "event_id": "p1", "event_time_seconds": 1.2}])
            write_csv(path / "windows.csv", [{"window_id": "synthetic", "prediction_status": "ok",
                                              "timebase_verified": "yes", "timebase": "annotation_video_seconds", "duration_seconds": 30}])
            ref.score(forms, path / "events.csv", path / "windows.csv", path / "scored", "R1")
            import json
            scored = json.loads((path / "scored" / "event_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual((scored["tp"], scored["fp"], scored["fn"], scored["f1"]), (1, 0, 0, 1.))

    def test_pending_reference_does_not_score(self):
        w, e, i = reference_fixture()
        w.loc[0, "annotation_status"] = "pending"
        w.loc[0, "manual_breath_count"] = ""
        with tempfile.TemporaryDirectory(dir=ROOT) as name:
            path = Path(name)
            write_csv(path / "annotation_windows.csv", w)
            write_csv(path / "reference_events.csv", e.iloc[:0])
            write_csv(path / "unobservable_intervals.csv", i)
            with self.assertRaisesRegex(ValueError, "No complete"):
                ref.score(path, path / "not_created.csv", path / "not_created_either.csv", path / "output", "R1")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(EvidenceTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    write_json(ROOT / ("test_results_" + stamp() + ".json"), {
        "scope": "synthetic_software_guard_tests_not_real_animal_metrics", "tests_run": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors), "successful": result.wasSuccessful()})
    raise SystemExit(0 if result.wasSuccessful() else 1)
