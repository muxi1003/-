"""Synthetic tests only, never animal reference events."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import evaluate_event_submission as ev
from test_evidence_tools import reference_fixture


class SubmissionTests(unittest.TestCase):
    def test_numeric_csv_json_equivalence(self):
        ev.compare_export(pd.DataFrame([{'event_time_seconds': '12', 'notes': ''}]),
                          [{'event_time_seconds': '12.0', 'notes': ''}])

    def test_changed_notes_rejected(self):
        with self.assertRaises(ValueError):
            ev.compare_export(pd.DataFrame([{'notes': 'original'}]), [{'notes': 'changed'}])

    def test_changed_times_rejected(self):
        with self.assertRaises(ValueError):
            ev.compare_export(pd.DataFrame([{'event_time_seconds': '12'}]), [{'event_time_seconds': '12.1'}])

    def check_invalid_prediction(self, duration, count, status='ok'):
        w, e, intervals = reference_fixture()
        pe = pd.DataFrame([dict(window_id='synthetic', event_id='p1', event_time_seconds='1')])
        if status == 'abstain':
            pe = pe.iloc[:0]
        pw = pd.DataFrame([dict(window_id='synthetic', duration_seconds=duration, predicted_count=count,
                               prediction_status=status, timebase_verified='true', timebase='annotation_video_seconds')])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, table in [('annotation_windows.csv', w), ('reference_events.csv', e), ('unobservable_intervals.csv', intervals)]:
                ev.write_csv(root / 'reference' / name, table)
            ev.write_csv(root / 'pe.csv', pe)
            ev.write_csv(root / 'pw.csv', pw)
            with self.assertRaises(ValueError):
                ev.ref.score(root / 'reference', root / 'pe.csv', root / 'pw.csv', root / 'out', 'R1')

    def test_nonfinite_prediction_duration_rejected(self):
        for value in ['nan', 'inf', '-inf', '']:
            with self.subTest(value=value):
                self.check_invalid_prediction(value, '1')

    def test_inconsistent_declared_count_rejected(self):
        for value in ['999', '-1', '1.5', 'nan']:
            with self.subTest(value=value):
                self.check_invalid_prediction('30', value)

    def test_abstention_cannot_declare_zero_count(self):
        self.check_invalid_prediction('30', '0', status='abstain')

    def test_abstention_is_false_negative_and_partial_excluded(self):
        w, e, intervals = reference_fixture()
        w['annotation_round'] = 'R2'
        e['annotation_round'] = 'R2'
        w['window_id'] = 'abstained'
        e['window_id'] = 'abstained'
        extra = w.iloc[0].copy()
        extra['window_id'] = 'observed'
        partial = extra.copy()
        partial['window_id'] = 'partial'
        partial['annotation_status'] = 'partial'
        w = pd.concat([w, pd.DataFrame([extra, partial])], ignore_index=True)
        extra_e = e.iloc[0].copy()
        extra_e['window_id'] = 'observed'
        partial_e = extra_e.copy()
        partial_e['window_id'] = 'partial'
        e = pd.concat([e, pd.DataFrame([extra_e, partial_e])], ignore_index=True)
        pw = pd.DataFrame([dict(window_id=wid, duration_seconds='30', predicted_count='' if wid == 'abstained' else '1',
                               prediction_status='abstain' if wid == 'abstained' else 'ok',
                               timebase_verified='true', timebase='annotation_video_seconds') for wid in w.window_id])
        pe = pd.DataFrame([dict(window_id=wid, event_id='p1', event_time_seconds=1.) for wid in ['observed', 'partial']])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, table in [('annotation_windows.csv', w), ('reference_events.csv', e), ('unobservable_intervals.csv', intervals)]:
                ev.write_csv(root / 'reference' / name, table)
            result = ev.evaluate(root / 'reference', pw, pe, root / 'out', {wid: wid for wid in w.window_id})
            self.assertEqual((result['tp'], result['fp'], result['fn'], result['n']), (1, 0, 1, 1))
            self.assertAlmostEqual(result['f1'], 2/3)


if __name__ == '__main__':
    unittest.main()
