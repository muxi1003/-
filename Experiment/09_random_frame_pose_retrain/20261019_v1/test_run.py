import unittest

import run


class SplitTests(unittest.TestCase):
    def test_exact_80_20_counts(self):
        split = run.assignments([f"frame_{i:06d}" for i in range(40542)])
        self.assertEqual(sum(s == "val" for s in split.values()), 8109)
        self.assertEqual(sum(s == "train" for s in split.values()), 32433)

    def test_repeatable_and_order_independent(self):
        keys = [f"cowA:{i:03d}" for i in range(100)]
        self.assertEqual(run.assignments(keys), run.assignments(keys[::-1]))

    def test_not_grouped(self):
        split = run.assignments([f"cowA:{i:03d}" for i in range(100)])
        self.assertEqual(set(split.values()), {"train", "val"})

    def test_unique_membership(self):
        with self.assertRaises(ValueError):
            run.assignments(["same", "same"])
        with self.assertRaises(ValueError):
            run.assignments([])


if __name__ == "__main__":
    unittest.main()
