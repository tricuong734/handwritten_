"""Test split isolation, CNN alignment and protection of held-out evaluation."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.nv2 import make_split, validate_predictions, train, save_json
from src.nv2_models import model_specs


class NV2Tests(unittest.TestCase):
    def test_split_is_disjoint_complete_and_reproducible(self):
        y = np.tile(np.arange(10), 100)
        parts = make_split(y, 42)
        self.assertEqual([len(x) for x in parts], [700, 150, 150])
        self.assertEqual(len(set(np.concatenate(parts))), len(y))
        for left, right in zip(parts, make_split(y, 42)):
            np.testing.assert_array_equal(left, right)
            self.assertEqual(len(np.unique(y[left])), 10)

    def test_cnn_alignment_and_invalid_predictions(self):
        frame = pd.DataFrame({'row_id': [3, 1, 2], 'y_pred': [7, 8, 9]})
        np.testing.assert_array_equal(validate_predictions(frame, np.array([1, 2, 3])), [8, 9, 7])
        for bad in [frame.iloc[:2], frame.assign(row_id=[1, 1, 2]), frame.assign(y_pred=[-1, 2, 3])]:
            with self.assertRaises(ValueError):
                validate_predictions(bad, np.array([1, 2, 3]))

    def test_retraining_blocked_after_test_opened(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            save_json(path / 'config.json', {})
            save_json(path / 'test_frozen.json', {})
            with self.assertRaisesRegex(ValueError, 'tap test'):
                train(SimpleNamespace(output=path))

    def test_required_model_coverage(self):
        specs = model_specs()
        self.assertEqual(set(specs), {'logistic_regression', 'knn', 'svm_linear', 'svm_rbf', 'random_forest'})
        self.assertEqual(specs['svm_linear'][0].kernel, 'linear')
        self.assertEqual(specs['svm_rbf'][0].kernel, 'rbf')


if __name__ == '__main__':
    unittest.main()
