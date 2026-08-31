from __future__ import annotations

import numpy as np
import pandas as pd

from basic_descriptor_mlp.features import FeatureState


def test_row_normalization_preserves_zero_rows() -> None:
    values = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    result = FeatureState._normalize_rows(values)
    np.testing.assert_allclose(result[0], [0.6, 0.8])
    np.testing.assert_allclose(result[1], [0.0, 0.0])


def test_one_hot_unknown_is_zero() -> None:
    result = FeatureState._one_hot(pd.Series(["a", "unknown"]), ["a", "b"])
    np.testing.assert_array_equal(result, [[1.0, 0.0], [0.0, 0.0]])
