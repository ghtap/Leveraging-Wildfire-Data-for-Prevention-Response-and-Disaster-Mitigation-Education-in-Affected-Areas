"""Tes regresi iter_blocks: jumlah tanggal lengkap, urutan sama, tidak ada duplikasi."""
import numpy as np
import pandas as pd
from src.data.build_features import FeatureBuilder


def test_iter_blocks_monthly():
    builder = FeatureBuilder.__new__(FeatureBuilder)
    builder.dates = pd.date_range("2024-01-01", "2024-04-30")
    builder.t_first = 12
    builder.t_last = 110
    builder.block = lambda t: pd.DataFrame({"idx": t})
    blocks = list(builder.iter_blocks())
    assert len(blocks) == 4
    found = np.concatenate([b["idx"].to_numpy() for b in blocks])
    np.testing.assert_array_equal(found, np.arange(12, 111))
    assert all(b["idx"].size <= 31 for b in blocks)


def test_iter_blocks_empty_range():
    builder = FeatureBuilder.__new__(FeatureBuilder)
    builder.dates = pd.date_range("2024-01-01", periods=10)
    builder.t_first = 9
    builder.t_last = 3
    builder.block = lambda t: pd.DataFrame({"idx": t})
    assert list(builder.iter_blocks()) == []
