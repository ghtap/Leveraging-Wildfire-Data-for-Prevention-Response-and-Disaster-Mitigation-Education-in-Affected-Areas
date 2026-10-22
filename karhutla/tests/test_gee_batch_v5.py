"""Unit tests untuk planner batch baru; tanpa koneksi GEE dan tanpa akses file mentah."""
import pandas as pd
import pytest
from src.data import fetch_climate as fc


@pytest.fixture
def mini_world(tmp_path, monkeypatch):
    (tmp_path / 'data' / 'interim').mkdir(parents=True)
    cfg = {
        'paths': {'interim': 'data/interim'},
        'climate': {'gee_dir': 'data/raw/gee'},
        'dates': {'start': '2025-08-01', 'end': '2025-08-03'},
    }
    data = pd.DataFrame({
        'cell_id': [10, 11, 12, 20, 21],
        'province': ['Riau', 'Riau', 'Riau', 'Jambi', 'Jambi'],
        'lon': [101.0, 101.1, 101.2, 103.0, 103.1],
        'lat': [0.5, 0.6, 0.7, -1.0, -1.1],
    })
    data.to_csv(tmp_path / 'data' / 'interim' / 'grid.csv', index=False)
    monkeypatch.setattr(fc, 'rpath', lambda rel: tmp_path / rel)
    return cfg, tmp_path


def test_plan_idempotent_and_complete(mini_world):
    cfg, root = mini_world
    args = dict(source='both', start='2025-08-01', end='2025-08-04',
                province='all', cells_per_batch=2, days_per_batch=2)
    fc.buat_rencana(cfg, **args)
    m = fc._read_manifest(cfg)
    assert len(m) == 12
    assert m.name.is_unique
    assert m.expected_rows.sum() == 30  # 5 sel x 3 hari x 2 sumber
    assert set(m.status) == {'PLANNED'}
    fc.buat_rencana(cfg, **args)
    assert len(fc._read_manifest(cfg)) == 12


def test_reject_overlap(mini_world):
    cfg, _ = mini_world
    fc.buat_rencana(cfg, source='chirps', start='2025-08-01', end='2025-08-04',
                    province='Riau', cells_per_batch=2, days_per_batch=2)
    with pytest.raises(ValueError, match='tumpang tindih'):
        fc.buat_rencana(cfg, source='chirps', start='2025-08-02', end='2025-08-03',
                        province='Riau', cells_per_batch=1, days_per_batch=1)


def test_cleaning_checks_and_preserves_missing(mini_world):
    cfg, root = mini_world
    fc.buat_rencana(cfg, source='chirps', start='2025-08-01', end='2025-08-03',
                    province='Jambi', cells_per_batch=2, days_per_batch=2)
    record = fc._read_manifest(cfg).iloc[0]
    path = root / 'batch.csv'
    d = pd.DataFrame({
        'cell_id': [20,21,20,21],
        'date': ['2025-08-01','2025-08-01','2025-08-02','2025-08-02'],
        'precipitation': [0., 2.3, -9999., 4.2],
    })
    d.to_csv(path, index=False)
    clean = fc._sanitize_csv_batch(path, record, {20,21})
    assert len(clean) == 4
    assert clean.precip_mm.isna().sum() == 1  # tidak ubah -9999 menjadi hujan=0
    d.loc[0, 'precipitation'] = -2
    d.to_csv(path, index=False)
    with pytest.raises(ValueError, match='negatif'):
        fc._sanitize_csv_batch(path, record, {20,21})
