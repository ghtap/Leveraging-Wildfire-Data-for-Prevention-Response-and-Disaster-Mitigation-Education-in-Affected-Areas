"""Utilitas bersama untuk pipeline data (Orang A)."""
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
ID_BASE = 100_000  # cell_id = iy * ID_BASE + ix


def load_config(path=None) -> dict:
    p = Path(path) if path else ROOT / "config.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def rpath(rel) -> Path:
    """Path relatif terhadap root repo (path absolut dibiarkan)."""
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


def norm_name(s) -> str:
    return str(s).replace(" ", "").lower()


def cell_index(lon, lat, d):
    ix = np.floor((np.asarray(lon, float) + 180.0) / d + 1e-9).astype(np.int64)
    iy = np.floor((np.asarray(lat, float) + 90.0) / d + 1e-9).astype(np.int64)
    return ix, iy


def cell_id(lon, lat, d):
    """ID sel dari koordinat, murni aritmetika (tanpa spatial join)."""
    ix, iy = cell_index(lon, lat, d)
    return iy * ID_BASE + ix


def id_to_center(cid, d):
    iy, ix = np.divmod(np.asarray(cid, np.int64), ID_BASE)
    return -180.0 + (ix + 0.5) * d, -90.0 + (iy + 0.5) * d  # lon, lat
