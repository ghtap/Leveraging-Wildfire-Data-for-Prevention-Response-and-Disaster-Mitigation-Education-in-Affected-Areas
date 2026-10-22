import copy
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.common import cell_id, load_config  # noqa: E402


@pytest.fixture
def cfg():
    c = load_config()
    c["provinces"] = ["Prov A", "Prov B"]
    c["dates"] = {
        "start": "2020-01-01",
        "end": "2020-12-31"
    }

    # Grid 0.1 khusus data pengujian sintetis
    c["grid"]["cell_deg"] = 0.1

    c["features"]["precip_windows"] = [7, 30]
    c["features"]["hotspot_windows"] = [7, 30]
    return copy.deepcopy(c)


@pytest.fixture
def boundary():
    return gpd.GeoDataFrame({"ADM1_NAME": ["ProvA", "ProvB"]},
                            geometry=[box(110.0, -2.0, 110.5, -1.5), box(110.5, -2.0, 111.0, -1.5)], crs=4326)


@pytest.fixture
def world(cfg, boundary):
    """Grid 10x5 + iklim + hotspot sintetis acak (seed tetap)."""
    from src.data.build_grid import build_grid
    rng = np.random.default_rng(0)
    grid = build_grid(cfg, boundary)
    dates = pd.date_range(cfg["dates"]["start"], cfg["dates"]["end"])
    n = len(grid)
    idx = pd.MultiIndex.from_product([grid.cell_id, dates], names=["cell_id", "date"])
    clim = pd.DataFrame(index=idx).reset_index()
    m = len(clim)
    clim["precip_mm"] = rng.gamma(0.5, 8, m).astype("float32")
    clim["t2m_c"] = rng.normal(27, 2, m).astype("float32")
    clim["rh"] = rng.uniform(50, 95, m).astype("float32")
    clim["sm1"] = rng.uniform(0.2, 0.5, m).astype("float32")
    clim["wind_ms"] = rng.uniform(0, 6, m).astype("float32")
    hot = clim[["cell_id", "date"]].sample(frac=0.03, random_state=1).copy()
    hot["n_hotspot"] = rng.integers(1, 4, len(hot)).astype("int32")
    hot["frp_sum"] = rng.uniform(1, 50, len(hot)).astype("float32")
    enso = pd.DataFrame({"month": pd.date_range("2019-01-01", "2020-12-01", freq="MS")})
    enso["oni"] = np.linspace(-1, 1, len(enso)); enso["dmi"] = np.linspace(0.5, -0.5, len(enso))
    return dict(cfg=cfg, grid=grid, clim=clim, hot=hot, enso=enso, dates=dates)
