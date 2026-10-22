import numpy as np
import pandas as pd
import pytest

from src.data.build_features import FeatureBuilder, days_since, run_length
from src.data.common import cell_id, id_to_center
from src.data.fetch_climate import derive_era5, parse_dmi, read_oni
from src.data.fetch_firms import aggregate_daily, clean_firms


def test_cell_id_roundtrip():
    lon, lat = np.array([100.23, 116.79, 110.05]), np.array([-4.87, 2.89, -1.95])
    c = cell_id(lon, lat, 0.1)
    clon, clat = id_to_center(c, 0.1)
    assert np.all(np.abs(clon - lon) <= 0.05 + 1e-9) and np.all(np.abs(clat - lat) <= 0.05 + 1e-9)
    assert cell_id(100.2, -1.0, 0.1) == cell_id(100.2000001, -1.0, 0.1)  # batas sel stabil


def test_grid_province_assignment(world):
    g = world["grid"]
    assert len(g) == 50 and set(g.province) == {"Prov A", "Prov B"}
    assert (g[g.province == "Prov A"].lon < 110.5).all()
    assert g.cell_id.is_unique


def test_grid_missing_province_raises(cfg, boundary):
    from src.data.build_grid import build_grid
    cfg["provinces"].append("Prov C")
    with pytest.raises(ValueError):
        build_grid(cfg, boundary)


def _raw(rows):
    cols = ["latitude", "longitude", "acq_date", "acq_time", "confidence", "frp", "type", "source"]
    return pd.DataFrame(rows, columns=cols)


def test_clean_firms_rules(cfg):
    cfg["firms"]["sensors"] = ["VIIRS_SNPP", "MODIS"]
    raw = _raw([
        (-1.0, 110.0, "2020-01-01", 100, "n", 5.0, 0, "VIIRS_SNPP_SP"),    # ok
        (-1.0, 110.0, "2020-01-01", 100, "n", 5.0, 0, "VIIRS_SNPP_NRT"),   # NRT ditimpa SP
        (-1.1, 110.1, "2020-01-02", 100, "n", 5.0, np.nan, "VIIRS_SNPP_NRT"),  # NRT tanpa SP: dipertahankan
        (-1.2, 110.2, "2020-01-01", 100, "l", 5.0, 0, "VIIRS_SNPP_SP"),    # low conf: buang
        (-1.3, 110.3, "2020-01-01", 100, "h", 5.0, 1, "VIIRS_SNPP_SP"),    # type gunung api: buang
        (-1.4, 110.4, "2020-01-01", 100, "20", 5.0, 0, "MODIS_SP"),        # MODIS conf rendah: buang
        (-1.5, 110.4, "2020-01-01", 100, "80", 5.0, 0, "MODIS_SP"),        # ok
        (-1.0, 110.0, "2020-01-01", 100, "n", 5.0, 0, "VIIRS_NOAA20_SP"),  # sensor tak dipilih: buang
    ])
    out = clean_firms(raw, cfg)
    assert sorted(zip(out.sensor, out.is_nrt)) == [("MODIS", False), ("VIIRS_SNPP", False), ("VIIRS_SNPP", True)]


def test_aggregate_daily_drops_outside_grid(world):
    g = world["grid"]
    df = pd.DataFrame({"latitude": [-1.55, -1.55, 10.0], "longitude": [110.05, 110.06, 110.05],
                       "acq_date": pd.to_datetime(["2020-01-05"] * 3), "frp": [1.0, 2.0, 3.0]})
    out = aggregate_daily(df, g, 0.1)
    assert len(out) == 1 and out.n_hotspot.iloc[0] == 2 and out.frp_sum.iloc[0] == 3.0


def test_run_length_and_days_since():
    m = np.array([[1], [1], [0], [1]], bool)
    assert run_length(m).ravel().tolist() == [1, 2, 0, 1]
    ev = np.array([[0], [1], [0], [0]], bool)
    assert days_since(ev, cap=10).ravel().tolist() == [10, 0, 1, 2]
    assert days_since(np.zeros((3, 1), bool), cap=2).ravel().tolist() == [2, 2, 2]


def test_parse_dmi_and_oni(tmp_path):
    txt = "1870 2026\n2025  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2\n2026  0.5 -9999 -9999 -9999 -9999 -9999 -9999 -9999 -9999 -9999 -9999 -9999\n-9999\nCatatan teks\n"
    d = parse_dmi(txt)
    assert len(d) == 13 and d.dmi.iloc[0] == 0.1 and d.month.iloc[-1] == pd.Timestamp("2026-01-01")
    p = tmp_path / "oni.csv"
    p.write_text("SEAS,YR,TOTAL,ANOM\nDJF,2020,27.0,0.5\nNDJ,2020,27.0,-0.3\n")
    o = read_oni(p)
    assert o.set_index("month").oni.to_dict() == {pd.Timestamp("2020-01-01"): 0.5, pd.Timestamp("2020-12-01"): -0.3}


def test_derive_era5_rh():
    df = pd.DataFrame({"cell_id": [1], "date": pd.to_datetime(["2020-01-01"]),
                       "temperature_2m": [300.15], "dewpoint_temperature_2m": [300.15],
                       "volumetric_soil_water_layer_1": [0.3], "u_component_of_wind_10m": [3.0],
                       "v_component_of_wind_10m": [4.0]})
    o = derive_era5(df)
    assert o.rh.iloc[0] == pytest.approx(100, abs=0.01) and o.wind_ms.iloc[0] == pytest.approx(5.0)


def _builder(w, clim=None, hot=None):
    return FeatureBuilder(w["cfg"], w["grid"], hot if hot is not None else w["hot"],
                          clim if clim is not None else w["clim"], w["enso"], w["cfg"]["dates"]["end"])


def test_features_match_bruteforce(world):
    w, fb = world, _builder(world)
    t = np.array([100, 200])
    blk = fb.block(t)
    H = 7
    for cid in w["grid"].cell_id.iloc[[0, 17, 49]]:
        c = w["clim"][w["clim"].cell_id == cid].set_index("date")["precip_mm"]
        h = w["hot"][w["hot"].cell_id == cid].set_index("date")["n_hotspot"].reindex(w["dates"], fill_value=0)
        for ti in t:
            d = w["dates"][ti]
            row = blk[(blk.cell_id == cid) & (blk.date == d)].iloc[0]
            assert row.precip_sum_7d == pytest.approx(c.iloc[ti - 6: ti + 1].sum(), rel=1e-4)
            assert row.precip_sum_30d == pytest.approx(c.iloc[ti - 29: ti + 1].sum(), rel=1e-4)
            assert row.hot_7d == h.iloc[ti - 6: ti + 1].sum()
            nxt = h.iloc[ti + 1: ti + 1 + H].sum()
            assert row.n_hot_next_7d == nxt and row.y_7d == int(nxt >= 1)
            hist = h.iloc[: ti + 1]
            last = hist[hist > 0].index.max()
            exp = 365 if pd.isna(last) else (d - last).days
            assert row.days_since_hot == exp


def test_no_leakage_features_ignore_future(world):
    """Mengacak data setelah hari t tidak boleh mengubah fitur di hari t."""
    w = world
    t0 = 150
    cut = w["dates"][t0]
    rng = np.random.default_rng(5)
    clim2, hot2 = w["clim"].copy(), w["hot"].copy()
    fut = clim2["date"] > cut
    for c in ["precip_mm", "t2m_c", "rh", "sm1", "wind_ms"]:
        clim2.loc[fut, c] = rng.uniform(0, 100, fut.sum()).astype("float32")
    hot2 = pd.concat([hot2[hot2.date <= cut],
                      hot2[hot2.date > cut].assign(n_hotspot=99)])
    a, b = _builder(w).block(np.array([t0])), _builder(w, clim2, hot2).block(np.array([t0]))
    feats = [c for c in a.columns if c not in ("y_7d", "n_hot_next_7d")]
    pd.testing.assert_frame_equal(a[feats], b[feats])
    assert not a["n_hot_next_7d"].equals(b["n_hot_next_7d"])  # label memang bergantung pada masa depan


def test_latency_shifts_climate_only(world):
    w = world
    w["cfg"]["features"]["obs_latency_days"] = 5
    fb5 = _builder(w)
    w["cfg"]["features"]["obs_latency_days"] = 0
    fb0 = _builder(w)
    a, b = fb5.block(np.array([120])), fb0.block(np.array([115]))
    assert np.allclose(a["precip_sum_7d"], b["precip_sum_7d"])      # iklim t-5
    assert np.array_equal(a["hot_7d"], fb0.block(np.array([120]))["hot_7d"])  # hotspot tidak digeser


def test_end_to_end_parquet(world, tmp_path):
    import pyarrow.parquet as pq
    from src.data.make_dataset import make_dataset
    inp = dict(grid=world["grid"], hotspots=world["hot"], climate=world["clim"], enso=world["enso"],
               last_firms_date=world["cfg"]["dates"]["end"], static=None)
    rep = make_dataset(world["cfg"], inp, tmp_path / "dataset_v1.parquet")
    df = pq.read_table(tmp_path / "dataset_v1.parquet").to_pandas()
    assert len(df) == rep["rows"] and df.date.max() == pd.Timestamp("2020-12-24")  # akhir - 7 hari
    assert df.date.min() == pd.Timestamp("2020-01-01") + pd.Timedelta(days=30)     # warm-up = jendela terpanjang
    assert df.groupby(["cell_id", "date"]).size().max() == 1
    assert (tmp_path / "dataset_v1_schema.json").exists()


def test_static_peat_and_distance(world, tmp_path):
    import geopandas as gpd
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import box
    from src.data.fetch_static import dist_to_built, peat_fraction
    g, d = world["grid"], 0.1
    # gambut menutupi separuh bawah Prov A (lat -2.0..-1.75) -> 50% sel di baris itu
    peat = gpd.GeoDataFrame(geometry=[box(110.0, -2.0, 110.5, -1.75)], crs=4326)
    peat.to_file(tmp_path / "peat.gpkg", driver="GPKG")
    pf = peat_fraction(g, tmp_path / "peat.gpkg", d)
    cells_a = g[g.province == "Prov A"]
    assert pf[cells_a[cells_a.lat < -1.8].cell_id].min() == pytest.approx(1.0, abs=1e-3)
    assert pf[cells_a[cells_a.lat > -1.7].cell_id].max() == pytest.approx(0.0, abs=1e-3)
    assert pf[g[g.province == "Prov B"].cell_id].max() == pytest.approx(0.0, abs=1e-3)
    # raster built-up: satu piksel permukiman di (110.95, -1.55)
    arr = np.zeros((100, 200), np.float32)
    arr[int((-1.5 - -1.55) / 0.01), int((110.95 - 110.0) / 0.01)] = 1.0
    with rasterio.open(tmp_path / "b.tif", "w", driver="GTiff", height=100, width=200, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(110.0, -1.5, 0.01, 0.01)) as dst:
        dst.write(arr, 1)
    dist = dist_to_built(g, tmp_path / "b.tif", 0.02)
    near = g[(g.lon > 110.9) & (g.lat > -1.6)].cell_id
    far = g[(g.lon < 110.1) & (g.lat < -1.9)].cell_id
    assert dist[near].max() < 8 and dist[far].min() > 80


def test_vi_availability_and_dw_prior_year(world):
    w = world
    ids = w["grid"].cell_id.to_numpy()
    starts = pd.date_range("2019-12-01", "2020-12-31", freq="16D")           # tanggal awal komposit
    vi = pd.DataFrame([(c, s, float(s.dayofyear + 1000 * (s.year - 2019)), 0.5) for s in starts for c in ids],
                      columns=["cell_id", "date", "NDVI", "EVI"])
    dw = pd.DataFrame([(c, y, float(y)) for y in (2018, 2019) for c in ids], columns=["cell_id", "year", "dw_trees"])
    fb = FeatureBuilder(w["cfg"], w["grid"], w["hot"], w["clim"], w["enso"], w["cfg"]["dates"]["end"], vi=vi, dw=dw)
    avail, ffill = w["cfg"]["features"]["vi_availability_days"], w["cfg"]["features"]["vi_max_ffill_days"]
    for ti in (60, 150, 250):
        d = w["dates"][ti]
        row = fb.block(np.array([ti])).iloc[0]
        ok = [s for s in starts if s + pd.Timedelta(days=avail) <= d]
        exp = float(ok[-1].dayofyear + 1000 * (ok[-1].year - 2019)) if (d - (ok[-1] + pd.Timedelta(days=avail))).days <= ffill else np.nan
        assert row.ndvi_last == pytest.approx(exp)
        assert row.dw_trees == 2019.0                                          # 2020 memakai komposit 2019
    # nilai komposit masa depan tidak boleh mengubah fitur: ubah semua komposit yang baru tersedia setelah t
    t0 = 150; cut = w["dates"][t0]
    vi2 = vi.copy(); late = vi2["date"] + pd.Timedelta(days=avail) > cut; vi2.loc[late, "NDVI"] = -999
    fb2 = FeatureBuilder(w["cfg"], w["grid"], w["hot"], w["clim"], w["enso"], w["cfg"]["dates"]["end"], vi=vi2, dw=dw)
    assert np.array_equal(fb.block(np.array([t0]))["ndvi_last"], fb2.block(np.array([t0]))["ndvi_last"])


def test_peat_raster_gpm2(world, tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    from src.data.fetch_static import peat_fraction_raster
    g = world["grid"]
    arr = np.zeros((50, 100), np.uint8)              # 0.01 deg, lat -1.5..-2.0, lon 110..111
    arr[25:, :50] = 1                                # separuh bawah Prov A: didominasi gambut
    arr[:25, 50:] = 2                                # separuh atas Prov B: mosaik
    with rasterio.open(tmp_path / "gpm2.tif", "w", driver="GTiff", height=50, width=100, count=1, dtype="uint8",
                       crs="EPSG:4326", transform=from_origin(110.0, -1.5, 0.01, 0.01), nodata=255) as dst:
        dst.write(arr, 1)
    df = peat_fraction_raster(g, tmp_path / "gpm2.tif", 0.1)
    a_low = g[(g.lon < 110.5) & (g.lat < -1.8)].cell_id
    b_up = g[(g.lon > 110.5) & (g.lat > -1.7)].cell_id
    assert df.loc[a_low, "peat_dom_frac"].min() == pytest.approx(1.0, abs=0.02)
    assert df.loc[b_up, "peat_frac"].min() == pytest.approx(1.0, abs=0.02) and df.loc[b_up, "peat_dom_frac"].max() < 0.02
    assert df.loc[g[(g.lon < 110.5) & (g.lat > -1.7)].cell_id, "peat_frac"].max() < 0.02
