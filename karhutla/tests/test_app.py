import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hashlib
import json
import os
import shutil
import subprocess
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.app.karhutla_lab import build, verify
from src.app.predict_api import FEATURES_B, PredictAPI
from src.app.risk_levels import level_index, level_name, levels_from_spec, load_levels, summarize
from src.data.common import ROOT, load_config

SPEC = dict(model="LightGBM regularized", status="RESEARCH_CANDIDATE", threshold_f1=0.13, threshold_recall70=0.055,
            validation_pr_auc=0.54, important_limitations=["Test September pernah terekspos"])


def _data(rng, cells=60, days=16):
    dates = pd.date_range("2026-08-20", periods=days)
    n = cells * days
    df = pd.DataFrame({"cell_id": np.tile(np.arange(cells) + 1000, days).astype("int64"), "date": np.repeat(dates.values, cells)})
    df["rain_prev14_mm"] = rng.gamma(2, 20, n); df["rain_prev7_mm"] = df.rain_prev14_mm * rng.uniform(0.2, 0.8, n)
    df["hotspot_prev14"] = rng.poisson(3, n).astype(float); df["hotspot_prev7"] = np.minimum(df.hotspot_prev14, rng.poisson(1.5, n))
    df["frp_prev7"] = df.hotspot_prev7 * rng.uniform(3, 20, n)
    df["temp_prev7_mean"] = rng.normal(27, 1.5, n); df["rh_prev7_mean"] = rng.uniform(50, 95, n)
    df["soil_prev7_mean"] = rng.uniform(0.2, 0.5, n); df["wind_prev7_mean"] = rng.uniform(0, 5, n)
    df["month_2026"] = pd.DatetimeIndex(df.date).month.astype(float)
    z = -1.5 - 0.03 * df.rain_prev14_mm + 0.5 * df.hotspot_prev14 + 0.05 * df.frp_prev7 - 3 * (df.soil_prev7_mean - 0.35)
    df["y_7d"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    df["split"] = np.where(df.date < "2026-08-25", "train", np.where(df.date < "2026-09-01", "validation", "test"))
    return df


@pytest.fixture(params=["date32", "timestamp"])
def env(tmp_path, request):
    rng = np.random.default_rng(0)
    df = _data(rng)
    df.loc[df.index[-5:], "rain_prev7_mm"] = np.nan                       # fitur kosong di akhir
    t = pa.Table.from_pandas(df, preserve_index=False)
    if request.param == "date32":
        t = t.set_column(t.schema.get_field_index("date"), "date", t.column("date").cast(pa.date32()))
    pq.write_table(t, tmp_path / "dataset.parquet")
    model = lgb.LGBMClassifier(n_estimators=60, num_leaves=15, verbose=-1, random_state=0).fit(df[FEATURES_B], df["y_7d"])
    joblib.dump(model, tmp_path / "model.joblib")
    spec = {**SPEC, "feature_order": FEATURES_B, "sha256": hashlib.sha256((tmp_path / "model.joblib").read_bytes()).hexdigest()}
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    cells = sorted(df.cell_id.unique())
    pd.DataFrame({"cell_id": cells, "lon": 100 + np.arange(len(cells)) * 0.045, "lat": -1.0,
                  "province": np.where(np.arange(len(cells)) < 30, "Riau", "Jambi")}).to_csv(tmp_path / "grid.csv", index=False)
    (tmp_path / "metrics.json").write_text(json.dumps({"pr_auc": 0.55, "threshold_results": [{"precision": 0.44}, {"precision": 0.27}]}))
    (tmp_path / "figs").mkdir(); (tmp_path / "figs" / "stage19_pr_curve_2026.png").write_bytes(b"\x89PNG")
    cfg = load_config()
    cfg["app"] = dict(model_path=str(tmp_path / "model.joblib"), model_meta_path=str(tmp_path / "spec.json"),
                      dataset_path=str(tmp_path / "dataset.parquet"), grid_path=str(tmp_path / "grid.csv"),
                      metrics_path=str(tmp_path / "metrics.json"), figures_dir=str(tmp_path / "figs"),
                      stage22_summary=str(tmp_path / "s22.json"), n_days=6)
    return cfg, df, tmp_path, model


def test_levels_from_spec_and_validation():
    lv = levels_from_spec(SPEC)
    assert [l["min"] for l in lv] == [0.0, 0.055, 0.13, 0.4]
    p = np.array([0, 0.054, 0.055, 0.129, 0.13, 0.399, 0.4, np.nan])
    assert level_index(p, lv).tolist() == [0, 0, 1, 1, 2, 2, 3, -1]
    assert level_name(np.nan, lv) == "Tidak ada data"
    with pytest.raises(ValueError):
        levels_from_spec({**SPEC, "threshold_f1": 0.5})                    # f1 harus < very_high
    custom = [{"name": "A", "min": 0, "color": "#000"}, {"name": "B", "min": 0.3, "color": "#fff"}]
    assert [l["name"] for l in load_levels({"app": {"risk_levels": custom}}, SPEC)] == ["A", "B"]
    with pytest.raises(ValueError):
        load_levels({"app": {"risk_levels": [{"name": "A", "min": 0.1, "color": "#000"}]}})


def test_summarize_counts():
    s = summarize(pd.DataFrame({"province": ["A", "A", "B"], "prob": [0.01, 0.5, 0.07]}), levels_from_spec(SPEC))
    assert s.loc["A", "Rendah"] == 1 and s.loc["A", "Sangat Tinggi"] == 1 and s.loc["B", "Sedang"] == 1


def test_predict_api_loads_spec_and_merges_grid(env):
    cfg, df, tmp, model = env
    api = PredictAPI.from_files(cfg)
    assert api.feature_cols == FEATURES_B and [l["min"] for l in api.levels][1:3] == [0.055, 0.13]
    out = api.predict_date("2026-08-30")
    assert len(out) == 60 and {"lon", "lat", "province"} <= set(out) and out["province"].notna().all()
    assert out["prob"].between(0, 1).all() and np.allclose(out["prob"], model.predict_proba(df[df.date == "2026-08-30"][FEATURES_B])[:, 1], atol=1e-5)
    rec = api.predict_recent(3)
    assert rec["date"].nunique() == 3 and rec["date"].max() == pd.Timestamp(df["date"].max())
    assert rec["prob"].isna().sum() == 5                                      # baris dengan fitur kosong -> tanpa skor
    assert api.predict_frame(rec, require_complete=False).size == len(rec) and not np.isnan(api.predict_frame(rec, require_complete=False)).any()
    with pytest.raises(KeyError):
        api.predict_frame(df.drop(columns=["frp_prev7"]))


def test_sha_mismatch_warns(env):
    cfg, _, tmp, _ = env
    (tmp / "spec.json").write_text(json.dumps({**SPEC, "feature_order": FEATURES_B, "sha256": "0" * 64}))
    with pytest.warns(UserWarning, match="sha256"):
        PredictAPI.from_files(cfg)


def test_group_curves_move_nested_windows_together(env):
    cfg, df, _, _ = env
    api = PredictAPI.from_files(cfg)
    anchors = ["rain_prev14_mm", "hotspot_prev14"]
    cur = api.response_curves(df, anchors, {"rain_prev14_mm": ["rain_prev14_mm", "rain_prev7_mm"],
                                            "hotspot_prev14": ["hotspot_prev14", "hotspot_prev7", "frp_prev7"]})
    assert cur["rain_prev14_mm"]["members"] == ["rain_prev14_mm", "rain_prev7_mm"]
    assert cur["rain_prev14_mm"]["dlogit"][0] > cur["rain_prev14_mm"]["dlogit"][-1]      # hujan turun -> skor naik
    assert cur["hotspot_prev14"]["dlogit"][-1] > cur["hotspot_prev14"]["dlogit"][0]      # hotspot naik -> skor naik
    assert cur["rain_prev14_mm"]["dlogit"][cur["rain_prev14_mm"]["z"].index(0)] == 0


def _strict_json(path):
    def bad(c): raise ValueError(f"nilai tidak valid di JSON: {c}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=bad)


def test_build_real_site(env):
    cfg, df, tmp, _ = env
    out = build(cfg, tmp / "dist", demo=False, days=6)
    meta = _strict_json(out / "data" / "dates.json")
    assert meta["demo"] is False and len(meta["dates"]) == 6 and meta["model_info"]["status"] == "RESEARCH_CANDIDATE"
    assert "rain_prev14_mm" in meta["curves"] and meta["curves"]["rain_prev14_mm"]["members"][1] == "rain_prev7_mm"
    assert set(meta["splits"]) <= {"pelatihan (in-sample)", "validasi", "uji"} and "uji" in meta["splits"]
    assert [f["file"] for f in meta["figures"]] == ["data/figures/stage19_pr_curve_2026.png"] and (out / meta["figures"][0]["file"]).exists()
    day = _strict_json(out / "data" / "d" / f"{meta['dates'][-1]}.json")
    assert len(day["p"]) == 60 and any(v is None for v in day["p"]) and all(0 <= v <= 1000 for v in day["p"] if v is not None)
    feat = _strict_json(out / "data" / "f" / f"{meta['dates'][-1]}.json")
    assert set(feat) == set(FEATURES_B) and None in feat["rain_prev7_mm"]
    # statistik per level hanya dari data non-pelatihan, dan hanya baris bersk or ber-label
    api = PredictAPI.from_files(cfg)
    rec = api.predict_recent(6)
    ok = rec[(~rec.split.str.contains("train")) & rec.prob.notna() & rec.y_7d.notna()]
    assert sum(s["n"] for s in meta["level_stats"]) == len(ok)
    assert meta["base_rate"] == pytest.approx(ok.y_7d.mean(), abs=1e-4)
    assert (out / "data" / "metrics.json").exists() and (out / "_headers").exists() and (out / "vendor/leaflet/leaflet.js").exists()


def test_verify_against_stage22(env):
    cfg, df, tmp, model = env
    api = PredictAPI.from_files(cfg)
    last = pd.Timestamp(df["date"].max()).strftime("%Y-%m-%d")
    d = api.predict_date(last, require_complete=False)
    summ = dict(map_date=last, n_grids=len(d), threshold_f1=0.13, n_alerts=int((d.prob >= 0.13).sum()), average_probability=float(d.prob.mean()))
    (tmp / "s22.json").write_text(json.dumps(summ))
    assert verify(cfg) is True
    (tmp / "s22.json").write_text(json.dumps({**summ, "n_alerts": summ["n_alerts"] + 1}))
    assert verify(cfg) is False


def test_build_demo(env):
    cfg, _, tmp, _ = env
    out = build(cfg, tmp / "dist_demo", demo=True, days=6)
    meta = _strict_json(out / "data" / "dates.json")
    assert meta["demo"] is True and meta["model_info"]["status"] == "DEMO" and meta["figures"] == []
    assert not (out / "data" / "metrics.json").exists() and max(p.stat().st_size for p in out.rglob("*") if p.is_file()) < 5e6


@pytest.mark.skipif(shutil.which("node") is None, reason="node tidak ada")
def test_js_logic():
    r = subprocess.run(["node", str(ROOT / "tests/web/logic_test.js")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node tidak ada")
@pytest.mark.parametrize("demo", [True, False])
def test_dashboard_smoke_jsdom(env, demo):
    cfg, _, tmp, _ = env
    out = build(cfg, tmp / "dist_smoke", demo=demo, days=5)
    r = subprocess.run(["node", str(ROOT / "tests/web/smoke.js"), str(out)], capture_output=True, text=True,
                       env={**os.environ, "NODE_PATH": os.environ.get("NODE_PATH", "") or str(ROOT / "node_modules")})
    if "Cannot find module 'jsdom'" in r.stderr:
        pytest.skip("jsdom belum dipasang (npm i jsdom)")
    assert r.returncode == 0, r.stdout + r.stderr
