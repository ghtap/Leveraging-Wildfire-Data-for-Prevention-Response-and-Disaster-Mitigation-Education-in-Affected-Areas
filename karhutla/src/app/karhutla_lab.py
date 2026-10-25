"""Karhutla Lab: bangun dashboard interaktif statis (siap deploy ke Netlify).

    python -m src.app.karhutla_lab verify                      # cocokkan dengan hasil Orang B (stage22)
    python -m src.app.karhutla_lab build [--demo] [--days 30]  # tulis dist/
    python -m src.app.karhutla_lab serve [--port 8000]         # pratinjau lokal

Situs murni statis (HTML/JS + JSON prediksi): tidak butuh server Python di Netlify.
Model/data berubah -> jalankan `build` lagi lalu deploy ulang folder dist/.
"""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.common import load_config, rpath
from .predict_api import DemoModel, PredictAPI, app_cfg
from .risk_levels import level_index

WEB = Path(__file__).parent / "web"
HEADER_FEATURES = {  # kunci: (label, satuan)
    "rain_prev7_mm": ("Hujan 7 hari sebelumnya", "mm"),
    "rain_prev14_mm": ("Hujan 14 hari sebelumnya", "mm"),
    "hotspot_prev7": ("Hotspot di sel, 7 hari sebelumnya", "titik"),
    "hotspot_prev14": ("Hotspot di sel, 14 hari sebelumnya", "titik"),
    "frp_prev7": ("FRP total 7 hari sebelumnya", "MW"),
    "temp_prev7_mean": ("Suhu rata-rata 7 hari", "°C"),
    "rh_prev7_mean": ("Kelembapan udara 7 hari", "%"),
    "soil_prev7_mean": ("Kelembapan tanah 7 hari", "m³/m³"),
    "wind_prev7_mean": ("Angin rata-rata 7 hari", "m/s"),
    "month_2026": ("Bulan", ""),
}
# jangkar slider -> fitur yang ikut bergeser (jendela bertumpuk harus konsisten)
GROUPS = {
    "rain_prev14_mm": ["rain_prev14_mm", "rain_prev7_mm"],
    "hotspot_prev14": ["hotspot_prev14", "hotspot_prev7", "frp_prev7"],
    "temp_prev7_mean": ["temp_prev7_mean"], "rh_prev7_mean": ["rh_prev7_mean"],
    "soil_prev7_mean": ["soil_prev7_mean"], "wind_prev7_mean": ["wind_prev7_mean"],
}
SLIDER_LABEL = {"rain_prev14_mm": "Hujan 14 hari (7 hari ikut bergeser)",
                "hotspot_prev14": "Hotspot 14 hari (7 hari dan FRP ikut)"}
FIG_TITLE = {"stage19_pr_curve_2026.png": "Kurva precision-recall (uji September 2026)",
             "stage19_confusion_matrix_2026.png": "Confusion matrix (uji September 2026)",
             "stage20_shap_global_2026.png": "SHAP global", "stage20_feature_importance_2026.png": "Kepentingan fitur"}
SPLIT_ID = (("train", "pelatihan (in-sample)"), ("val", "validasi"), ("test", "uji"))


def _num(x, nd=4):
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


def _clean(a, nd=2):
    a = np.asarray(a, float).round(nd)
    return [None if not np.isfinite(x) else (int(x) if float(x).is_integer() else float(x)) for x in a]


def _split_label(s):
    s = str(s).lower()
    return next((lab for key, lab in SPLIT_ID if key in s), str(s))


# ---------------------------------------------------------------- demo
def make_demo(cfg: dict, n_days: int, seed: int = 0):
    """Data sintetis (BUKAN prediksi nyata) agar tampilan bisa dicoba tanpa model Orang B."""
    rng = np.random.default_rng(seed)
    boxes = {"Riau": (100.0, 103.8, -1.1, 2.9), "Jambi": (101.0, 104.4, -2.6, -1.1),
             "Sumatera Selatan": (102.2, 106.2, -4.9, -2.6), "Kalimantan Barat": (108.7, 114.2, -3.0, 2.1),
             "Kalimantan Tengah": (110.7, 115.8, -3.6, -0.5), "Kalimantan Selatan": (114.3, 116.6, -4.3, -1.8)}
    d, rows = 0.25, []
    for prov, (x0, x1, y0, y1) in boxes.items():
        for x in np.arange(x0 + d / 2, x1, d):
            for y in np.arange(y0 + d / 2, y1, d):
                rows.append((prov, round(x, 3), round(y, 3)))
    grid = pd.DataFrame(rows, columns=["province", "lon", "lat"]).drop_duplicates(["lon", "lat"]).reset_index(drop=True)
    grid.insert(0, "cell_id", np.arange(len(grid), dtype=np.int64) + 1)
    n = len(grid)
    s = np.sin(grid.lon * 1.3) + np.cos(grid.lat * 1.7)
    s = ((s - s.mean()) / s.std()).to_numpy()
    end = pd.Timestamp(cfg["dates"]["end"]) - pd.Timedelta(days=7)
    dates = pd.date_range(end=end, periods=n_days)
    model, frames = DemoModel(), []
    for i, dt in enumerate(dates):
        r = 0.25 + 0.7 * i / max(n_days - 1, 1)
        f = pd.DataFrame({"date": dt, "cell_id": grid.cell_id, "province": grid.province, "lon": grid.lon, "lat": grid.lat})
        f["rain_prev14_mm"] = np.clip(120 * (1 - 0.85 * r - 0.15 * s) + rng.normal(0, 8, n), 0, None)
        f["rain_prev7_mm"] = f["rain_prev14_mm"] * np.clip(rng.normal(0.5, 0.1, n), 0.1, 1)
        f["hotspot_prev14"] = rng.poisson(5 * r * np.exp(0.4 * s)).astype(float)
        f["hotspot_prev7"] = np.minimum(f["hotspot_prev14"], rng.poisson(2.5 * r * np.exp(0.4 * s)))
        f["frp_prev7"] = f["hotspot_prev7"] * rng.uniform(5, 25, n)
        f["temp_prev7_mean"] = 26 + 3 * r + 0.5 * s + rng.normal(0, 0.4, n)
        f["rh_prev7_mean"] = np.clip(88 - 25 * r - 3 * s + rng.normal(0, 2, n), 30, 100)
        f["soil_prev7_mean"] = np.clip(0.46 - 0.2 * r - 0.02 * s + rng.normal(0, 0.01, n), 0.05, 0.6)
        f["wind_prev7_mean"] = np.clip(2 + rng.normal(0, 0.6, n), 0, None)
        f["month_2026"] = float(dt.month)
        f["prob"] = model.predict_proba(f)[:, 1]
        f["y_7d"] = (rng.random(n) < f["prob"]).astype(int)
        f["split"] = "validation" if i < n_days // 2 else "test"
        frames.append(f)
    spec = dict(model="DEMO", status="DEMO", threshold_f1=0.13, threshold_recall70=0.055,
                important_limitations=["Data sintetis, bukan prediksi nyata"])
    return pd.concat(frames, ignore_index=True), PredictAPI(model, list(DemoModel.feature_names_in_), cfg, spec), d


# ---------------------------------------------------------------- tulis situs
def write_site(df: pd.DataFrame, api: PredictAPI, out: Path, cell_deg: float, demo: bool, cfg: dict):
    levels, a, spec = api.levels, app_cfg(cfg), api.spec
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(WEB, out)
    for sub in ("d", "f", "figures"):
        (out / "data" / sub).mkdir(parents=True)

    df = df.sort_values(["date", "cell_id"]).reset_index(drop=True)
    dates = sorted(df["date"].unique())
    first = df[df["date"] == dates[0]].set_index("cell_id")
    ids = first.index.to_numpy()
    provs = sorted(first["province"].dropna().unique())
    pidx = first["province"].map({p: i for i, p in enumerate(provs)}).fillna(-1).astype(int).to_numpy()
    feats = [k for k in HEADER_FEATURES if k in df.columns]
    dump = lambda p, o: (out / p).write_text(json.dumps(o, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    dump("data/grid.json", dict(n=len(ids), cell_deg=cell_deg, ids=[int(i) for i in ids], lon=_clean(first["lon"], 3),
                               lat=_clean(first["lat"], 3), prov=[int(i) for i in pidx], provinces=provs))

    has_y = "y_7d" in df.columns and df["y_7d"].notna().any()
    has_split = "split" in df.columns
    ts = {k: dict(mean=[], high=[], obs=[]) for k in ["ALL"] + provs}
    hi = 2 if len(levels) > 2 else len(levels) - 1
    day_split, pool = [], []
    for dt in dates:
        g = df[df["date"] == dt].set_index("cell_id").reindex(ids)
        p = g["prob"].to_numpy(float)
        y = g["y_7d"].to_numpy(float) if has_y else np.full(len(ids), np.nan)
        y_ok = bool(np.isfinite(y).any())
        name = pd.Timestamp(dt).strftime("%Y-%m-%d")
        dump(f"data/d/{name}.json", dict(p=[None if not np.isfinite(v) else int(round(v * 1000)) for v in p],
                                         y=[int(i) for i in np.flatnonzero(y == 1)] if y_ok else None))
        dump(f"data/f/{name}.json", {k: _clean(g[k]) for k in feats})
        lv = level_index(p, levels)
        for key in ts:
            m = (np.ones(len(ids), bool) if key == "ALL" else (pidx == provs.index(key))) & np.isfinite(p)
            ts[key]["mean"].append(_num(p[m].mean()) if m.any() else None)
            ts[key]["high"].append(_num(np.mean(lv[m] >= hi)) if m.any() else None)
            mo = m & np.isfinite(y)
            ts[key]["obs"].append(_num(y[mo].mean()) if mo.any() else None)
        sp = g["split"].dropna().mode().iloc[0] if has_split and g["split"].notna().any() else None
        day_split.append(_split_label(sp) if sp is not None else None)
        if y_ok and not (sp is not None and "train" in str(sp).lower()):
            pool.append((lv, y))  # statistik per level hanya dari data di luar pelatihan
    if pool:
        lv_all, y_all = np.concatenate([a_ for a_, _ in pool]), np.concatenate([b_ for _, b_ in pool])
        okm = np.isfinite(y_all) & (lv_all >= 0)
        level_stats = [dict(name=l["name"], n=int((okm & (lv_all == k)).sum()),
                            hit=_num(y_all[okm & (lv_all == k)].mean()) if (okm & (lv_all == k)).any() else None)
                       for k, l in enumerate(levels)]
        base_rate = _num(y_all[okm].mean())
    else:
        level_stats, base_rate = None, None

    sample = df[df["date"].isin(dates[-min(len(dates), 10):])]
    anchors = [k for k, mem in GROUPS.items() if all(m in api.feature_cols and m in df.columns for m in mem)]
    curves = api.response_curves(sample, anchors, {k: GROUPS[k] for k in anchors})
    figs = []
    for fn in a["figures"]:
        src = rpath(a["figures_dir"]) / fn
        if not demo and src.exists():
            shutil.copy(src, out / "data" / "figures" / fn)
            figs.append(dict(file=f"data/figures/{fn}", title=FIG_TITLE.get(fn, fn)))
    meta = dict(
        demo=demo, horizon=cfg["label"]["horizon_days"], dates=[pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates],
        splits=day_split, levels=levels, ts=ts, curves=curves, has_obs=bool(has_y), level_stats=level_stats,
        base_rate=base_rate, figures=figs,
        model_info=dict(model=spec.get("model"), status=spec.get("status"), threshold_f1=spec.get("threshold_f1"),
                        threshold_recall70=spec.get("threshold_recall70"), validation_pr_auc=spec.get("validation_pr_auc"),
                        limitations=spec.get("important_limitations", [])),
        features=[dict(key=k, label=SLIDER_LABEL.get(k, HEADER_FEATURES[k][0]), unit=HEADER_FEATURES[k][1],
                       lo=float(np.nanpercentile(df[k], 1)), hi=float(np.nanpercentile(df[k], 99)), whatif=k in curves)
                  for k in feats],
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        label_def=f"Skor model untuk adanya ≥{cfg['label']['min_hotspots']} hotspot di sel dalam {cfg['label']['horizon_days']} hari setelah tanggal prakiraan. Skor belum dikalibrasi, jadi bukan peluang sebenarnya.")
    dump("data/dates.json", meta)

    mp = rpath(a["metrics_path"])
    if not demo and mp.exists():
        shutil.copy(mp, out / "data" / "metrics.json")
    (out / "_headers").write_text("/\n  Cache-Control: no-cache\n/data/*\n  Cache-Control: public, max-age=3600\n"
                                  "/assets/*\n  Cache-Control: public, max-age=3600\n", encoding="utf-8")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"dist siap: {out} ({len(dates)} tanggal, {len(ids)} sel, {size:.1f} MB)")


def build(cfg, out=None, demo=False, days=None):
    a = app_cfg(cfg)
    days = days or a["n_days"]
    out = rpath(out or a["dist_dir"])
    if demo:
        df, api, deg = make_demo(cfg, days)
    else:
        api = PredictAPI.from_files(cfg)
        df, deg = api.predict_recent(days), cfg["grid"]["cell_deg"]
    write_site(df, api, out, deg, demo, cfg)
    return out


def verify(cfg) -> bool:
    """Cocokkan prediksi dashboard pada tanggal peta Orang B dengan stage22_visual_summary."""
    a = app_cfg(cfg)
    api = PredictAPI.from_files(cfg)
    s = json.loads(rpath(a["stage22_summary"]).read_text(encoding="utf-8"))
    df = api.predict_date(s["map_date"], require_complete=False)
    n_alert, mean = int((df["prob"] >= s["threshold_f1"]).sum()), float(df["prob"].mean())
    ok = (len(df) == s["n_grids"] and n_alert == s["n_alerts"] and abs(mean - s["average_probability"]) < 1e-4)
    print(f"tanggal {s['map_date']}: sel {len(df)} (B: {s['n_grids']}), alert {n_alert} (B: {s['n_alerts']}), "
          f"rata-rata skor {mean:.6f} (B: {s['average_probability']:.6f})")
    print("COCOK dengan hasil Orang B" if ok else "TIDAK COCOK: periksa dataset/model/urutan fitur sebelum dipublikasikan")
    return ok


def serve(path, port=8000):
    import functools
    import http.server
    h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(path))
    print(f"buka http://localhost:{port}  (Ctrl+C untuk berhenti)")
    http.server.ThreadingHTTPServer(("", port), h).serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "serve", "verify"])
    ap.add_argument("--demo", action="store_true", help="data sintetis, hanya untuk uji tampilan")
    ap.add_argument("--days", type=int)
    ap.add_argument("--out")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    cfg = load_config()
    if a.cmd == "build":
        build(cfg, a.out, a.demo, a.days)
    elif a.cmd == "verify":
        raise SystemExit(0 if verify(cfg) else 1)
    else:
        serve(rpath(a.out or app_cfg(cfg)["dist_dir"]), a.port)
