"""Fungsi prediksi untuk dashboard, disesuaikan dengan artefak Orang B (LightGBM 2026).

Sumber kebenaran: models_final/model_spec_2026.json (urutan fitur, ambang, sha256) dan
models_final/lightgbm_candidate_2026.joblib. Dataset: data/processed/dataset_fitur_2026_v3.parquet.
"""
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.common import load_config, rpath
from .risk_levels import level_name, load_levels

APP_DEFAULTS = dict(
    model_path="models_final/lightgbm_candidate_2026.joblib",
    model_meta_path="models_final/model_spec_2026.json",
    metrics_path="reports/orang_b/stage19_evaluation_2026.json",
    stage22_summary="reports/orang_b/stage22_visual_summary_2026.json",
    dataset_path="data/processed/dataset_fitur_2026_v3.parquet",
    grid_path="data/interim/grid.csv",
    figures_dir="figures/orang_b",
    figures=["stage19_pr_curve_2026.png", "stage19_confusion_matrix_2026.png",
             "stage20_shap_global_2026.png", "stage20_feature_importance_2026.png"],
    dist_dir="dist", n_days=30, require_complete_features=True, very_high_threshold=0.40)

FEATURES_B = ["rain_prev7_mm", "rain_prev14_mm", "hotspot_prev7", "hotspot_prev14", "frp_prev7",
              "temp_prev7_mean", "rh_prev7_mean", "soil_prev7_mean", "wind_prev7_mean", "month_2026"]


def app_cfg(cfg: dict | None = None) -> dict:
    return {**APP_DEFAULTS, **((cfg or {}).get("app") or {})}


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def _pick(names, options):
    low = {str(n).lower().strip(): n for n in names}
    for o in options:
        if o in low:
            return low[o]
    return None


def read_grid(path) -> pd.DataFrame:
    """grid.csv -> cell_id, lon, lat, province (nama kolom dicocokkan longgar)."""
    g = pd.read_csv(path)
    m = {"cell_id": _pick(g.columns, ["cell_id", "id"]), "lon": _pick(g.columns, ["lon", "longitude", "x"]),
         "lat": _pick(g.columns, ["lat", "latitude", "y"]),
         "province": _pick(g.columns, ["province", "provinsi", "adm1_name"])}
    miss = [k for k, v in m.items() if v is None]
    if miss:
        raise KeyError(f"grid.csv tidak punya kolom untuk {miss}; kolom ada: {list(g.columns)}")
    return g[list(m.values())].set_axis(list(m), axis=1)


class DemoModel:
    """Model sintetis HANYA untuk demo tampilan; bukan hasil pelatihan apa pun."""
    feature_names_in_ = np.array(FEATURES_B)

    def predict_proba(self, X):
        z = (-3.2 - 0.012 * X["rain_prev14_mm"] + 0.15 * X["hotspot_prev14"] + 0.02 * X["frp_prev7"]
             + 0.10 * (X["temp_prev7_mean"] - 27) - 0.05 * (X["rh_prev7_mean"] - 80)
             - 4.0 * (X["soil_prev7_mean"] - 0.35) + 0.05 * (X["wind_prev7_mean"] - 2))
        p = 1 / (1 + np.exp(-np.asarray(z, float)))
        return np.c_[1 - p, p]


class PredictAPI:
    def __init__(self, model, feature_cols, cfg=None, spec=None):
        self.cfg, self.model, self.feature_cols = cfg or {}, model, list(feature_cols)
        self.spec = spec or {}
        self.a = app_cfg(cfg)
        self.levels = load_levels(cfg, spec)

    # ------------------------------------------------------------ muat
    @classmethod
    def from_files(cls, cfg=None):
        import joblib
        cfg = cfg or load_config()
        a = app_cfg(cfg)
        mp = rpath(a["model_path"])
        if not mp.exists():
            raise FileNotFoundError(f"model tidak ditemukan: {mp} (atau jalankan dengan --demo)")
        sp = rpath(a["model_meta_path"])
        spec = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
        if spec.get("sha256"):
            h = hashlib.sha256(mp.read_bytes()).hexdigest()
            if h != spec["sha256"]:
                warnings.warn("sha256 model berbeda dari model_spec; pastikan ini model yang benar", stacklevel=2)
        model = joblib.load(mp)
        cols = spec.get("feature_order")
        if not cols and hasattr(getattr(model, "booster_", None), "feature_name"):
            cols = list(model.booster_.feature_name())
        cols = cols or getattr(model, "feature_names_in_", None)
        if cols is None:
            raise ValueError("daftar fitur model tidak diketahui; perlu feature_order di model_spec")
        return cls(model, list(cols), cfg, spec)

    # ------------------------------------------------------------ prediksi
    def predict_frame(self, df: pd.DataFrame, require_complete: bool | None = None) -> np.ndarray:
        """Skor model per baris. Baris dengan fitur NaN -> NaN bila require_complete (bawaan dari config)."""
        miss = [c for c in self.feature_cols if c not in df]
        if miss:
            raise KeyError(f"kolom fitur hilang: {miss[:5]}{'...' if len(miss) > 5 else ''}")
        X = df[self.feature_cols].astype("float32")
        p = np.asarray(self.model.predict_proba(X))
        p = (p[:, 1] if p.ndim == 2 else p).astype(np.float32)
        rc = self.a["require_complete_features"] if require_complete is None else require_complete
        if rc:
            p[~np.isfinite(X.to_numpy()).all(axis=1)] = np.nan
        return p

    def read_dataset(self, dates=None, last_n: int | None = None, path=None) -> pd.DataFrame:
        """Baca hanya tanggal tertentu / n tanggal terakhir; lon/lat/province dari grid.csv bila tidak ada."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        path = rpath(path or self.a["dataset_path"])
        pf = pq.ParquetFile(path)
        names = set(pf.schema_arrow.names)
        dtype = pf.schema_arrow.field("date").type
        all_dates = np.sort(pd.to_datetime(
            pq.read_table(path, columns=["date"]).column("date").to_pandas(date_as_object=False).unique()))
        sel = pd.DatetimeIndex(pd.to_datetime(dates) if dates is not None else all_dates[-(last_n or self.a["n_days"]):])
        want = ["date", "cell_id", "province", "lon", "lat", "split", "y_7d"] + self.feature_cols
        cols = [c for c in dict.fromkeys(want) if c in names]
        lo = sel.min().date() if pa.types.is_date(dtype) else sel.min()
        df = pq.read_table(path, columns=cols, filters=[("date", ">=", lo)]).to_pandas(date_as_object=False)
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"].isin(sel)].reset_index(drop=True)
        need = [c for c in ("lon", "lat", "province") if c not in df]
        if need:
            g = read_grid(rpath(self.a["grid_path"]))[["cell_id"] + need]
            df = df.merge(g, on="cell_id", how="left")
        return df

    def predict_recent(self, n_days: int | None = None) -> pd.DataFrame:
        df = self.read_dataset(last_n=n_days)
        df["prob"] = self.predict_frame(df)
        return df

    def predict_date(self, date, cells=None, require_complete: bool | None = None) -> pd.DataFrame:
        df = self.read_dataset(dates=[pd.Timestamp(date)])
        if cells is not None:
            df = df[df["cell_id"].isin(cells)]
        df["prob"] = self.predict_frame(df, require_complete)
        df["level"] = level_name(df["prob"], self.levels)
        return df.reset_index(drop=True)

    # ------------------------------------------------------------ what-if
    def what_if(self, row: pd.DataFrame | pd.Series, overrides: dict) -> float:
        r = row.to_frame().T if isinstance(row, pd.Series) else row.copy()
        for k, v in overrides.items():
            r[k] = v
        return float(self.predict_frame(r.astype({c: "float64" for c in self.feature_cols if c in r}))[0])

    def response_curves(self, X: pd.DataFrame, anchors, groups: dict | None = None,
                        deltas=(-2, -1, -0.5, 0, 0.5, 1, 2), max_rows: int = 3000, seed: int = 0) -> dict:
        """Rata-rata perubahan logit skor bila fitur jangkar digeser d simpangan baku (ceteris paribus).
        groups[anchor] = fitur yang ikut bergeser sebesar d simpangan bakunya sendiri (mis. jendela 7 hari
        ikut jendela 14 hari) agar kombinasi nilai tetap masuk akal."""
        groups = groups or {}
        Xc = X.dropna(subset=self.feature_cols)
        if len(Xc) < 200:
            Xc = X
        Xc = Xc.sample(min(max_rows, len(Xc)), random_state=seed)
        base = _logit(self.predict_frame(Xc, require_complete=False))
        out = {}
        for a in anchors:
            members = groups.get(a, [a])
            sd = float(X[a].std())
            if not np.isfinite(sd) or sd == 0:
                continue
            dl = []
            for d in deltas:
                Xp = Xc.copy()
                for m in members:
                    s, lo, hi = float(X[m].std()), float(X[m].min()), float(X[m].max())
                    Xp[m] = np.clip(Xc[m].to_numpy() + d * (s if np.isfinite(s) else 0), lo, hi)
                dl.append(round(float(np.mean(_logit(self.predict_frame(Xp, require_complete=False)) - base)), 4))
            out[a] = dict(sigma=round(sd, 4), z=list(deltas), dlogit=dl, members=list(members))
        return out
