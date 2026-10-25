"""Tingkat risiko dari skor model (BELUM dikalibrasi) + warna + saran.

Ambang diambil dari model_spec_2026.json milik Orang B:
  Sedang        >= threshold_recall70  (sensitivitas tinggi, banyak peringatan palsu)
  Tinggi        >= threshold_f1        (keseimbangan precision/recall; sama dengan definisi 'alert' B)
  Sangat Tinggi >= very_high_threshold (SEMENTARA, default 0.40; belum divalidasi B)
"""
import numpy as np
import pandas as pd

DEFAULT_LEVELS = [
    dict(name="Rendah", min=0.0, color="#2e9e5b",
         advice="Di bawah ambang waspada. Pantau rutin; kejadian tetap mungkin terjadi."),
    dict(name="Sedang", min=0.2, color="#f2c230",
         advice="Waspada. Periksa titik rawan (lahan gambut, area bekas terbakar) dan siagakan patroli."),
    dict(name="Tinggi", min=0.4, color="#ef8a1f",
         advice="Tingkatkan patroli, siapkan sarana pemadaman, dan larang pembakaran lahan."),
    dict(name="Sangat Tinggi", min=0.6, color="#d7301f",
         advice="Siaga penuh. Koordinasikan dengan BPBD/Manggala Agni dan pantau harian."),
]
_META = [("Rendah", "#2e9e5b", "Di bawah ambang waspada. Pantau rutin; kejadian tetap mungkin terjadi."),
         ("Sedang", "#f2c230", "Di atas ambang sensitivitas tinggi: menjaring sebagian besar kejadian historis, tetapi banyak peringatan palsu. Waspada dan periksa titik rawan."),
         ("Tinggi", "#ef8a1f", "Di atas ambang F1 model (definisi alert Orang B). Tingkatkan patroli dan siapkan sarana pemadaman."),
         ("Sangat Tinggi", "#d7301f", "Skor tertinggi. Koordinasikan dengan pihak berwenang dan pantau harian.")]


def levels_from_spec(spec: dict, very_high: float = 0.40) -> list[dict]:
    f1 = float(spec["threshold_f1"])
    lo = spec.get("threshold_recall70")
    lo = float(lo) if lo is not None else f1 / 2
    if not (0 < lo < f1 < very_high < 1):
        raise ValueError(f"ambang harus 0 < recall70 ({lo}) < f1 ({f1}) < very_high ({very_high}) < 1")
    mins = [0.0, lo, f1, very_high]
    return [dict(name=n, min=m, color=c, advice=a) for (n, c, a), m in zip(_META, mins)]


def load_levels(cfg: dict | None = None, spec: dict | None = None) -> list[dict]:
    """Prioritas: cfg['app']['risk_levels'] > ambang dari spec model > bawaan. Divalidasi."""
    app = (cfg or {}).get("app") or {}
    levels = app.get("risk_levels")
    if not levels:
        levels = levels_from_spec(spec, app.get("very_high_threshold", 0.40)) if spec else DEFAULT_LEVELS
    levels = sorted((dict(l) for l in levels), key=lambda l: l["min"])
    if levels[0]["min"] != 0:
        raise ValueError("kelas pertama harus punya min = 0")
    if len({l["min"] for l in levels}) != len(levels):
        raise ValueError("nilai min antar-kelas harus berbeda")
    for l in levels:
        if not (0 <= l["min"] < 1) or not {"name", "color"} <= l.keys():
            raise ValueError(f"kelas tidak valid: {l}")
        l.setdefault("advice", "")
    return levels


def level_index(prob, levels) -> np.ndarray:
    """Indeks kelas per skor (NaN -> -1)."""
    p = np.asarray(prob, float)
    idx = np.searchsorted([l["min"] for l in levels], p, side="right") - 1
    return np.where(np.isnan(p), -1, idx)


def level_name(prob, levels) -> np.ndarray:
    names = np.array([l["name"] for l in levels] + ["Tidak ada data"], dtype=object)
    return names[level_index(prob, levels)]  # -1 -> elemen terakhir


def summarize(df: pd.DataFrame, levels, by: str = "province", prob_col: str = "prob") -> pd.DataFrame:
    d = df.assign(level=level_name(df[prob_col], levels))
    cnt = d.groupby([by, "level"]).size().unstack(fill_value=0)
    cnt = cnt.reindex(columns=[l["name"] for l in levels], fill_value=0)
    out = cnt.copy()
    out["n"] = cnt.sum(axis=1)
    out["mean_prob"] = d.groupby(by)[prob_col].mean()
    return out
