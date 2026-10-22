"""FIRMS: unduh (opsional) dan bersihkan hotspot jadi tabel harian per sel."""
import argparse
import hashlib
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd

from .common import cell_id, load_config, rpath

STEP = 5  # batas API: maks 5 hari per request


# ---------------------------------------------------------------- unduh
def download(cfg: dict, key: str, workers: int = 2):
    """Unduh potongan 5 hari ke cache dan gabung jadi firms_raw.csv.
    Aman diulang: potongan yang sudah ada dilewati."""
    import requests
    f, d = cfg["firms"], cfg["dates"]
    start, end = date.fromisoformat(d["start"]), date.fromisoformat(d["end"])
    sp_end = date.fromisoformat(f["sp_end"])
    cache = rpath("data/raw/firms_cache")
    cache.mkdir(parents=True, exist_ok=True)

    jobs = []
    for bb in f["bboxes"].values():
        for s in f["sensors"]:
            for src, a, b in ((f"{s}_SP", start, sp_end),
                              (f"{s}_NRT", sp_end + timedelta(days=1), end)):
                x = a
                while x <= b:
                    jobs.append((src, bb, x)); x += timedelta(days=STEP)

    def fetch(j):
        src, bb, x = j
        fp = cache / (hashlib.md5(f"{src}{bb}{x}".encode()).hexdigest() + ".csv")
        if fp.exists():
            return None if fp.stat().st_size <= 1 else pd.read_csv(fp).assign(source=src)
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{src}/{bb}/{STEP}/{x}"
        for k in range(8):
            try:
                r = requests.get(url, timeout=90)
            except requests.RequestException:
                time.sleep(10 * (k + 1)); continue
            t = r.text.strip()
            if r.ok and t.startswith("latitude"):
                if "\n" not in t:
                    fp.write_text(" "); return None
                fp.write_text(t)
                return pd.read_csv(io.StringIO(t)).assign(source=src)
            if r.status_code in (429, 500, 502, 503, 504) or "Invalid MAP_KEY" in t or "limit" in t.lower():
                time.sleep(min(30 * (k + 1), 180)); continue
            return None
        return None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        frames = [x for x in ex.map(fetch, jobs) if x is not None and len(x)]
    raw = pd.concat(frames).drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time", "source"])
    raw.to_csv(rpath(f["raw_csv"]), index=False)
    print(f"{len(frames)}/{len(jobs)} potongan -> {len(raw)} baris")


# ---------------------------------------------------------------- bersihkan
def clean_firms(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Filter sensor/kepercayaan/tipe, utamakan SP di atas NRT, buang duplikat."""
    f = cfg["firms"]
    df = raw.copy()
    df["sensor"] = df["source"].str.replace(r"_(SP|NRT)$", "", regex=True)
    df["is_nrt"] = df["source"].str.endswith("_NRT")
    df = df[df["sensor"].isin(f["sensors"])]

    conf = df["confidence"].astype("string").str.strip().str.lower()
    is_viirs = df["sensor"].str.startswith("VIIRS")
    keep = pd.Series(True, index=df.index)
    if f["drop_viirs_low_conf"]:
        keep &= ~(is_viirs & (conf == "l"))
    keep &= is_viirs | (pd.to_numeric(conf, errors="coerce") >= f["min_modis_conf"])
    if "type" in df:  # 0 = vegetasi; kolom ini tidak ada/NaN di sebagian NRT
        keep &= df["type"].isna() | (df["type"] == 0)
    df = df[keep].copy()

    df["acq_date"] = pd.to_datetime(df["acq_date"])
    sp_days = df.loc[~df["is_nrt"], ["sensor", "acq_date"]].drop_duplicates().assign(_sp=True)
    df = df.merge(sp_days, on=["sensor", "acq_date"], how="left")
    df = df[~(df["is_nrt"] & df["_sp"].notna())].drop(columns="_sp")
    return df.drop_duplicates(["latitude", "longitude", "acq_date", "acq_time", "sensor"])


def aggregate_daily(df: pd.DataFrame, grid: pd.DataFrame, d: float) -> pd.DataFrame:
    df = df.assign(cell_id=cell_id(df["longitude"], df["latitude"], d))
    df = df[df["cell_id"].isin(grid["cell_id"])]
    out = (df.groupby(["cell_id", "acq_date"])
             .agg(n_hotspot=("latitude", "size"), frp_sum=("frp", "sum"))
             .reset_index().rename(columns={"acq_date": "date"}))
    out["n_hotspot"] = out["n_hotspot"].astype("int32")
    out["frp_sum"] = out["frp_sum"].astype("float32")
    return out


def main(cfg=None):
    cfg = cfg or load_config()
    out = rpath(cfg["paths"]["interim"])
    grid = pd.read_csv(out / "grid.csv")
    raw = pd.read_csv(rpath(cfg["firms"]["raw_csv"]), low_memory=False,
                      usecols=lambda c: c in {"latitude", "longitude", "acq_date", "acq_time",
                                              "confidence", "frp", "type", "source"},
                      dtype={"confidence": "string"})
    clean = clean_firms(raw, cfg)
    daily = aggregate_daily(clean, grid, cfg["grid"]["cell_deg"])
    daily.to_parquet(out / "hotspots_daily.parquet", index=False)
    last = clean["acq_date"].max()
    (out / "firms_meta.json").write_text(json.dumps({"last_date": str(last.date())}))
    qc = clean.groupby([clean["acq_date"].dt.to_period("M"), "is_nrt"]).size().unstack(fill_value=0)
    print(f"{len(raw)} mentah -> {len(clean)} bersih -> {len(daily)} sel-hari; data terakhir {last.date()}")
    print("QC bulanan (False=SP, True=NRT), 12 bulan terakhir:\n", qc.tail(12).to_string())
    return daily


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", metavar="MAP_KEY", help="unduh dulu via API FIRMS")
    a = ap.parse_args()
    c = load_config()
    if a.download:
        download(c, a.download)
    main(c)
