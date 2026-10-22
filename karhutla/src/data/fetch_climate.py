"""Iklim per sel: CHIRPS v3, ERA5-Land, ONI, DMI.

Alur terkelola: --plan (tidak kirim task) -> --submit (maksimal beberapa task)
-> --status -> unduh CSV dari Drive ke data/raw/gee/batches/ -> --ingest
-> --assemble (PREVIEW). --assemble --final menolak periode/wilayah yang belum lengkap.

Alur --export lama tetap tersedia untuk satu batch uji. Mode --direct menarik
batch kecil langsung lewat computeFeatures() ke Python (tanpa Drive). Data
mentah asli dipertahankan. Mode interaktif ini mempunyai batas kuota/waktu.
"""
import argparse
import re
import json
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .common import load_config, rpath

SPECS = {
    "chirps": dict(asset="UCSB-CHC/CHIRPS/V3/DAILY_SAT", bands=["precipitation"], scale=5566),
    "era5": dict(asset="ECMWF/ERA5_LAND/DAILY_AGGR", scale=11132,
                 bands=["temperature_2m", "dewpoint_temperature_2m", "volumetric_soil_water_layer_1",
                        "u_component_of_wind_10m", "v_component_of_wind_10m"]),
}


# ---------------------------------------------------------------- ekspor GEE
def _cells(cfg, ee):
    d = cfg["grid"]["cell_deg"]; h = d / 2
    grid = pd.read_csv(rpath(cfg["paths"]["interim"]) / "grid.csv")
    return ee.FeatureCollection([
        ee.Feature(ee.Geometry.Rectangle([x - h, y - h, x + h, y + h], "EPSG:4326", False), {"cell_id": int(i)})
        for i, x, y in zip(grid.cell_id, grid.lon, grid.lat)])


def export_gee_vi(cfg: dict):
    """NDVI/EVI MOD13Q1 per komposit 16 hari (piksel SummaryQA <= 1), satu task per tahun."""
    import ee
    c = cfg["climate"]
    ee.Initialize(project=c["gee_project"])
    cells = _cells(cfg, ee)
    y0, y1 = pd.Timestamp(cfg["dates"]["start"]).year, pd.Timestamp(cfg["dates"]["end"]).year
    for y in range(y0, y1 + 1):
        ic = ee.ImageCollection("MODIS/061/MOD13Q1").filterDate(f"{y}-01-01", f"{y + 1}-01-01")

        def per_img(img):
            v = img.select(["NDVI", "EVI"]).updateMask(img.select("SummaryQA").lte(1)).multiply(0.0001)
            day = img.date().format("YYYY-MM-dd")
            return v.reduceRegions(collection=cells, reducer=ee.Reducer.mean(), scale=250) \
                    .map(lambda f: f.set("date", day))

        ee.batch.Export.table.toDrive(collection=ic.map(per_img).flatten(), description=f"modisvi_{y}",
                                      folder=c["gee_folder"], fileFormat="CSV",
                                      selectors=["cell_id", "date", "NDVI", "EVI"]).start()
        print("task dimulai: modisvi", y)

def pilih_grid_batch(grid: pd.DataFrame, province: str, max_cells: int) -> pd.DataFrame:
    """Pilih subset sel secara deterministik; tidak mengubah grid.csv."""
    required = {"cell_id", "lon", "lat", "province"}
    if not required.issubset(grid.columns):
        raise ValueError(f"grid.csv kurang kolom {sorted(required - set(grid.columns))}")
    if not (1 <= max_cells <= 200):
        raise ValueError("Mode uji hanya mengizinkan 1-200 sel per batch")
    sub = grid.loc[grid["province"].eq(province)].sort_values("cell_id")
    if sub.empty:
        raise ValueError(f"Provinsi {province!r} tidak ada dalam grid.csv")
    n = min(len(sub), max_cells)
    # Sampel tersebar di urutan ID sel, agar tidak hanya dua baris pertama.
    idx = np.linspace(0, len(sub) - 1, n).astype(int)
    out = sub.iloc[idx].copy()
    if out["cell_id"].duplicated().any() or out[["lon", "lat"]].isna().any().any():
        raise ValueError("Grid batch memiliki duplikat/koordinat kosong")
    return out


def validasi_periode_batch(start: str, end: str) -> int:
    """Batas akhir eksklusif; satu batch uji maksimal 7 hari."""
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    n = (b - a).days
    if not (1 <= n <= 7) or a.hour or b.hour:
        raise ValueError("Gunakan periode 1-7 hari, tanggal akhir bersifat eksklusif")
    return n


def export_gee(cfg: dict, *, source: str, start: str, end: str,
               province: str, max_cells: int):
    """EKSPOR TERBATAS untuk tes batch iklim; tidak mengekspor seluruh dataset.

    Perintah --export WAJIB menerima sumber, tanggal, provinsi dan jumlah sel.
    Nilai kosong diekspor sebagai -9999, tidak diperlakukan sebagai nol hujan.
    """
    if source not in SPECS:
        raise ValueError(f"source harus salah satu dari {sorted(SPECS)}")
    n_days = validasi_periode_batch(start, end)
    grid = pd.read_csv(rpath(cfg["paths"]["interim"]) / "grid.csv")
    sample = pilih_grid_batch(grid, province, max_cells)

    import ee
    c = cfg["climate"]
    ee.Initialize(project=c["gee_project"])
    spec = dict(SPECS[source])
    if source == "chirps":
        spec["asset"] = c["chirps_asset"]  # gunakan CHIRPS v3, jangan campur versi
    bands = spec["bands"]
    pts = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]),
                   {"cell_id": int(r.cell_id), "province": str(r.province)})
        for r in sample.itertuples(index=False)
    ])
    ic = (ee.ImageCollection(spec["asset"]).filterDate(start, end).select(bands))
    available = int(ic.size().getInfo())
    if available != n_days:
        raise RuntimeError(f"Citra {source} tidak lengkap: {available} dari {n_days} hari")

    def extract(img):
        day = img.date().format("YYYY-MM-dd")
        # -9999 menjaga ketersediaan baris untuk piksel bertopeng.
        # Diterjemahkan menjadi NaN pada saat pembacaan CSV.
        return (img.select(bands).unmask(-9999, False)
                .sampleRegions(collection=pts, properties=["cell_id", "province"],
                               scale=spec["scale"], geometries=False, tileScale=4)
                .map(lambda f: f.set("date", day)))

    rows = ic.map(extract).flatten()
    safe_province = re.sub(r"[^a-z0-9]+", "_", province.lower()).strip("_")
    name = f"{source}_{start.replace('-', '')}_{end.replace('-', '')}_{safe_province}_n{len(sample)}"
    drive_folder = c["gee_folder"] + "_BATCH_TEST"
    task = ee.batch.Export.table.toDrive(
        collection=rows, description=name, folder=drive_folder,
        fileNamePrefix=name, fileFormat="CSV", selectors=["cell_id", "date"] + bands)
    task.start()
    print(f"Source: {source} | sel: {len(sample)} | hari: {n_days} | perkiraan baris: {n_days * len(sample)}")
    print(f"TASK DIMULAI: {name} | ID: {task.id}")
    print(f"Drive folder: {drive_folder}")
    print("Tunggu COMPLETED; unduh CSV batch ke folder uji TERPISAH (data/raw/gee/batch_test).")


# ---------------------------------------------------------------- baca hasil
def load_gee_csvs(folder, prefix: str, bands: list[str]) -> pd.DataFrame:
    files = sorted(p for p in folder.glob(f"{prefix}_*.csv"))
    if not files:
        raise FileNotFoundError(f"tidak ada {prefix}_*.csv di {folder}")
    df = pd.concat((pd.read_csv(p) for p in files), ignore_index=True)
    if len(bands) == 1 and bands[0] not in df and "mean" in df:  # jaga-jaga penamaan GEE
        df = df.rename(columns={"mean": bands[0]})
    df["cell_id"] = df["cell_id"].round().astype("int64")
    df["date"] = pd.to_datetime(df["date"])
    df[bands] = df[bands].replace(-9999, np.nan).astype("float32")
    if df.duplicated(["cell_id", "date"]).any():
        raise ValueError(f"Ada grid-tanggal ganda pada CSV {prefix}, jangan dibuang diam-diam")
    return df[["cell_id", "date"] + bands]


def _esat(t_c):  # tekanan uap jenuh (hPa), rumus Magnus
    return 6.1094 * np.exp(17.625 * t_c / (t_c + 243.04))


def derive_era5(df: pd.DataFrame) -> pd.DataFrame:
    t = df["temperature_2m"] - 273.15
    td = df["dewpoint_temperature_2m"] - 273.15
    out = pd.DataFrame({
        "cell_id": df["cell_id"], "date": df["date"],
        "t2m_c": t,
        "rh": (100 * _esat(td) / _esat(t)).clip(0, 100),
        "sm1": df["volumetric_soil_water_layer_1"],
        "wind_ms": np.hypot(df["u_component_of_wind_10m"], df["v_component_of_wind_10m"]),
    })
    return out.astype({c: "float32" for c in ["t2m_c", "rh", "sm1", "wind_ms"]})


def build_climate_daily(cfg: dict) -> pd.DataFrame:
    folder = rpath(cfg["climate"]["gee_dir"])
    ch = load_gee_csvs(folder, "chirps", SPECS["chirps"]["bands"]).rename(columns={"precipitation": "precip_mm"})
    er = derive_era5(load_gee_csvs(folder, "era5", SPECS["era5"]["bands"]))
    out = ch.merge(er, on=["cell_id", "date"], how="outer").sort_values(["date", "cell_id"])
    return out.reset_index(drop=True)


def load_vi(cfg: dict):
    """Komposit MOD13Q1 (date = tanggal awal komposit). None bila belum diunduh."""
    folder = rpath(cfg["climate"]["gee_dir"])
    if not list(folder.glob("modisvi_*.csv")):
        return None
    return load_gee_csvs(folder, "modisvi", ["NDVI", "EVI"])


# ---------------------------------------------------------------- ENSO / IOD
SEASONS = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]


def read_oni(path) -> pd.DataFrame:
    """ONI (SEAS, YR, TOTAL, ANOM) -> bulanan; musim 3 bulan dipetakan ke bulan tengahnya."""
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    m = df["SEAS"].str.strip().map({s: i + 1 for i, s in enumerate(SEASONS)})
    out = pd.DataFrame({"month": pd.to_datetime(dict(year=df["YR"], month=m, day=1)), "oni": df["ANOM"].astype(float)})
    return out.dropna().drop_duplicates("month")


def parse_dmi(text: str) -> pd.DataFrame:
    """Format PSL: baris 'tahun + 12 nilai'; -9999 = hilang; baris lain (header/catatan) diabaikan."""
    rows = []
    for line in text.splitlines():
        tok = line.split()
        if len(tok) == 13 and re.fullmatch(r"\d{4}", tok[0]):
            try:
                vals = [float(x) for x in tok[1:]]
            except ValueError:
                continue
            for m, v in enumerate(vals, 1):
                if v > -999:
                    rows.append((pd.Timestamp(int(tok[0]), m, 1), v))
    return pd.DataFrame(rows, columns=["month", "dmi"])


def build_enso(cfg: dict) -> pd.DataFrame:
    c = cfg["climate"]
    oni = read_oni(rpath(c["oni_file"]))
    dmi = parse_dmi(rpath(c["dmi_file"]).read_text())
    return oni.merge(dmi, on="month", how="outer").sort_values("month").reset_index(drop=True)


def main(cfg=None):
    cfg = cfg or load_config()
    out = rpath(cfg["paths"]["interim"])
    clim = build_climate_daily(cfg)
    clim.to_parquet(out / "climate_daily.parquet", index=False)
    vi = load_vi(cfg)
    if vi is not None:
        vi.to_parquet(out / "vi_composites.parquet", index=False)
        print(f"VI: {len(vi)} baris, komposit terakhir {vi.date.max().date()}")
    else:
        print("PERINGATAN: modisvi_*.csv tidak ada -> tanpa fitur NDVI/EVI")
    enso = build_enso(cfg)
    enso.to_csv(out / "enso_monthly.csv", index=False)
    print(f"iklim: {len(clim)} baris, {clim.date.min().date()} s/d {clim.date.max().date()}; "
          f"ENSO: {enso.month.min():%Y-%m} s/d {enso.month.max():%Y-%m}")



# ---------------------------------------------------------------- otomatisasi TERKENDALI
# Mode baru tidak mengganti --export lama. Manifest menyimpan pekerjaan yang
# direncanakan, ID task, dan status agar pekerjaan tidak dikirim berulang.
MANIFEST_COLUMNS = [
    "name", "source", "province", "start", "end", "chunk", "row_start", "row_end",
    "n_cells", "expected_rows", "grid_sha256", "status", "task_id", "error",
]
ACTIVE_STATES = {"SUBMITTING", "READY", "RUNNING", "UNKNOWN"}


def _manifest_path(cfg) -> Path:
    return rpath(cfg["paths"]["interim"]) / "gee_batch_manifest.csv"


def _grid_snapshot(cfg) -> tuple[pd.DataFrame, str]:
    path = rpath(cfg["paths"]["interim"]) / "grid.csv"
    df = pd.read_csv(path)
    needed = {"cell_id", "lon", "lat", "province"}
    if not needed.issubset(df):
        raise ValueError(f"grid.csv tidak memiliki kolom: {needed - set(df.columns)}")
    if df.cell_id.isna().any() or df.cell_id.duplicated().any():
        raise ValueError("cell_id kosong/ganda di grid.csv")
    if df[["lat", "lon", "province"]].isna().any().any():
        raise ValueError("koordinat/provinsi kosong di grid.csv")
    df = df.sort_values(["province", "cell_id"]).reset_index(drop=True)
    cols = ["cell_id", "province", "lon", "lat"]
    digest = hashlib.sha256(df[cols].to_csv(index=False, float_format="%.9f").encode()).hexdigest()
    return df, digest


def _read_manifest(cfg) -> pd.DataFrame:
    path = _manifest_path(cfg)
    if not path.exists():
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    df = pd.read_csv(path, dtype={"name": str, "task_id": str, "grid_sha256": str},
                     keep_default_na=False)
    missing = set(MANIFEST_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Manifest rusak: kolom hilang {sorted(missing)}")
    for c in ("chunk", "row_start", "row_end", "n_cells", "expected_rows"):
        df[c] = pd.to_numeric(df[c], errors="raise").astype("int64")
    return df[MANIFEST_COLUMNS]


def _save_manifest(cfg, df: pd.DataFrame):
    p = _manifest_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(".csv.tmp")
    df.to_csv(temp, index=False)
    temp.replace(p)


def _validate_snapshot(cfg, manifest: pd.DataFrame) -> pd.DataFrame:
    grid, digest = _grid_snapshot(cfg)
    if len(manifest) and not manifest.grid_sha256.eq(digest).all():
        raise RuntimeError("grid.csv berbeda dari saat manifest dibuat. Jangan teruskan; "
                           "periksa perubahan grid dan buat proyek manifest terpisah.")
    return grid


def _date_str(x) -> str:
    t = pd.Timestamp(x)
    if pd.isna(t) or t != t.normalize():
        raise ValueError("Tanggal wajib YYYY-MM-DD tanpa jam")
    return t.strftime("%Y-%m-%d")


def _prov_slug(prov: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", prov.lower()).strip("_")


def buat_rencana(cfg, *, source: str, start: str, end: str, province: str,
                 cells_per_batch: int, days_per_batch: int):
    """Hanya membuat manifest CSV. Tidak ada koneksi GEE dan tidak ada task."""
    if source not in (*SPECS, "both"):
        raise ValueError("--source: chirps, era5, atau both")
    start, end = _date_str(start), _date_str(end)
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    if a >= b:
        raise ValueError("Tanggal akhir harus setelah tanggal awal (end eksklusif)")
    if not 1 <= cells_per_batch <= 3000 or not 1 <= days_per_batch <= 31:
        raise ValueError("Gunakan 1..3000 grid dan 1..31 hari per batch")
    if cells_per_batch * days_per_batch > 30000:
        raise ValueError("Maksimal 30.000 baris per task untuk memulai dengan aman")
    grid, digest = _grid_snapshot(cfg)
    provs = sorted(grid.province.unique().tolist()) if province == "all" else [province]
    if any(p not in set(grid.province) for p in provs):
        raise ValueError(f"Provinsi tidak dikenal: {province}")
    old = _read_manifest(cfg)
    _validate_snapshot(cfg, old)
    proposed = []
    sources = list(SPECS) if source == "both" else [source]
    for src in sources:
        for prov in provs:
            sub = grid.loc[grid.province.eq(prov)].sort_values("cell_id")
            for t0 in pd.date_range(a, b - pd.Timedelta(days=1), freq=f"{days_per_batch}D"):
                t1 = min(t0 + pd.Timedelta(days=days_per_batch), b)
                for chunk, lo in enumerate(range(0, len(sub), cells_per_batch)):
                    hi = min(lo + cells_per_batch, len(sub))
                    cell_ids = sub.iloc[lo:hi].cell_id.astype("int64").tolist()
                    chunk_hash = hashlib.sha1(",".join(map(str, cell_ids)).encode()).hexdigest()[:7]
                    name = (f"{src}_{t0:%Y%m%d}_{t1:%Y%m%d}_{_prov_slug(prov)}_"
                            f"c{chunk:03d}_n{hi-lo}_h{chunk_hash}")
                    # Tolak rencana overlap meski ukuran chunk/nama berubah.
                    for _, exist in old[(old.source == src) & (old.province == prov)].iterrows():
                        if t0 < pd.Timestamp(exist["end"]) and pd.Timestamp(exist["start"]) < t1:
                            if max(lo, int(exist.row_start)) < min(hi, int(exist.row_end)):
                                if exist["name"] != name:
                                    raise ValueError(f"Rencana tumpang tindih dengan {exist['name']}")
                    proposed.append(dict(name=name, source=src, province=prov,
                                         start=f"{t0:%Y-%m-%d}", end=f"{t1:%Y-%m-%d}",
                                         chunk=chunk, row_start=lo, row_end=hi,
                                         n_cells=hi-lo, expected_rows=(hi-lo)*(t1-t0).days,
                                         grid_sha256=digest, status="PLANNED",
                                         task_id="", error=""))
    new = pd.DataFrame(proposed, columns=MANIFEST_COLUMNS)
    new = new[~new.name.isin(old.name)]
    together = pd.concat([old, new], ignore_index=True)
    _save_manifest(cfg, together)
    print(f"Rencana baru: {len(new)} task | total dalam manifest: {len(together)}")
    print(f"Perkiraan baris BARU: {int(new.expected_rows.sum()):,} | RAM tidak dialokasikan sekarang")
    print(f"Manifest: {_manifest_path(cfg)}")
    print("Tidak ada task yang dikirim. Lihat manifest sebelum --submit.")


def _feature_points(sub, ee):
    return ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]),
                   {"cell_id": int(r.cell_id)}) for r in sub.itertuples(index=False)
    ])


def _refresh_tasks(cfg, manifest: pd.DataFrame, ee, *, verbose: bool = False):
    to_update = manifest.index[(manifest.task_id != "") & manifest.status.isin(ACTIVE_STATES)].tolist()
    for j in range(0, len(to_update), 50):
        indices = to_update[j:j+50]
        ids = manifest.loc[indices, "task_id"].tolist()
        statuses = {d.get("id"): d for d in ee.data.getTaskStatus(ids)}
        for idx in indices:
            entry = statuses.get(manifest.at[idx, "task_id"], {})
            if entry.get("state"):
                manifest.at[idx, "status"] = entry["state"]
                if entry.get("error_message"):
                    manifest.at[idx, "error"] = entry["error_message"][:500]
    _save_manifest(cfg, manifest)
    if verbose:
        print(manifest.status.value_counts().to_string())
    return manifest


def cek_status(cfg):
    import ee
    m = _read_manifest(cfg)
    if m.empty:
        print("Manifest kosong. Jalankan --plan terlebih dulu.")
        return
    _validate_snapshot(cfg, m)
    ee.Initialize(project=cfg["climate"]["gee_project"])
    _refresh_tasks(cfg, m, ee, verbose=True)


def kirim_batch(cfg, *, max_new_tasks: int, max_active_tasks: int):
    """Mengirim task terbatas, tidak ada ekspor massal otomatis."""
    if not (1 <= max_new_tasks <= 3) or not (1 <= max_active_tasks <= 5):
        raise ValueError("Batas aman: --max-new-tasks <= 3, --max-active-tasks <= 5")
    m = _read_manifest(cfg)
    if m.empty:
        raise ValueError("Manifest belum ada; jalankan --plan dahulu")
    grid = _validate_snapshot(cfg, m)
    import ee
    c = cfg["climate"]
    ee.Initialize(project=c["gee_project"])
    m = _refresh_tasks(cfg, m, ee)
    active = int(m.status.isin(ACTIVE_STATES).sum())
    slots = min(max_new_tasks, max_active_tasks - active)
    if slots <= 0:
        print(f"Sudah ada {active} task aktif. Tunggu; gunakan --status.")
        return
    # Selingi sumber untuk tanggal/grid yang sama; mudah memantau pasangan.
    waiting = m[m.status.eq("PLANNED")].sort_values(
        ["province", "start", "row_start", "source"])
    pending = waiting.index.tolist()[:slots]
    if not pending:
        print("Tidak ada batch PLANNED. Gunakan --status atau --ingest.")
        return
    for idx in pending:
        record = m.loc[idx]
        spec = dict(SPECS[record.source])
        if record.source == "chirps":
            spec["asset"] = c.get("chirps_asset", spec["asset"])
            if "V3" not in spec["asset"].upper():
                raise ValueError("Konfigurasi CHIRPS bukan v3; hentikan agar versi tidak tercampur")
        sub = (grid.loc[grid.province.eq(record.province)]
               .sort_values("cell_id").iloc[int(record.row_start):int(record.row_end)])
        if len(sub) != record.n_cells:
            raise RuntimeError("Jumlah grid berubah")
        pts = _feature_points(sub, ee)
        ic = (ee.ImageCollection(spec["asset"])
              .filterDate(record.start, record.end).select(spec["bands"]))
        available = int(ic.size().getInfo())
        expected_days = (pd.Timestamp(record.end)-pd.Timestamp(record.start)).days
        if available != expected_days:
            raise RuntimeError(f"{record.name}: {available}/{expected_days} citra; task tidak dibuat")
        bands = spec["bands"]
        def per_img(img):
            stamp = img.date().format("YYYY-MM-dd")
            return (img.select(bands).unmask(-9999, False)
                    .sampleRegions(collection=pts, properties=["cell_id"],
                                   scale=spec["scale"], geometries=False, tileScale=4)
                    .map(lambda ft: ft.set("date", stamp)))
        collection = ic.map(per_img).flatten()
        task = ee.batch.Export.table.toDrive(
            collection=collection, description=record["name"],
            fileNamePrefix=record["name"], folder=c["gee_folder"] + "_BATCH_FULL",
            fileFormat="CSV", selectors=["cell_id", "date"] + bands)
        # Task ID Earth Engine baru diketahui setelah start(). Catat status
        # SUBMITTING lebih dulu agar penghentian mendadak tidak otomatis
        # mengirim ulang task dan menghasilkan data duplikat.
        m.at[idx, "task_id"] = ""
        m.at[idx, "status"] = "SUBMITTING"
        _save_manifest(cfg, m)
        try:
            task.start()
        except Exception as exc:
            m.at[idx, "status"] = "START_ERROR"
            m.at[idx, "error"] = str(exc)[:500]
            _save_manifest(cfg, m)
            raise
        m.at[idx, "task_id"] = task.id
        m.at[idx, "status"] = "READY"
        _save_manifest(cfg, m)
        print(f"TASK DIMULAI: {record['name']} | {task.id}")
    print(f"Drive: {c['gee_folder']}_BATCH_FULL")
    print("Setelah COMPLETED, download CSV ke data/raw/gee/batches/ lalu --ingest.")


def _sanitize_csv_batch(path: Path, record: pd.Series, expected_ids: set[int]):
    bands = SPECS[record.source]["bands"]
    df = pd.read_csv(path)
    if len(df) != int(record.expected_rows):
        raise ValueError(f"{path.name}: baris {len(df)} != target {record.expected_rows}")
    required = {"cell_id", "date", *bands}
    if not required.issubset(df):
        raise ValueError(f"{path.name}: kolom kurang {required - set(df)}")
    ids = pd.to_numeric(df.cell_id, errors="raise")
    if ids.isna().any() or not (ids == ids.astype("int64")).all():
        raise ValueError(f"{path.name}: cell_id tidak valid")
    df["cell_id"] = ids.astype("int64")
    if set(df.cell_id) != expected_ids:
        raise ValueError(f"{path.name}: set cell_id berbeda dari rencana batch")
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    if not ((df.date >= pd.Timestamp(record.start)) & (df.date < pd.Timestamp(record.end))).all():
        raise ValueError(f"{path.name}: ada tanggal di luar periode")
    if df.duplicated(["cell_id", "date"]).any():
        raise ValueError(f"{path.name}: ada duplikat grid-tanggal")
    for b in bands:
        df[b] = pd.to_numeric(df[b], errors="coerce").replace(-9999, np.nan).astype("float32")
    if record.source == "chirps":
        if df["precipitation"].dropna().lt(0).any():
            raise ValueError("CHIRPS berisi hujan negatif")
        out = df[["cell_id", "date", "precipitation"]].rename(columns={"precipitation": "precip_mm"})
    else:
        t = df.temperature_2m.dropna()
        if (~t.between(180, 350)).any():
            raise ValueError("Suhu ERA5 di luar batas kelayakan 180..350 K")
        out = derive_era5(df)
    if out.duplicated(["cell_id", "date"]).any():
        raise ValueError("Duplikat setelah konversi")
    return out.sort_values(["date", "cell_id"]).reset_index(drop=True)


def ingest_batches(cfg, *, max_files: int):
    """Memproses file yang diunduh, satu file sekaligus; sumber mentah tidak dihapus."""
    m = _read_manifest(cfg)
    if m.empty:
        raise ValueError("Manifest belum ada")
    grid = _validate_snapshot(cfg, m)
    folder = rpath(cfg["climate"]["gee_dir"]) / "batches"
    done = 0
    eligible = m.index[m.status.isin(["COMPLETED", "DOWNLOADED", "NEEDS_REVIEW", "DIRECT_ERROR"])].tolist()
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder unduhan belum ada: {folder}")
    for idx in eligible:
        if done >= max_files:
            break
        record = m.loc[idx]
        candidates = list(folder.glob(record["name"] + ".csv"))
        if not candidates:
            continue
        sub = (grid.loc[grid.province.eq(record.province)].sort_values("cell_id")
               .iloc[int(record.row_start):int(record.row_end)])
        try:
            clean = _sanitize_csv_batch(candidates[0], record, set(map(int, sub.cell_id)))
        except Exception as exc:
            m.at[idx, "status"] = "NEEDS_REVIEW"
            m.at[idx, "error"] = str(exc)[:500]
            _save_manifest(cfg, m)
            raise
        dest = rpath(cfg["paths"]["interim"]) / "climate_parts" / record.source
        dest.mkdir(parents=True, exist_ok=True)
        p = dest / (record["name"] + ".parquet")
        tmp = p.with_suffix(".parquet.tmp")
        clean.to_parquet(tmp, index=False, compression="zstd")
        tmp.replace(p)
        missing = int(clean.drop(columns=["cell_id", "date"]).isna().sum().sum())
        m.at[idx, "status"] = "INGESTED"
        m.at[idx, "error"] = f"missing_numeric_values={missing}"
        _save_manifest(cfg, m)
        done += 1
        print(f"INGESTED {record['name']} | {len(clean)} baris | null numerik {missing}")
    print(f"Total CSV yang berhasil diproses saat ini: {done}; data mentah tetap ada.")
    if not done:
        print("Cek task COMPLETED dengan --status, nama CSV, dan folder data/raw/gee/batches/.")


# --------------------------------------------------- unduh langsung, tanpa Google Drive
# Earth Engine computeFeatures mengirim DataFrame lewat API INTERAKTIF; gunakan
# hanya batch kecil. Mode ini bukan pengganti eksportir batch untuk jutaan baris.
DIRECT_MAX_ROWS = 5000
DIRECT_ALLOWED_STATES = {"PLANNED", "COMPLETED", "DIRECT_ERROR"}


def _direct_candidates(manifest: pd.DataFrame) -> pd.DataFrame:
    """Dahulukan pasangan yang sudah parsial, lalu batch terawal."""
    pending = manifest[manifest.status.isin(DIRECT_ALLOWED_STATES)].copy()
    if pending.empty:
        return pending
    completed = manifest[manifest.status.eq("INGESTED")]
    key = ["province", "start", "end", "row_start", "row_end"]
    pair_keys = set(tuple(r) for r in completed[key].itertuples(index=False, name=None))
    pending["pair_first"] = [
        0 if tuple(r) in pair_keys else 1
        for r in pending[key].itertuples(index=False, name=None)
    ]
    # Urutkan per pasangan, bukan per sumber: segera bentuk pasangan CHIRPS+ERA5.
    return pending.sort_values(
        ["pair_first", "province", "start", "row_start", "source"], kind="stable"
    )


def _fetch_direct_dataframe(ee, cfg: dict, record: pd.Series,
                            grid_batch: pd.DataFrame) -> pd.DataFrame:
    """Tarik paling banyak 5000 baris via API interaktif, tanpa membuat task Drive."""
    spec = dict(SPECS[record.source])
    if record.source == "chirps":
        spec["asset"] = cfg["climate"].get("chirps_asset", spec["asset"])
        if "V3" not in spec["asset"].upper():
            raise ValueError("CHIRPS harus v3; jangan campur versi sumber")
    points = _feature_points(grid_batch, ee)
    ic = (ee.ImageCollection(spec["asset"]).filterDate(record.start, record.end)
          .select(spec["bands"]))
    days = (pd.Timestamp(record.end) - pd.Timestamp(record.start)).days
    available = int(ic.size().getInfo())
    if available != days:
        raise ValueError(f"{record['name']}: hanya {available}/{days} citra")

    def extract(image):
        day = image.date().format("YYYY-MM-dd")
        return (image.select(spec["bands"]).unmask(-9999, False)
                .sampleRegions(collection=points, properties=["cell_id"],
                               scale=spec["scale"], geometries=False, tileScale=4)
                .map(lambda ft: ft.set("date", day)))

    # computeFeatures auto-page ke DataFrame; ukuran respon dibatasi oleh caller.
    table = ic.map(extract).flatten()
    df = ee.data.computeFeatures({
        "expression": table,
        "fileFormat": "PANDAS_DATAFRAME",
        "pageSize": 1000,
    })
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Earth Engine tidak mengembalikan DataFrame")
    required = ["cell_id", "date", *spec["bands"]]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Respon GEE kurang kolom: {sorted(missing)}")
    return df[required].copy()


def ambil_langsung(cfg, *, max_batches: int, stop_on_error: bool = False):
    """Unduh+clean+Parquet otomatis ke lokal tanpa perantara Google Drive.

    Batch yang sudah INGESTED diabaikan; CSV asli di raw tidak ditimpa.
    Batch yang sedang READY/RUNNING tak disentuh agar tidak mereplikasi ekspor aktif.
    Jika GEE interaktif kehabisan waktu, catat kegagalan dan lanjutkan via Drive.
    """
    if not 1 <= max_batches <= 50:
        raise ValueError("--max-direct-batches harus 1..50")
    m = _read_manifest(cfg)
    if m.empty:
        raise ValueError("Manifest kosong; jalankan --plan terlebih dahulu")
    grid = _validate_snapshot(cfg, m)
    items = _direct_candidates(m)
    if items.empty:
        print("Tidak ada batch PLANNED/COMPLETED/DIRECT_ERROR. Cek --status.")
        return
    import ee
    ee.Initialize(project=cfg["climate"]["gee_project"])
    raw_dir = rpath(cfg["climate"]["gee_dir"]) / "batches"
    out_dir = rpath(cfg["paths"]["interim"]) / "climate_parts"
    raw_dir.mkdir(parents=True, exist_ok=True)
    total_done, total_failed = 0, 0
    consecutive_errors = 0
    for idx, record in items.iterrows():
        if total_done + total_failed >= max_batches:
            break
        if int(record.expected_rows) > DIRECT_MAX_ROWS:
            print(f"SKIP {record['name']}: {record.expected_rows} baris > "
                  f"batas direct {DIRECT_MAX_ROWS}; gunakan mode --submit/Drive.")
            continue
        sub = (grid.loc[grid.province.eq(record.province)]
               .sort_values("cell_id").iloc[int(record.row_start):int(record.row_end)])
        if len(sub) != int(record.n_cells):
            raise RuntimeError("Grid batch tidak konsisten dengan manifest")
        csv_path = raw_dir / (record["name"] + ".csv")
        dest = out_dir / record.source / (record["name"] + ".parquet")
        if dest.exists():
            # Recovery jika proses sebelumnya telah menulis Parquet sebelum status.
            _ = pd.read_parquet(dest, columns=["cell_id", "date"])
            m.at[idx, "status"] = "INGESTED"
            _save_manifest(cfg, m)
            print(f"SUDAH ADA {dest.name}; status diselaraskan tanpa request GEE")
            total_done += 1
            continue
        try:
            if not csv_path.exists():
                print(f"AMBIL LANGSUNG: {record['name']} ({record.expected_rows} baris)", flush=True)
                df = _fetch_direct_dataframe(ee, cfg, record, sub)
                # Validasi dataframe SEBELUM menulis file raw; CSV sementara
                # tidak boleh tertinggal sebagai sumber yang tampak sah.
                temporary = csv_path.with_suffix(".csv.partial")
                try:
                    df.to_csv(temporary, index=False)
                    clean = _sanitize_csv_batch(temporary, record, set(map(int, sub.cell_id)))
                    temporary.replace(csv_path)
                finally:
                    if temporary.exists():
                        temporary.unlink()
            else:
                clean = _sanitize_csv_batch(csv_path, record, set(map(int, sub.cell_id)))
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".parquet.partial")
            try:
                clean.to_parquet(tmp, index=False, compression="zstd")
                tmp.replace(dest)
            finally:
                if tmp.exists():
                    tmp.unlink()
            missing = int(clean.drop(columns=["cell_id", "date"]).isna().sum().sum())
            m.at[idx, "status"] = "INGESTED"
            m.at[idx, "error"] = f"direct_missing_numeric={missing}"
            _save_manifest(cfg, m)
            print(f"INGESTED {record['name']} | {len(clean):,} baris | null {missing}")
            total_done += 1
            consecutive_errors = 0
        except Exception as exc:
            # Jangan membuat data palsu, tidak menghapus raw lama.
            m.at[idx, "status"] = "DIRECT_ERROR"
            m.at[idx, "error"] = str(exc)[:450]
            _save_manifest(cfg, m)
            total_failed += 1
            consecutive_errors += 1
            print(f"GAGAL {record['name']}: {type(exc).__name__}: {exc}", flush=True)
            if stop_on_error:
                raise
            if consecutive_errors >= 2:
                print("STOP OTOMATIS: 2 batch berturut-turut gagal. "
                      "Jangan habiskan kuota; cek kolom error di manifest.")
                break
    print(f"SELESAI: {total_done} INGESTED baru; {total_failed} gagal. "
          f"Raw tambahan: {raw_dir} | Parquet: {out_dir}")
    if total_failed:
        print("Batch gagal dapat dicoba ulang lewat --direct atau --submit/Drive. "
              "Periksa error di manifest.")


def _final_coverage_ok(cfg, m, grid):
    """Validasi cakupan semua tanggal dan semua sel, tanpa membuka setiap CSV."""
    a, b = pd.Timestamp(cfg["dates"]["start"]), pd.Timestamp(cfg["dates"]["end"])
    # end penelitian inklusif, sedangkan batch end eksklusif
    last = b + pd.Timedelta(days=1)
    for source in SPECS:
        part = m[(m.source == source) & (m.status == "INGESTED")]
        if part.empty:
            return False
        for prov in grid.province.unique():
            total = len(grid[grid.province.eq(prov)])
            rows = part[part.province.eq(prov)]
            # Cek setiap hari untuk memastikan setiap segmen indeks [0, total)
            for day in pd.date_range(a, last - pd.Timedelta(days=1), freq="D"):
                cover = rows[(pd.to_datetime(rows.start) <= day) & (pd.to_datetime(rows.end) > day)]
                intervals = sorted(zip(cover.row_start, cover.row_end))
                cursor = 0
                for x, y in intervals:
                    if x != cursor: return False
                    cursor = int(y)
                if cursor != total: return False
    return True


def assemble_parts(cfg, *, final: bool):
    """Join dua sumber per batch dan tulis Parquet secara streaming.

    Preview: hanya pasangan yang terunduh. Final: perlu cakupan penuh tanggal & grid.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    m = _read_manifest(cfg)
    grid = _validate_snapshot(cfg, m)
    if final and not _final_coverage_ok(cfg, m, grid):
        raise RuntimeError("Cakupan historis belum lengkap; menolak membuat climate_daily.parquet final")
    ing = m[m.status.eq("INGESTED")].copy()
    if ing.empty:
        raise ValueError("Belum ada batch INGESTED")
    def key(rec):
        return (rec.province, rec.start, rec.end, int(rec.row_start), int(rec.row_end))
    chirps = {key(row): row for _, row in ing[ing.source.eq("chirps")].iterrows()}
    era5 = {key(row): row for _, row in ing[ing.source.eq("era5")].iterrows()}
    common = sorted(set(chirps) & set(era5))
    if not common:
        raise ValueError("Belum ada pasangan CHIRPS dan ERA5 dengan tanggal/grid yang sama")
    if final and (set(chirps) != set(era5)):
        raise ValueError("Mode final menolak pasangan yang tidak lengkap")
    dest = rpath(cfg["paths"]["interim"]) / ("climate_daily.parquet" if final else "climate_preview.parquet")
    temp = dest.with_suffix(dest.suffix + ".tmp")
    writer = None
    n = 0
    try:
        for k in common:
            p1 = rpath(cfg["paths"]["interim"]) / "climate_parts/chirps" / (chirps[k]["name"] + ".parquet")
            p2 = rpath(cfg["paths"]["interim"]) / "climate_parts/era5" / (era5[k]["name"] + ".parquet")
            left, right = pd.read_parquet(p1), pd.read_parquet(p2)
            merged = left.merge(right, on=["cell_id", "date"], how="outer",
                                validate="one_to_one", indicator=True)
            if not merged._merge.eq("both").all():
                raise ValueError(f"Pasangan tidak cocok untuk {k}")
            merged = merged.drop(columns="_merge").sort_values(["date", "cell_id"])
            table = pa.Table.from_pandas(merged, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temp, table.schema, compression="zstd")
            writer.write_table(table)
            n += len(merged)
        if writer is None:
            raise ValueError("Tidak ada pasangan")
    finally:
        if writer is not None:
            writer.close()
    temp.replace(dest)
    print(f"Tersimpan: {dest} | {n:,} baris | {len(common)} pasangan batch")
    if not final:
        print("INI HANYA PREVIEW. Jangan gunakan untuk training final / uji temporal.")



# ---------------------------------------------------------------- 2026: satu perintah operator, dua sumber sekaligus
# Mode ini SEPENUHNYA TERPISAH dari manifest v5/v6 dan file Riau 2025.
# Earth Engine menjalankan banyak task asinkron untuk menghindari memori penuh.
GEE2026_COLS = ["name", "province", "start", "end", "row_start", "row_end",
                "n_cells", "expected_rows", "grid_sha256", "status", "task_id", "error"]
GEE2026_BANDS = SPECS["era5"]["bands"]
GEE2026_FOLDER = "DIMASTI_KARHUTLA_2026_GABUNG"


def _2026_root(cfg):
    return rpath("data/processing")


def _2026_manifest(cfg):
    return _2026_root(cfg) / "gee_2026_manifest.csv"


def _2026_dates(cfg):
    """Tanggal inklusif dari config, tetapi batasi tepat di tahun 2026."""
    begin = max(pd.Timestamp("2026-01-01"), pd.Timestamp(cfg["dates"]["start"]))
    end = min(pd.Timestamp("2026-12-31"), pd.Timestamp(cfg["dates"]["end"]))
    if begin > end:
        raise ValueError("Tanggal config tidak mencakup tahun 2026")
    return begin, end


def _2026_plan(cfg, stop_exclusive, days_per_task=14, cells_per_task=7000):
    """Rencanakan BERSAMA CHIRPS+ERA5, tanpa mengubah manifest GEE v6 lama."""
    start, end_incl = _2026_dates(cfg)
    stop = min(pd.Timestamp(stop_exclusive), end_incl + pd.Timedelta(days=1))
    if stop <= start:
        raise ValueError("Cakupan tanggal 2026 kosong")
    grid, sha = _grid_snapshot(cfg)
    path = _2026_manifest(cfg)
    if path.exists():
        old = pd.read_csv(path, keep_default_na=False, dtype={"task_id": str})
        if set(old.columns) != set(GEE2026_COLS):
            raise ValueError("Manifest 2026 tidak sesuai skema; file dibiarkan utuh")
        if not old.grid_sha256.eq(sha).all():
            raise ValueError("grid.csv berubah; manifest 2026 tidak boleh dipakai")
    else:
        old = pd.DataFrame(columns=GEE2026_COLS)
    # Bekukan cakupan tanggal batch pertama. Saat katalog GEE bertambah,
    # jangan menggeser ujung chunk terakhir dan membuat ekspor overlap.
    if len(old):
        return old, grid, 0
    proposed = []
    # start < end eksklusif, satu task memuat CHIRPS+ERA5 pada grid/tanggal sama.
    for province in sorted(grid.province.unique()):
        chunk = grid.loc[grid.province.eq(province)].sort_values("cell_id")
        for start_date in pd.date_range(start, stop - pd.Timedelta(days=1), freq=f"{days_per_task}D"):
            end_date = min(start_date + pd.Timedelta(days=days_per_task), stop)
            for lo in range(0, len(chunk), cells_per_task):
                hi = min(lo + cells_per_task, len(chunk))
                key = hashlib.sha1(",".join(map(str, chunk.iloc[lo:hi].cell_id)).encode()).hexdigest()[:6]
                name = (f"gee26_{start_date:%Y%m%d}_{end_date:%Y%m%d}_"
                        f"{_prov_slug(province)}_c{lo:05d}_{key}")
                proposed.append(dict(name=name, province=province,
                    start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"),
                    row_start=lo, row_end=hi, n_cells=hi-lo,
                    expected_rows=(hi-lo)*(end_date-start_date).days,
                    grid_sha256=sha, status="PLANNED", task_id="", error=""))
    planned = pd.DataFrame(proposed, columns=GEE2026_COLS)
    existing = set(old.name)
    # Jika tanggal bertambah dari GEE, perluasan tanpa menimpa pekerjaan lama.
    new = planned[~planned.name.isin(existing)].copy()
    if len(old):
        for nr in new.itertuples(index=False):
            overlap = old[(old.province == nr.province) &
                          (pd.to_datetime(old.start) < pd.Timestamp(nr.end)) &
                          (pd.to_datetime(old.end) > pd.Timestamp(nr.start)) &
                          (old.row_start.astype(int) < int(nr.row_end)) &
                          (old.row_end.astype(int) > int(nr.row_start))]
            if len(overlap):
                raise RuntimeError("Cakupan GEE baru menimpa batch 2026 yang sudah tercatat. "
                                   "Pertahankan manifest lama; jangan ekspor duplikat. "
                                   "Untuk memperluas tanggal hubungi pengelola data.")
    all_jobs = pd.concat([old, new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    _2026_write_manifest(path, all_jobs)
    return all_jobs, grid, len(new)


def _2026_write_manifest(path, manifest):
    temp = path.with_suffix(".csv.partial")
    manifest[GEE2026_COLS].to_csv(temp, index=False)
    temp.replace(path)


def _2026_featurecollection(ee, cfg, rec, grid):
    sub = (grid.loc[grid.province.eq(rec.province)]
           .sort_values("cell_id").iloc[int(rec.row_start):int(rec.row_end)])
    if len(sub) != int(rec.n_cells):
        raise ValueError("Grid berbeda dari manifest")
    pts = _feature_points(sub, ee)
    ch = ee.ImageCollection(cfg["climate"]["chirps_asset"]).filterDate(rec.start, rec.end)
    er = ee.ImageCollection(SPECS["era5"]["asset"]).filterDate(rec.start, rec.end)
    num_days = (pd.Timestamp(rec.end)-pd.Timestamp(rec.start)).days
    dates = ee.List.sequence(0, num_days - 1)
    first = ee.Date(rec.start)

    def day_rows(offset):
        stamp = first.advance(ee.Number(offset), "day")
        next_day = stamp.advance(1, "day")
        ch_img = ee.Image(ch.filterDate(stamp, next_day).first()).select("precipitation").rename("precip_mm")
        er_img = ee.Image(er.filterDate(stamp, next_day).first()).select(GEE2026_BANDS)
        both = ch_img.addBands(er_img).unmask(-9999, False)
        return (both.sampleRegions(collection=pts, properties=["cell_id"],
                                   scale=5566, geometries=False, tileScale=8)
                .map(lambda ft: ft.set("date", stamp.format("YYYY-MM-dd"))))

    # List.map -> FeatureCollection of collections -> flatten; satu output tabel
    # dengan BOTH sources, bukan ekspor CSV terpisah CHIRPS/ERA5.
    return ee.FeatureCollection(dates.map(day_rows)).flatten()


def _2026_last_coverage(ee, cfg):
    """Batas bersama dua katalog. Jangan isi citra kosong dengan nol."""
    req_start, req_end = _2026_dates(cfg)
    # getInfo hanya untuk dua metadata akhir; sampling besar tetap di server.
    ch = (ee.ImageCollection(cfg["climate"]["chirps_asset"])
          .filterDate("2026-01-01", (req_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")))
    er = (ee.ImageCollection(SPECS["era5"]["asset"])
          .filterDate("2026-01-01", (req_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")))
    if int(ch.size().getInfo()) == 0 or int(er.size().getInfo()) == 0:
        raise RuntimeError("CHIRPS atau ERA5 belum menyediakan citra dalam rentang 2026")
    ch_last = pd.Timestamp(ch.sort("system:time_start", False).first().date().format("YYYY-MM-dd").getInfo())
    er_last = pd.Timestamp(er.sort("system:time_start", False).first().date().format("YYYY-MM-dd").getInfo())
    last = min(req_end, ch_last, er_last)
    start = max(req_start, pd.Timestamp("2026-01-01"))
    days = (last-start).days+1
    # Hanya terima periode kontinu. Jika ada tanggal bolong, jangan memaksa ekspor.
    for source, collection in (("CHIRPS", ch), ("ERA5", er)):
        count = int(collection.filterDate(start.strftime("%Y-%m-%d"),
                      (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")).size().getInfo())
        if count != days:
            raise RuntimeError(f"{source}: {count}/{days} hari terdeteksi; hentikan agar tidak membuat baris palsu")
    return start, last


def jalankan_gee_2026(cfg, *, max_submit=0):
    """Satu CLI merencanakan dan mengirim task gabungan 2026. max_submit=0 -> semua."""
    import ee
    ee.Initialize(project=cfg["climate"]["gee_project"])
    first, last = _2026_last_coverage(ee, cfg)
    manifest, grid, added = _2026_plan(cfg, last + pd.Timedelta(days=1))
    path = _2026_manifest(cfg)
    planned_last = pd.to_datetime(manifest["end"]).max() - pd.Timedelta(days=1)
    print(f"GEE bersama CHIRPS + ERA5: {first.date()} sampai {planned_last.date()} | {len(grid):,} grid")
    if planned_last < last:
        print("Manifest 2026 lama dipertahankan; tanggal tambahan tidak otomatis dibuat agar tidak overlap.")
    print(f"Rencana {len(manifest)} ekspor gabungan ({added} baru), bukan satu task raksasa")
    print(f"Hasil Drive: {GEE2026_FOLDER} | manifest: {path}")
    n_started = 0
    # Tidak pernah otomatis mengirim ulang READY/RUNNING/COMPLETED/SUBMITTING.
    for idx in manifest.index[manifest.status.eq("PLANNED")]:
        if max_submit and n_started >= max_submit:
            break
        rec = manifest.loc[idx]
        try:
            table = _2026_featurecollection(ee, cfg, rec, grid)
            task = ee.batch.Export.table.toDrive(
                collection=table, description=rec["name"], folder=GEE2026_FOLDER,
                fileNamePrefix=rec["name"], fileFormat="CSV",
                selectors=["cell_id", "date", "precip_mm", *GEE2026_BANDS])
            # Simpan status lebih dahulu agar terputus tidak langsung mengirim duplikat.
            manifest.at[idx, "status"] = "SUBMITTING"
            _2026_write_manifest(path, manifest)
            task.start()
            manifest.at[idx, "task_id"] = str(task.id)
            manifest.at[idx, "status"] = "READY"
            n_started += 1
            print(f"[{n_started}] TASK {rec['name']} -> {task.id}", flush=True)
        except Exception as exc:
            manifest.at[idx, "status"] = "ERROR"
            manifest.at[idx, "error"] = str(exc)[:450]
            print(f"STOP: gagal mengirim {rec['name']}: {exc}")
            _2026_write_manifest(path, manifest)
            raise
        _2026_write_manifest(path, manifest)
    print(f"Berhasil submit {n_started} task. Earth Engine memprosesnya di antrean server.")
    print("Jika terlalu banyak task, gunakan --limit-task-2026 5 dahulu; lalu ulang perintah yang sama.")
    print("Sesudah semua COMPLETED dan CSV tersinkron di PC, gunakan --merge-2026.")



def status_gee_2026(cfg):
    """Status pekerjaan 2026, terpisah dari manifest 2025; tanpa mengirim ulang."""
    p = _2026_manifest(cfg)
    if not p.exists():
        print("Belum ada manifest 2026. Jalankan --run-2026.")
        return
    m = pd.read_csv(p, keep_default_na=False)
    active = m.index[m.status.isin(["READY", "RUNNING"]) & m.task_id.ne("")]
    if len(active):
        import ee
        ee.Initialize(project=cfg["climate"]["gee_project"])
        for batch in range(0, len(active), 50):
            index = active[batch:batch + 50]
            tasks = {s.get("id"): s for s in ee.data.getTaskStatus(m.loc[index, "task_id"].tolist())}
            for idx in index:
                state = tasks.get(m.at[idx, "task_id"], {})
                if state.get("state"):
                    m.at[idx, "status"] = state["state"]
                    m.at[idx, "error"] = str(state.get("error_message", ""))[:400]
        _2026_write_manifest(p, m)
    print(m.status.value_counts().to_string())
    failed = m[m.status.isin(["FAILED", "CANCELLED", "ERROR"])]
    if not failed.empty:
        print("Task gagal (tidak di-retry otomatis):")
        print(failed[["name", "error"]].head(5).to_string(index=False))


def _2026_sanitize_csv(csv_path, rec, grid_ids):
    """Jangan diam-diam melepas nilai hilang atau baris duplikat."""
    cols = ["cell_id", "date", "precip_mm", *GEE2026_BANDS]
    raw = pd.read_csv(csv_path)
    if not set(cols).issubset(raw.columns):
        raise ValueError(f"{csv_path.name}: kolom kurang {sorted(set(cols)-set(raw.columns))}")
    if len(raw) != int(rec.expected_rows):
        raise ValueError(f"{csv_path.name}: {len(raw)} baris, target {rec.expected_rows}")
    raw = raw[cols].copy()
    raw["cell_id"] = pd.to_numeric(raw.cell_id, errors="raise").astype("int64")
    raw["date"] = pd.to_datetime(raw.date, errors="raise")
    if raw.duplicated(["cell_id", "date"]).any():
        raise ValueError(f"{csv_path.name}: duplikasi sel/tanggal")
    if set(raw.cell_id) != set(grid_ids):
        raise ValueError(f"{csv_path.name}: cell_id tidak cocok dengan grid")
    required_days = (pd.Timestamp(rec.end) - pd.Timestamp(rec.start)).days
    if not raw.groupby("cell_id").size().eq(required_days).all():
        raise ValueError(f"{csv_path.name}: tidak semua grid memiliki seluruh tanggal")
    if not raw.date.between(pd.Timestamp(rec.start), pd.Timestamp(rec.end) - pd.Timedelta(days=1)).all():
        raise ValueError(f"{csv_path.name}: tanggal di luar batch")
    for c in cols[2:]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce").replace(-9999, np.nan).astype("float32")
    if raw.precip_mm.dropna().lt(0).any():
        raise ValueError("Curah hujan negatif dari CHIRPS")
    return derive_era5(raw).merge(raw[["cell_id", "date", "precip_mm"]], on=["cell_id", "date"], validate="one_to_one")


def _2026_hot_labels(cfg):
    """Label y_7d FIRMS: gunakan masa depan t+1..t+7 saja, bukan t."""
    path = rpath(cfg["paths"]["interim"]) / "hotspots_daily.parquet"
    if not path.exists():
        raise FileNotFoundError(f"File FIRMS belum ada: {path}")
    fir = pd.read_parquet(path)
    fir["date"] = pd.to_datetime(fir.date)
    fir = fir.loc[(fir.date.dt.year == 2026) & (fir.n_hotspot > 0)].copy()
    if fir.duplicated(["cell_id", "date"]).any():
        raise ValueError("FIRMS berisi duplikat cell_id/date")
    label = pd.concat([fir[["cell_id", "date", "n_hotspot"]].assign(
        date=lambda d, t=i: d.date - pd.Timedelta(days=t)) for i in range(1, 8)], ignore_index=True)
    label = (label.groupby(["cell_id", "date"], as_index=False)["n_hotspot"]
             .sum().rename(columns={"n_hotspot": "n_hot_next_7d"}))
    return fir, label


def _2026_split(day):
    """Tidak ada target 7-hari yang menyeberang train/validation."""
    if day <= pd.Timestamp("2026-07-24"):
        return "train"
    if pd.Timestamp("2026-08-01") <= day <= pd.Timestamp("2026-08-24"):
        return "validation"
    if day >= pd.Timestamp("2026-09-01"):
        return "test"
    return "gap_7_hari"


def gabung_gee_firms_2026(cfg, *, drive_dir=None):
    """Baca CSV GEE 2026 tersinkron, validasi, join FIRMS, tulis Parquet incremental.

    HANYA buat file final ketika seluruh task dalam manifest dan seluruh CSV ada.
    Tidak pernah menimpa data RAW/interim dan manifest GEE v5/v6.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = _2026_manifest(cfg)
    if not path.exists():
        raise FileNotFoundError("Manifest 2026 belum ada. Jalankan --run-2026 dulu.")
    m = pd.read_csv(path, keep_default_na=False)
    grid, sha = _grid_snapshot(cfg)
    if not m.grid_sha256.eq(sha).all():
        raise ValueError("grid.csv tidak sama dengan saat task dibuat")
    folder = Path(drive_dir) if drive_dir else (_2026_root(cfg) / "gee_2026_downloads")
    missing = [r.name for r in m.itertuples(index=False) if not (folder / (r.name + ".csv")).exists()]
    if missing:
        print(f"Belum tersedia {len(missing)}/{len(m)} CSV GEE di {folder}")
        print("Contoh file belum ada:", ", ".join(missing[:4]))
        raise FileNotFoundError("Tunggu semua task COMPLETED; sinkronkan/copy semua CSV ke folder tersebut")
    last_firms = pd.Timestamp(json.loads((rpath(cfg["paths"]["interim"]) / "firms_meta.json").read_text())["last_date"])
    valid_last = min(last_firms-pd.Timedelta(days=7), pd.Timestamp(cfg["dates"]["end"]))
    if valid_last < pd.Timestamp("2026-01-01"):
        raise ValueError("Label FIRMS t+7 tahun 2026 belum tersedia")
    fir, labels = _2026_hot_labels(cfg)
    # Semua file diperiksa dulu agar output lama tidak diganti oleh hasil parsial.
    out = _2026_root(cfg) / "dataset_gabungan_2026.parquet"
    temp = out.with_suffix(".parquet.partial")
    writer = None
    total, missing_weather = 0, 0
    schema_cols = ["cell_id", "date", "province", "lon", "lat", "precip_mm", "t2m_c",
                   "rh", "sm1", "wind_ms", "n_hotspot", "frp_sum", "n_hot_next_7d", "y_7d", "split"]
    try:
        for rec in m.itertuples(index=False):
            sub = (grid.loc[grid.province.eq(rec.province)].sort_values("cell_id")
                   .iloc[int(rec.row_start):int(rec.row_end)])
            data = _2026_sanitize_csv(folder / (rec.name + ".csv"), rec, set(sub.cell_id))
            data = data.loc[data.date.le(valid_last)].copy()
            if data.empty:
                continue
            data = data.merge(sub[["cell_id", "province", "lon", "lat"]], on="cell_id", validate="many_to_one")
            data = data.merge(fir[["cell_id", "date", "n_hotspot", "frp_sum"]], how="left",
                              on=["cell_id", "date"], validate="one_to_one")
            data = data.merge(labels, how="left", on=["cell_id", "date"], validate="one_to_one")
            data["n_hotspot"] = data.n_hotspot.fillna(0).astype("int32")
            data["frp_sum"] = data.frp_sum.fillna(0).astype("float32")
            data["n_hot_next_7d"] = data.n_hot_next_7d.fillna(0).astype("int32")
            data["y_7d"] = data.n_hot_next_7d.gt(0).astype("int8")
            data["split"] = data.date.map(_2026_split).astype("string")
            data = data[schema_cols]
            data = data.loc[data.split.ne("gap_7_hari")]
            if data.empty:
                continue
            table = pa.Table.from_pandas(data, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temp, table.schema, compression="zstd")
            writer.write_table(table)
            total += len(data)
            missing_weather += int(data[["precip_mm", "t2m_c", "rh", "sm1", "wind_ms"]].isna().sum().sum())
        if writer is None:
            raise ValueError("Tidak ada baris valid yang dapat digabung")
    finally:
        if writer is not None:
            writer.close()
    temp.replace(out)
    report = {"year": 2026, "rows": total, "grids": len(grid), "task_files": len(m),
              "label_last_date": str(valid_last.date()), "weather_missing_values": missing_weather,
              "note": "Keterangan split memakai jeda 7 hari; NaN cuaca dipertahankan, bukan nol",
              "dataset": str(out)}
    out.with_name("QC_gabungan_2026.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"SELESAI: {total:,} observasi FIRMS + CHIRPS + ERA5 2026 → {out}")
    print(f"Missing cuaca = {missing_weather:,}; lihat QC_gabungan_2026.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Iklim DIMAS-TI: batch terencana + mode lama")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--export", action="store_true", help="SATU batch uji lama (maks 200 grid/7 hari)")
    mode.add_argument("--plan", action="store_true", help="buat manifest; tidak menjalankan task")
    mode.add_argument("--submit", action="store_true", help="kirim task dari manifest secara terbatas")
    mode.add_argument("--status", action="store_true", help="perbarui status task dari Earth Engine")
    mode.add_argument("--ingest", action="store_true", help="validasi unduhan, simpan Parquet per batch")
    mode.add_argument("--assemble", action="store_true", help="gabungkan pasangan CHIRPS/ERA5 terproses")
    mode.add_argument("--run-2026", action="store_true", help="SATU perintah kirim gabungan CHIRPS+ERA5 enam provinsi tahun 2026")
    mode.add_argument("--status-2026", action="store_true", help="cek seluruh task GEE 2026 sekali jalan")
    mode.add_argument("--merge-2026", action="store_true", help="gabungkan CSV GEE 2026 dengan FIRMS ke data/processing")
    mode.add_argument("--direct", action="store_true", help="AMBIL GEE langsung ke Python + bersihkan + Parquet; tanpa Drive")
    ap.add_argument("--limit-task-2026", type=int, default=0, help="0=semua; batasi jika ingin menguji 1..5 task")
    ap.add_argument("--drive-dir", help="folder hasil ekspor GEE 2026 tersinkron di komputer")
    ap.add_argument("--final", action="store_true", help="hanya bersama --assemble; butuh cakupan lengkap")
    ap.add_argument("--source", choices=[*SPECS, "both"])
    ap.add_argument("--start", help="YYYY-MM-DD, inklusif")
    ap.add_argument("--end", help="YYYY-MM-DD, eksklusif")
    ap.add_argument("--province", help="Nama persis grid.csv, atau 'all' bersama --plan")
    ap.add_argument("--max-cells", type=int, help="mode --export lama (<=200)")
    ap.add_argument("--cells-per-batch", type=int, default=500, help="mode --plan; default 500")
    ap.add_argument("--days-per-batch", type=int, default=7, help="mode --plan; default 7")
    ap.add_argument("--max-new-tasks", type=int, default=1, help="mode --submit; default 1")
    ap.add_argument("--max-active-tasks", type=int, default=2, help="mode --submit; default 2")
    ap.add_argument("--max-files", type=int, default=10, help="mode --ingest; default 10")
    ap.add_argument("--max-direct-batches", type=int, default=2, help="mode --direct, 1..50; default 2")
    ap.add_argument("--stop-on-error", action="store_true", help="mode --direct; hentikan bila ada batch gagal")
    a = ap.parse_args()
    cfg = load_config()
    if a.run_2026:
        jalankan_gee_2026(cfg, max_submit=a.limit_task_2026)
    elif a.merge_2026:
        gabung_gee_firms_2026(cfg, drive_dir=a.drive_dir)
    elif a.status_2026:
        status_gee_2026(cfg)
    elif a.export:
        if any(v is None for v in (a.source, a.start, a.end, a.province, a.max_cells)):
            ap.error("--export memerlukan --source --start --end --province --max-cells")
        if a.source == "both":
            ap.error("--export hanya satu sumber")
        export_gee(cfg, source=a.source, start=a.start, end=a.end,
                   province=a.province, max_cells=a.max_cells)
    elif a.plan:
        if any(v is None for v in (a.source, a.start, a.end, a.province)):
            ap.error("--plan memerlukan --source --start --end --province")
        buat_rencana(cfg, source=a.source, start=a.start, end=a.end,
                     province=a.province, cells_per_batch=a.cells_per_batch,
                     days_per_batch=a.days_per_batch)
    elif a.submit:
        kirim_batch(cfg, max_new_tasks=a.max_new_tasks,
                    max_active_tasks=a.max_active_tasks)
    elif a.status:
        cek_status(cfg)
    elif a.direct:
        ambil_langsung(cfg, max_batches=a.max_direct_batches,
                       stop_on_error=a.stop_on_error)
    elif a.ingest:
        ingest_batches(cfg, max_files=a.max_files)
    elif a.assemble:
        assemble_parts(cfg, final=a.final)
    elif a.final:
        ap.error("--final harus bersama --assemble")
    else:
        # Perilaku lama untuk data CSV lengkap di data/raw/gee/.
        main(cfg)
