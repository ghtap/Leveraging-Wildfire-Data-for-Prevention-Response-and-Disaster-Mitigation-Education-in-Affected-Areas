"""Pilot CHIRPS (v2/v3) dengan sampling titik piksel untuk proyek DIMAS-TI.

Lokasi pengganti: karhutla/src/data/pilot_gee.py
Jalankan dari karhutla: python -m src.data.pilot_gee

Membaca config.yaml dan data/interim/grid.csv saja. Tidak mengubah data/raw/.
Hanya membuat SATU task GEE untuk 12 grid jika paling sedikit 10 nilai hujan valid.
ERA5-Land tidak diekspor ulang karena pilot sebelumnya sudah tersedia.

PENTING: sampling titik tengah grid di sini HANYA untuk pengujian sumber.
Untuk dataset final, metode agregasi grid harus ditetapkan dan divalidasi terpisah.
"""

import ee
import pandas as pd

from .common import load_config, rpath

TANGGAL = "2025-08-01"
TANGGAL_AKHIR_EKSKLUSIF = "2025-08-02"
FOLDER_PILOT = "DIMASTI_KARHUTLA_PILOT"
MIN_VALID = 10
SKALA_CHIRPS_M = 5566

# Tidak menggabungkan v2 dan v3 dalam satu seri waktu: ini hanya perbandingan pilot.
CHIRPS = {
    "v3": "UCSB-CHC/CHIRPS/V3/DAILY_SAT",
    "v2": "UCSB-CHG/CHIRPS/DAILY",
}


def pilih_sampel(grid: pd.DataFrame) -> pd.DataFrame:
    """Pilih dua grid dekat pusat distribusi grid tiap provinsi, bukan dua baris awal."""
    required = {"cell_id", "lon", "lat", "province"}
    if not required.issubset(grid.columns):
        raise ValueError(f"Kolom grid kurang: {sorted(required - set(grid.columns))}")
    if grid.empty or grid[list(required)].isna().any().any():
        raise ValueError("Grid kosong atau mempunyai nilai kosong pada kolom wajib")
    if grid["cell_id"].duplicated().any():
        raise ValueError("cell_id ganda pada grid.csv")
    if grid["province"].nunique() != 6:
        raise ValueError(f"Harus 6 provinsi, ditemukan {grid['province'].nunique()}")

    sampel = []
    for province, kelompok in grid.groupby("province", sort=True):
        tengah_lon = kelompok["lon"].median()
        tengah_lat = kelompok["lat"].median()
        # Jarak derajat hanya digunakan untuk memilih pilot kecil, bukan hitung jarak geodesi.
        jarak2 = (kelompok["lon"] - tengah_lon) ** 2 + (kelompok["lat"] - tengah_lat) ** 2
        terpilih = kelompok.assign(_jarak2=jarak2).nsmallest(2, "_jarak2")
        if len(terpilih) != 2:
            raise ValueError(f"Provinsi {province} memiliki kurang dari dua grid")
        sampel.append(terpilih.drop(columns="_jarak2"))
    return pd.concat(sampel, ignore_index=True)


def baca_hujan(asset: str, titik: ee.FeatureCollection):
    koleksi = (
        ee.ImageCollection(asset)
        .filterDate(TANGGAL, TANGGAL_AKHIR_EKSKLUSIF)
        .select(["precipitation"])
    )
    jumlah_citra = int(koleksi.size().getInfo())
    if jumlah_citra != 1:
        raise RuntimeError(f"{asset}: perlu 1 citra, ditemukan {jumlah_citra}")

    # sampleRegions secara eksplisit membaca piksel pada geometri titik.
    # reduceRegions pada titik tanpa area dapat menghasilkan keluaran null.
    # Piksel yang termask otomatis tidak ikut dikeluarkan; jumlahnya diuji di bawah.
    tabel = ee.Image(koleksi.first()).sampleRegions(
        collection=titik,
        properties=["cell_id", "province"],
        scale=SKALA_CHIRPS_M,
        geometries=False,
        tileScale=2,
    ).map(lambda f: f.set("date", TANGGAL))
    terisi = tabel.filter(ee.Filter.notNull(["precipitation"]))
    jumlah_valid = int(terisi.size().getInfo())
    minimum = terisi.aggregate_min("precipitation").getInfo() if jumlah_valid else None
    maksimum = terisi.aggregate_max("precipitation").getInfo() if jumlah_valid else None
    return tabel, jumlah_valid, minimum, maksimum


def main():
    cfg = load_config()
    project_id = cfg["climate"]["gee_project"]
    if not project_id or project_id == "NAMA_PROJECT_GCP":
        raise ValueError("Isi climate.gee_project dengan Project ID yang benar")
    ee.Initialize(project=project_id)

    grid = pd.read_csv(rpath(cfg["paths"]["interim"]) / "grid.csv")
    sampel = pilih_sampel(grid)
    titik = ee.FeatureCollection([
        ee.Feature(
            ee.Geometry.Point([float(row.lon), float(row.lat)]),
            {"cell_id": int(row.cell_id), "province": row.province},
        )
        for row in sampel.itertuples(index=False)
    ])

    print(f"Project: {project_id}")
    print(f"Tanggal: {TANGGAL}; grid pilot: {len(sampel)}")
    print("Metode uji: sampel titik tengah, 2 grid terpusat per provinsi")

    hasil = {}
    for versi, asset in CHIRPS.items():
        tabel, n_valid, min_hujan, max_hujan = baca_hujan(asset, titik)
        print(f"CHIRPS {versi}: {n_valid}/12 nilai tersedia; rentang mm/hari: {min_hujan} .. {max_hujan}")
        hasil[versi] = (tabel, n_valid)

    # Utamakan v3 jika cukup lengkap; bila tidak, bandingkan v2 secara terbuka.
    versi = "v3" if hasil["v3"][1] >= MIN_VALID else "v2"
    tabel, n_valid = hasil[versi]
    if n_valid < MIN_VALID:
        raise RuntimeError(
            f"Hanya {n_valid}/12 nilai dari versi terbaik yang diuji. "
            "Ekspor dibatalkan: periksa lokasi grid dan mask citra dahulu."
        )

    task_name = f"pilot_chirps_{versi}_20250801_sample"
    task = ee.batch.Export.table.toDrive(
        collection=tabel,
        description=task_name,
        folder=FOLDER_PILOT,
        fileNamePrefix=task_name,
        fileFormat="CSV",
        selectors=["cell_id", "date", "precipitation"],
    )
    task.start()
    print(f"TASK DIMULAI: {task_name} | ID: {task.id}")
    print(f"Sumber terpilih untuk PILOT: CHIRPS {versi}. Jangan campur versi dalam dataset final.")
    print("Tunggu task selesai, lalu unduh CSV dan cek nilai kosong. Jangan jalankan ulang task.")


if __name__ == "__main__":
    main()
