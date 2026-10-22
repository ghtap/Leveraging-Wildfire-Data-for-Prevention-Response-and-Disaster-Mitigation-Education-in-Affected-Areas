# PETUNJUK TERBARU ORANG A — TAHUN 2026 (BACA BAGIAN INI DAHULU)

> Bagian di bawah judul **Arsip README awal** merupakan dokumentasi lama dan **tidak**
> menjadi perintah untuk pengambilan tahun 2026. Jalur baru tidak menggunakan manifest Riau 2025.

Proyek asli Arya, seluruh `data/raw/` dan `data/interim/` dipertahankan **tanpa menghapus atau
menimpa file mentah**. Bagian yang berubah dari ZIP masukan hanya isi `config.yaml`,
`src/data/fetch_climate.py`, `requirements.txt`, dua notebook kosong, `README.md`,
dan `CHANGELOG_PERBAIKAN.md`. Semua nama dan jalur file lama dipertahankan.

## Cakupan dan batasan yang sebenarnya

- Tahun **2026 saja**. Mulai `2026-01-01`; akhir nominal `2026-10-16`, tetapi **GEE
  otomatis berhenti pada tanggal terakhir yang tersedia BERSAMA** di CHIRPS v3 dan ERA5-Land.
  Ini tidak berarti cuaca tersedia sampai 16 Oktober; lihat log terminal.
- Enam provinsi, seluruh grid `data/interim/grid.csv` (22.553 sel pada ZIP masukan).
  **Bukan sampel**. CHIRPS beresolusi asli ~0,05°, ERA5-Land ~11 km; grid 0,045°
  tidak menciptakan ketelitian meteorologi baru.
- `--run-2026` adalah **SATU perintah untuk pengguna**, tetapi di belakang layar
  mengantrekan banyak ekspor CHIRPS+ERA5 **DALAM FILE YANG SAMA PER BATCH**, bukan
  satu permintaan GEE raksasa. Proses Cloud bisa berlangsung lama atau gagal oleh kuota.
- Cakupan data dimulai Jan 2026. Fitur iklim historis 30/60/90 hari di awal tahun
  **tidak otomatis ada** tanpa data sebelum Jan 2026; CSV ini adalah data harian asli.

## Tiga perintah terminal (VS Code, terminal di folder `karhutla`)

**Persiapan instalasi paket Python** (sekali):

```powershell
python -m pip install -r requirements.txt
```

**Langkah 1 — satu pemanggilan mengirim seluruh ekspor 2026 ke Earth Engine:**

```powershell
python -m src.data.fetch_climate --run-2026
```

Program mengecek tanggal terakhir CHIRPS+ERA5, membuat `data/processing/gee_2026_manifest.csv`,
dan mendaftarkan ekspor **gabungan kedua sumber** untuk tiap kelompok provinsi/tanggal.
Ini **asinkron**: selesai mengirim bukan berarti semua CSV telah selesai dibuat.
Jika proses terputus, ulang perintah yang sama; tugas yang sudah tercatat tidak
akan diajukan ulang. Untuk coba lima task saja, tambahkan `--limit-task-2026 5`.

**Langkah 2 — sekali cek status semua pekerjaan:**

```powershell
python -m src.data.fetch_climate --status-2026
```

Semua task harus `COMPLETED`. CSV akan tersedia di **Google Drive** folder
`DIMASTI_KARHUTLA_2026_GABUNG`. Agar **tidak download CSV satu-satu**, sinkronkan
folder itu menggunakan **Google Drive for desktop**, atau unduh semua CSV sebagai
satu arsip lewat Drive dan ekstrak ke `karhutla/data/processing/gee_2026_downloads/`.
Jangan mengubah nama CSV. Proyek tidak memiliki izin untuk menyinkronkan Drive
langsung tanpa autentikasi/perangkat lunak Google Drive.

**Langkah 3 — satu perintah gabungkan CHIRPS, ERA5, FIRMS + label t+7:**

Jika CSV ada di `data/processing/gee_2026_downloads/`:

```powershell
python -m src.data.fetch_climate --merge-2026
```

Jika sudah pakai Drive for desktop, baca langsung folder yang tersinkron (ganti
lokasi dengan lokasi aktual di Windows milikmu):

```powershell
python -m src.data.fetch_climate --merge-2026 --drive-dir "G:\\My Drive\\DIMASTI_KARHUTLA_2026_GABUNG"
```

Hasil dari `--merge-2026`:

- `data/processing/dataset_gabungan_2026.parquet` : satu baris = satu grid-hari;
  CHIRPS `precip_mm`; ERA5 `t2m_c`, `rh`, `sm1`, `wind_ms`;
  FIRMS `n_hotspot`, `frp_sum`; label `y_7d`, `n_hot_next_7d`; `split`.
- `data/processing/QC_gabungan_2026.json`: cakupan, nilai kosong, metadata.
- `data/processing/gee_2026_manifest.csv`: status ekspor 2026.

**Pengaman**: jika satu saja CSV hilang / jumlah baris tidak cocok / ID ganda,
perintah menolak membuat hasil final. Angka `-9999` dari GEE adalah NaN; tidak
pernah diganti nol. Nilai nol untuk FIRMS hanya berlaku bila tidak ada deteksi
pada grid dan tanggal yang masih berada dalam rentang data FIRMS yang tersedia.
Batas label 2026 sesuai tanggal terakhir FIRMS asli minus 7 hari; split memakai
train hingga 24 Juli, validasi 1–24 Agustus, test mulai 1 September, dengan jeda
7 hari agar target antarkelompok tidak saling menyeberang.

**Notebook yang diisi**: `notebooks/EDA_Cleaning.ipynb` dan
`notebooks/Validasi_Grid.ipynb`. Sebelum ekspor selesai, notebook hanya
menganalisis FIRMS+grid yang sungguhan tersedia dan memberi pesan data iklim belum ada.

## Arsip README awal (panduan versi sebelum pemfokusan 2026)

# karhutla: prediksi risiko karhutla 7 hari ke depan

Bagian ini berisi pekerjaan **Orang A (data)**: `src/data/` dan `config.yaml`.

## Urutan menjalankan
```bash
pip install -r requirements.txt
python -m src.data.build_grid                       # grid.csv, grid.geojson
python -m src.data.fetch_firms                      # firms_raw.csv -> hotspots_daily.parquet
python -m src.data.fetch_climate --export           # mulai task GEE (CHIRPS, ERA5-Land)
python -m src.data.fetch_static  --export           # mulai task GEE (elevasi, lahan, built-up)
# jalankan task di code.earthengine.google.com/tasks, unduh CSV/TIF dari Drive ke data/raw/gee/
python -m src.data.make_dataset --steps climate,static,features
```
Prasyarat file: `data/raw/prov_boundary.geojson`, `firms_raw.csv`, `oni.csv`, `dmi_raw.txt`,
dan (opsional) `peat/gambut.gpkg`. Isi `climate.gee_project` dan `climate.gee_folder` di `config.yaml`.
Memori yang disarankan untuk dataset penuh: 8 GB atau lebih (Colab biasa cukup).

## Kontrak untuk Orang B dan C
`data/processed/dataset_v1.parquet` (satu baris = satu sel-hari), dengan `dataset_v1_schema.json`
(`meta_cols`, `feature_cols`, `label_cols`) dan `dataset_v1_report.json` (QC).
- Label: `y_7d` = 1 bila ada hotspot pada hari t+1..t+7; `n_hot_next_7d` = jumlahnya.
- Fitur hari t hanya memakai data sampai t (iklim dikurangi `obs_latency_days`); ONI/DMI dilag 2 bulan.
- Kolom `lon`, `lat`, `cell_id`, `province` adalah metadata untuk split spasial, bukan fitur.
- Baris tanpa label penuh (7 hari terakhir) dan masa pemanasan jendela terpanjang dibuang.

## Keputusan desain (ubah di config.yaml bila tidak cocok)
| Hal | Pilihan | Alasan |
|---|---|---|
| Grid | 0.1 derajat (~11 km), ~4.600 sel | sama dengan resolusi native ERA5-Land |
| Sensor hotspot | VIIRS S-NPP + NOAA-20 | tersedia sepanjang 2018 sampai kini; NOAA-21 dikeluarkan agar deret konsisten |
| SP vs NRT | SP diutamakan, NRT hanya jika SP belum ada | hindari hitung ganda; periode NRT tetap perlu dicatat |
| Sel tanpa data iklim | dibuang (dilaporkan di report) | sel pesisir yang piksel ERA5-Land-nya laut |

## Belum teruji di sini (tidak ada akses GEE/NASA dari sandbox)
`export_gee`, `export_gee_static`, dan `download` ditulis hati-hati tetapi belum dijalankan.
Semua logika lain diuji dengan data sintetis (`pytest tests`, 13 tes). Setelah ekspor GEE pertama:
1. cek `cells_dropped_no_climate` di report, dan `missing_pct_last_year` (ERA5-Land tertinggal beberapa bulan);
2. cek `dist_built_km` pada sel dekat kota (harus kecil) dan rawa terpencil (harus besar);
3. bandingkan jumlah hotspot per bulan SP vs NRT untuk memastikan tidak ada celah.

## Tambahan v2: CHIRPS v3, MOD13Q1, Dynamic World
- **CHIRPS**: `climate.chirps_asset` = `UCSB-CHC/CHIRPS/V3/DAILY_SAT` (band sama, drop-in). Katalog menyebut data sampai 2026-08-31, jadi Sep–Okt 2026 NaN.
- **MOD13Q1** (`modisvi_*.csv`): `ndvi_last`, `evi_last`, `ndvi_chg`. Komposit baru dianggap tersedia `vi_availability_days` setelah tanggal awalnya.
- **Dynamic World** (`dw_yearly.csv`): `dw_*` = rata-rata probabilitas kelas tahun **sebelumnya**.
- Ekspor: `fetch_climate --export` dan `fetch_static --export` sekarang juga memulai task VI dan DW.
