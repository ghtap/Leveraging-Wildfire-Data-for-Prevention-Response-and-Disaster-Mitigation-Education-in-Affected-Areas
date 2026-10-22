

## 20 Oktober 2026 — Fokus Orang A 2026 + ekspor kedua GEE sekaligus

- `config.yaml`: periode 2026-01-01 s.d. 2026-10-16; `test_start=2026-09-01`; `paths.processed=data/processing`.
- `src/data/fetch_climate.py`: fungsi `_2026_*`, `jalankan_gee_2026`, `status_gee_2026`, `gabung_gee_firms_2026`; opsi `--run-2026`, `--status-2026`, `--merge-2026`, `--drive-dir`, `--limit-task-2026`. Kode `--direct`, `--plan`, `--submit`, `--ingest` lama tetap tersedia.
- `notebooks/EDA_Cleaning.ipynb` dan `notebooks/Validasi_Grid.ipynb`: diisi dari kosong dengan kode analisis tahun 2026.
- `requirements.txt`: tambah `matplotlib` untuk visualisasi notebook.
- `README.md`: tambahkan panduan 2026 di awal; panduan lama tetap diarsipkan.
- **Tidak ada file lama dihapus/diubah namanya**. Semua data raw dan interim milik Arya disimpan apa adanya.
- Eksekusi GEE asli tidak dilakukan dalam lingkungan pembuatan ZIP karena tidak tersedia akun/akses GEE pengguna.
