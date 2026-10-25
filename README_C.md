# Orang C: dashboard Karhutla Lab

```
src/app/risk_levels.py     skor -> Rendah/Sedang/Tinggi/Sangat Tinggi (ambang dari model_spec_2026.json)
src/app/predict_api.py     PredictAPI: muat model B, prediksi per tanggal, kurva "bagaimana jika"
src/app/karhutla_lab.py    verify | build | serve  (situs statis di dist/)
src/app/web/               HTML/JS/CSS + Leaflet (tanpa CDN, tanpa build tool)
```

## Alur kerja
```bash
pip install pandas numpy pyarrow pyyaml joblib lightgbm
python -m src.app.karhutla_lab verify          # 1. cocokkan dengan hasil Orang B (stage22, 30 Sep 2026)
python -m src.app.karhutla_lab build --days 30 # 2. tulis dist/
python -m src.app.karhutla_lab serve           # 3. pratinjau di http://localhost:8000
```
`verify` menghitung ulang skor 30 Sep 2026 dan membandingkan jumlah sel, jumlah alert (skor >= ambang F1),
dan rata-rata skor dengan `reports/orang_b/stage22_visual_summary_2026.json`. Jika tidak cocok, jangan publikasikan.
`build --demo` membuat data SINTETIS (banner merah) hanya untuk melihat tampilan.

## Artefak yang dibaca (semua punya nilai bawaan; ubah lewat blok `app:` di config.yaml bila perlu)
| Kunci | Bawaan |
|---|---|
| model_path / model_meta_path | models_final/lightgbm_candidate_2026.joblib / model_spec_2026.json |
| dataset_path | data/processed/dataset_fitur_2026_v3.parquet |
| grid_path | data/interim/grid.csv (lon, lat, province bila tidak ada di dataset) |
| metrics_path | reports/orang_b/stage19_evaluation_2026.json |
| figures_dir | figures/orang_b (kurva PR, confusion matrix, SHAP, kepentingan fitur) |
| n_days | 30 (jumlah tanggal terakhir yang diekspor) |
| require_complete_features | true (sel dengan fitur NaN tidak diberi skor) |
| very_high_threshold | 0.40 (SEMENTARA; belum divalidasi Orang B) |

## Tingkat risiko
Sedang >= ambang recall70, Tinggi >= ambang F1 (sama dengan definisi alert B), Sangat Tinggi >= `very_high_threshold`.
Tab "Tentang" menampilkan tingkat ketepatan tiap level dari data di luar pelatihan; gunakan itu untuk menetapkan
`very_high_threshold` bersama Orang B. Skor model BELUM dikalibrasi, jadi ditampilkan sebagai skor, bukan peluang.

## Deploy ke Netlify
Situs statis: unggah folder `dist/` lewat app.netlify.com/drop, atau `netlify deploy --prod --dir=dist`.
Dengan repo Git: commit `dist/` dan pakai `netlify.toml` (publish = "dist"). Setiap data/model berubah, `build` lagi lalu deploy ulang.
Ukuran sekitar 35 MB untuk 30 hari dan 22.553 sel; browser hanya mengunduh ±60 KB gzip per tanggal.

## Batasan yang tampil di dashboard
Model berstatus RESEARCH_CANDIDATE: uji September sudah pernah terekspos, ketersediaan cuaca real-time belum
diverifikasi, baru diuji tahun 2026, dan memprediksi hotspot (bukan konfirmasi kebakaran). Tanggal pelatihan
ditandai "in-sample". Simulasi "bagaimana jika" adalah pendekatan (efek rata-rata model, dijumlahkan per fitur;
jendela 7 dan 14 hari digeser bersama).

## Pengujian
`python -m pytest tests/test_app.py` (butuh lightgbm; 2 kelompok tes JavaScript butuh `node`, dan jsdom lewat `npm i jsdom`).
