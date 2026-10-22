# Model Risiko Hotspot 2026 (Orang C)

Status: RESEARCH CANDIDATE (belum tervalidasi operasional)

## Input
Gunakan urutan fitur persis seperti dalam model_spec_2026.json.
Fitur rolling harus dihitung dari tanggal sebelum prediksi.
Jangan masukkan y_7d atau n_hot_next_7d sebagai fitur.

## Output
predict_proba(X)[:, 1] menghasilkan probabilitas model
untuk label hotspot tujuh hari berikutnya.

## Threshold
threshold_f1: prioritas keseimbangan precision dan recall.
threshold_recall70: alternatif sensitivitas lebih tinggi,
dengan risiko false alarm lebih banyak.

## Penting
Model ini belum membuktikan ketersediaan data cuaca
secara real-time, dan bukan alat konfirmasi kebakaran.
