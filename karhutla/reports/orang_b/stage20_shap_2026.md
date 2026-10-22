# Interpretasi Model LightGBM - DIMAS-TI 2026

## Metodologi
- Sampel validation: 6,000 observasi
- Metode: LightGBM native SHAP contributions
- Skala kontribusi: log-odds, bukan probabilitas

## Lima fitur teratas
1. month_2026: mean |SHAP| = 1.86176
2. hotspot_prev14: mean |SHAP| = 0.34464
3. soil_prev7_mean: mean |SHAP| = 0.25743
4. rh_prev7_mean: mean |SHAP| = 0.22104
5. hotspot_prev7: mean |SHAP| = 0.16804

## Contoh prediksi lokal
- Grid: 199306278
- Tanggal: 2026-08-08 00:00:00
- Provinsi: Riau
- Probabilitas model: 0.7868

## Interpretasi
SHAP positif mendorong log-odds hotspot naik; SHAP negatif mendorongnya turun.

SHAP menjelaskan perilaku model, bukan membuktikan penyebab kebakaran.

## Keterbatasan
- SHAP menunjukkan kontribusi model, bukan hubungan sebab-akibat
- Sampel diambil dari validation yang juga dipakai memilih model
- Ketersediaan cuaca secara real-time belum terverifikasi
- Hasil tidak membuktikan generalisasi lintas tahun