# AUDIT DATA LEAKAGE - DIMAS-TI 2026

Status: **LOLOS_PEMERIKSAAN_OFFLINE_TERBATAS**

## Temuan penting
- jeda_kurang_7_hari: False
- duplikat_grid_tanggal: 0
- missing_label: 0
- inkonsistensi_label_internal: 0
- bulan_tidak_cocok: 0
- label_berbeda_dari_FIRMS_sampel: 0
- jumlah_hotspot_next7_berbeda_sampel: 0
- rolling_hujan_berbeda_sampel: 0
- rolling_hotspot_berbeda_sampel: 0
- fitur_terlarang: []

## Label dan rolling
- Sampel label: 900 baris
- Sampel rolling: 600 baris

## Keterbatasan
- Jeda publikasi CHIRPS dan ERA5-Land
- Ketersediaan fitur pada waktu prediksi nyata
- Generalisasi spasial antarlokasi
- Test September sudah pernah digunakan sebelumnya
- Validasi rolling pada tanggal jeda belum diaudit langsung

Audit offline bukan jaminan bebas leakage sepenuhnya.