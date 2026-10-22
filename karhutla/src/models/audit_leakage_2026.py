
from pathlib import Path
import json
import duckdb

DATA = Path("data/processed/dataset_fitur_2026_v3.parquet")
BASE = Path("data/processed/dataset_gabungan_2026.parquet")
FIRMS = Path("data/interim/hotspots_daily.parquet")
SELECTED = Path("results_orang_b/selected_stage16_2026.json")

OUT = Path("reports/orang_b")
OUT.mkdir(parents=True, exist_ok=True)

REPORT_JSON = OUT / "audit_leakage_2026.json"
REPORT_MD = OUT / "audit_leakage_2026.md"

for path in [DATA, BASE, FIRMS, SELECTED]:
    if not path.exists():
        raise SystemExit(f"File tidak ditemukan: {path}")

if REPORT_JSON.exists() or REPORT_MD.exists():
    raise SystemExit(
        "Laporan audit sudah ada. Tidak ditimpa.\n"
        "Periksa laporan lama sebelum menjalankan ulang."
    )

def sqlpath(path):
    value = path.resolve().as_posix().replace("'", "''")
    return f"'{value}'"

db = duckdb.connect()
db.execute("SET threads = 4")
db.execute("SET memory_limit = '2GB'")

temp = Path("data/processing/duckdb_audit_temp")
temp.mkdir(parents=True, exist_ok=True)
db.execute(f"SET temp_directory = {sqlpath(temp)}")

d = sqlpath(DATA)
b = sqlpath(BASE)
h = sqlpath(FIRMS)

def columns(source):
    return [
        row[0] for row in db.execute(
            f"DESCRIBE SELECT * FROM read_parquet({source})"
        ).fetchall()
    ]

data_cols = columns(d)
firms_cols = columns(h)

required = {
    "cell_id", "date", "split", "y_7d",
    "n_hot_next_7d", "rain_prev7_mm",
    "hotspot_prev7", "month_2026"
}

missing = required - set(data_cols)
if missing:
    raise SystemExit(f"Kolom fitur tidak lengkap: {missing}")

if not {"cell_id", "date", "n_hotspot"}.issubset(firms_cols):
    raise SystemExit(
        f"Skema FIRMS perlu diperiksa: {firms_cols}"
    )

print("=== PAKET 17: AUDIT DATA LEAKAGE ===")

# 1. Audit split temporal
print("\n1. MEMERIKSA PEMBAGIAN WAKTU...")

rows = db.execute(f"""
    SELECT
        split,
        MIN(CAST(date AS DATE)),
        MAX(CAST(date AS DATE)),
        COUNT(*),
        SUM(CAST(y_7d AS BIGINT))
    FROM read_parquet({d})
    GROUP BY split
    ORDER BY split
""").fetchall()

splits = {
    name: {
        "awal": start.isoformat(),
        "akhir": end.isoformat(),
        "baris": int(n),
        "positif": int(pos),
        "prevalensi": float(pos / n),
    }
    for name, start, end, n, pos in rows
}

if set(splits) != {"train", "validation", "test"}:
    raise ValueError(f"Split tidak sesuai: {list(splits)}")

gaps = {}

for a, z in [
    ("train", "validation"),
    ("validation", "test"),
]:
    akhir = next(
        r[2] for r in rows if r[0] == a
    )
    awal = next(
        r[1] for r in rows if r[0] == z
    )

    n_gap = (awal - akhir).days - 1
    gaps[f"{a}_ke_{z}"] = n_gap

    print(f"Jeda {a} -> {z}: {n_gap} hari")

# 2. Integritas dataset
print("\n2. MEMERIKSA LABEL DAN DUPLIKAT...")

integrity = db.execute(f"""
SELECT
    COUNT(*) AS total,
    COUNT(*) FILTER (
        WHERE y_7d IS NULL OR n_hot_next_7d IS NULL
    ) AS missing_label,
    COUNT(*) FILTER (
        WHERE y_7d <> CASE
            WHEN n_hot_next_7d > 0 THEN 1
            ELSE 0 END
    ) AS label_inconsistent,
    COUNT(*) FILTER (
        WHERE CAST(month_2026 AS INTEGER)
            <> EXTRACT(MONTH FROM CAST(date AS DATE))
    ) AS invalid_month
FROM read_parquet({d})
""").fetchone()

duplicate = db.execute(f"""
SELECT COUNT(*) FROM (
    SELECT cell_id, CAST(date AS DATE)
    FROM read_parquet({d})
    GROUP BY 1, 2
    HAVING COUNT(*) > 1
)
""").fetchone()[0]

# 3. Validasi label terhadap FIRMS mentah yang telah diproses
print("\n3. MENGECEK LABEL TERHADAP SUMBER FIRMS...")

db.execute(f"""
CREATE TEMP TABLE label_sample AS
SELECT cell_id, tanggal, split, y_7d, n_hot_next_7d
FROM (
    SELECT
        cell_id,
        CAST(date AS DATE) AS tanggal,
        split,
        y_7d,
        n_hot_next_7d,
        ROW_NUMBER() OVER (
            PARTITION BY split, y_7d
            ORDER BY hash(cell_id, date)
        ) AS urutan
    FROM read_parquet({d})
)
WHERE urutan <= 150
""")

label_audit = db.execute(f"""
WITH truth AS (
    SELECT
        s.cell_id,
        s.tanggal,
        s.split,
        s.y_7d,
        s.n_hot_next_7d,
        COALESCE(SUM(h.n_hotspot), 0) AS actual_next7
    FROM label_sample s
    LEFT JOIN read_parquet({h}) h
        ON s.cell_id = h.cell_id
       AND CAST(h.date AS DATE) > s.tanggal
       AND CAST(h.date AS DATE)
           <= s.tanggal + INTERVAL '7 days'
    GROUP BY 1, 2, 3, 4, 5
)
SELECT
    COUNT(*),
    COUNT(*) FILTER (
        WHERE CAST(actual_next7 > 0 AS INTEGER)
            <> CAST(y_7d AS INTEGER)
    ),
    COUNT(*) FILTER (
        WHERE ABS(actual_next7 - n_hot_next_7d) > 0.001
    )
FROM truth
""").fetchone()

# 4. Audit rolling features pada sampel tanggal kontinu
print("\n4. MEMERIKSA FITUR HISTORIS H-1 SAMPAI H-7...")

db.execute(f"""
CREATE TEMP TABLE rolling_sample AS
SELECT cell_id, tanggal,
       rain_prev7_mm, hotspot_prev7
FROM (
    SELECT
        cell_id,
        CAST(date AS DATE) AS tanggal,
        split,
        rain_prev7_mm,
        hotspot_prev7,
        ROW_NUMBER() OVER (
            PARTITION BY split
            ORDER BY hash(cell_id, date)
        ) AS urutan
    FROM read_parquet({d})
    WHERE
        (split = 'train'
         AND CAST(date AS DATE)
             BETWEEN DATE '2026-02-01'
             AND DATE '2026-07-24')
        OR
        (split = 'validation'
         AND CAST(date AS DATE)
             BETWEEN DATE '2026-08-16'
             AND DATE '2026-08-24')
        OR
        (split = 'test'
         AND CAST(date AS DATE)
             BETWEEN DATE '2026-09-16'
             AND DATE '2026-09-30')
)
WHERE urutan <= 200
""")

rolling_audit = db.execute(f"""
WITH expected AS (
    SELECT
        s.cell_id, s.tanggal,
        s.rain_prev7_mm, s.hotspot_prev7,
        COUNT(b.precip_mm) AS hari_hujan,
        COUNT(b.n_hotspot) AS hari_hotspot,
        SUM(b.precip_mm) AS hujan_7d,
        SUM(b.n_hotspot) AS hotspot_7d
    FROM rolling_sample s
    LEFT JOIN read_parquet({b}) b
       ON s.cell_id = b.cell_id
      AND CAST(b.date AS DATE)
          BETWEEN s.tanggal - INTERVAL '7 days'
              AND s.tanggal - INTERVAL '1 day'
    GROUP BY 1, 2, 3, 4
)
SELECT
    COUNT(*),
    COUNT(*) FILTER (
        WHERE hari_hujan <> 7 OR hari_hotspot <> 7
    ),
    COUNT(*) FILTER (
        WHERE hari_hujan = 7
          AND (
              rain_prev7_mm IS NULL
              OR ABS(hujan_7d - rain_prev7_mm) > 0.01
          )
    ),
    COUNT(*) FILTER (
        WHERE hari_hotspot = 7
          AND (
              hotspot_prev7 IS NULL
              OR ABS(hotspot_7d - hotspot_prev7) > 0.01
          )
    )
FROM expected
""").fetchone()

# 5. Cek daftar fitur model terpilih
print("\n5. MEMERIKSA FITUR MODEL TERPILIH...")

with SELECTED.open(encoding="utf-8") as f:
    selected = json.load(f)

forbidden = {
    "y_7d", "n_hot_next_7d", "split",
    "date", "cell_id"
}

features = selected.get("features", [])
forbidden_found = sorted(forbidden.intersection(features))

# Ringkasan hasil
critical = {
    "jeda_kurang_7_hari": any(
        v < 7 for v in gaps.values()
    ),
    "duplikat_grid_tanggal": int(duplicate),
    "missing_label": int(integrity[1]),
    "inkonsistensi_label_internal": int(integrity[2]),
    "bulan_tidak_cocok": int(integrity[3]),
    "label_berbeda_dari_FIRMS_sampel": int(label_audit[1]),
    "jumlah_hotspot_next7_berbeda_sampel": int(label_audit[2]),
    "rolling_hujan_berbeda_sampel": int(rolling_audit[2]),
    "rolling_hotspot_berbeda_sampel": int(rolling_audit[3]),
    "fitur_terlarang": forbidden_found,
}

has_critical = any(bool(v) for v in critical.values())

report = {
    "tahun": 2026,
    "audit": "Paket 17 - Data Leakage",
    "status_offline": (
        "GAGAL_PERIKSA"
        if has_critical
        else "LOLOS_PEMERIKSAAN_OFFLINE_TERBATAS"
    ),
    "split": splits,
    "jeda_hari": gaps,
    "jumlah_baris": int(integrity[0]),
    "duplikat": int(duplicate),
    "sampel_label": {
        "diperiksa": int(label_audit[0]),
        "beda_label": int(label_audit[1]),
        "beda_jumlah_hotspot": int(label_audit[2]),
    },
    "sampel_rolling": {
        "diperiksa": int(rolling_audit[0]),
        "riwayat_tidak_lengkap": int(rolling_audit[1]),
        "beda_hujan": int(rolling_audit[2]),
        "beda_hotspot": int(rolling_audit[3]),
    },
    "critical_findings": critical,
    "model_stage16": selected.get("model"),
    "fitur_model": features,
    "belum_terverifikasi": [
        "Jeda publikasi CHIRPS dan ERA5-Land",
        "Ketersediaan fitur pada waktu prediksi nyata",
        "Generalisasi spasial antarlokasi",
        "Test September sudah pernah digunakan sebelumnya",
        "Validasi rolling pada tanggal jeda belum diaudit langsung",
    ],
}

REPORT_JSON.write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

lines = [
    "# AUDIT DATA LEAKAGE - DIMAS-TI 2026",
    "",
    f"Status: **{report['status_offline']}**",
    "",
    "## Temuan penting",
]

for key, value in critical.items():
    lines.append(f"- {key}: {value}")

lines += [
    "",
    "## Label dan rolling",
    f"- Sampel label: {label_audit[0]} baris",
    f"- Sampel rolling: {rolling_audit[0]} baris",
    "",
    "## Keterbatasan",
]

for issue in report["belum_terverifikasi"]:
    lines.append(f"- {issue}")

lines += [
    "",
    "Audit offline bukan jaminan bebas leakage sepenuhnya.",
]

REPORT_MD.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

db.close()

print("\n=== HASIL AUDIT ===")
for key, value in critical.items():
    print(f"{key}: {value}")

print("\nSTATUS:", report["status_offline"])
print("Laporan JSON:", REPORT_JSON)
print("Laporan Markdown:", REPORT_MD)
print("\nDATA DAN MODEL SEBELUMNYA TIDAK DIUBAH")
