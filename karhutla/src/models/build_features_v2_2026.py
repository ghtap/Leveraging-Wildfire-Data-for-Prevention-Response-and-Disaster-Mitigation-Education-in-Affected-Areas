
from pathlib import Path
import duckdb

SOURCE = Path("data/processed/dataset_gabungan_2026.parquet")
OUTPUT = Path("data/processed/dataset_fitur_2026_v2.parquet")
TEMP = Path("data/processing/dataset_fitur_2026_v2.building.parquet")

if not SOURCE.exists():
    raise SystemExit(f"Dataset asal tidak ditemukan: {SOURCE}")

if OUTPUT.exists():
    raise SystemExit(
        f"Dataset v2 sudah ada: {OUTPUT}\n"
        "Tidak ditimpa agar hasil lama tetap aman."
    )

if TEMP.exists():
    raise SystemExit(
        f"Ada file sementara dari proses sebelumnya: {TEMP}\n"
        "Periksa file ini sebelum menjalankan ulang."
    )

TEMP.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

def sql_path(path):
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"

db = duckdb.connect()
db.execute("SET threads = 4")
db.execute("SET memory_limit = '2GB'")
db.execute(
    "SET temp_directory = 'data/processing/duckdb_temp_v2'"
)

print("MEMBUAT FITUR HISTORIS 2026...")
print("Sumber:", SOURCE)
print("Output:", OUTPUT)
print("Data asli tidak akan diubah.")

sql = f"""
COPY (
    WITH base AS (
        SELECT
            *,
            CAST(date AS DATE) AS tanggal_fitur
        FROM read_parquet({sql_path(SOURCE)})
    )
    SELECT
        base.* EXCLUDE (tanggal_fitur),

        EXTRACT(MONTH FROM tanggal_fitur)
            AS month_2026,

        -- Catat kelengkapan riwayat kalender
        COUNT(n_hotspot) OVER w7 AS days_prev7,
        COUNT(precip_mm) OVER w14 AS days_rain_prev14,
        COUNT(precip_mm) OVER w30 AS days_rain_prev30,

        -- Akumulasi hujan masa lalu
        CASE WHEN COUNT(precip_mm) OVER w7 = 7
            THEN SUM(precip_mm) OVER w7
        END AS rain_prev7_mm,

        CASE WHEN COUNT(precip_mm) OVER w14 = 14
            THEN SUM(precip_mm) OVER w14
        END AS rain_prev14_mm,

        CASE WHEN COUNT(precip_mm) OVER w30 = 30
            THEN SUM(precip_mm) OVER w30
        END AS rain_prev30_mm,

        -- Riwayat hotspot dan intensitas FIRMS
        CASE WHEN COUNT(n_hotspot) OVER w7 = 7
            THEN SUM(n_hotspot) OVER w7
        END AS hotspot_prev7,

        CASE WHEN COUNT(n_hotspot) OVER w14 = 14
            THEN SUM(n_hotspot) OVER w14
        END AS hotspot_prev14,

        CASE WHEN COUNT(frp_sum) OVER w7 = 7
            THEN SUM(frp_sum) OVER w7
        END AS frp_prev7,

        -- Riwayat ERA5
        CASE WHEN COUNT(t2m_c) OVER w7 = 7
            THEN AVG(t2m_c) OVER w7
        END AS temp_prev7_mean,

        CASE WHEN COUNT(rh) OVER w7 = 7
            THEN AVG(rh) OVER w7
        END AS rh_prev7_mean,

        CASE WHEN COUNT(sm1) OVER w7 = 7
            THEN AVG(sm1) OVER w7
        END AS soil_prev7_mean,

        CASE WHEN COUNT(wind_ms) OVER w7 = 7
            THEN AVG(wind_ms) OVER w7
        END AS wind_prev7_mean

    FROM base

    WINDOW
        w7 AS (
            PARTITION BY cell_id
            ORDER BY tanggal_fitur
            RANGE BETWEEN
                INTERVAL '7 days' PRECEDING
                AND INTERVAL '1 day' PRECEDING
        ),
        w14 AS (
            PARTITION BY cell_id
            ORDER BY tanggal_fitur
            RANGE BETWEEN
                INTERVAL '14 days' PRECEDING
                AND INTERVAL '1 day' PRECEDING
        ),
        w30 AS (
            PARTITION BY cell_id
            ORDER BY tanggal_fitur
            RANGE BETWEEN
                INTERVAL '30 days' PRECEDING
                AND INTERVAL '1 day' PRECEDING
        )
) TO {sql_path(TEMP)}
(FORMAT PARQUET, COMPRESSION ZSTD)
"""

db.execute(sql)
TEMP.replace(OUTPUT)

print("\n=== KELENGKAPAN FITUR ===")

qc = db.execute(f"""
    SELECT
        split,
        COUNT(*) AS rows,
        COUNT(rain_prev7_mm) AS rain7_ready,
        COUNT(rain_prev14_mm) AS rain14_ready,
        COUNT(rain_prev30_mm) AS rain30_ready,
        COUNT(temp_prev7_mean) AS temp7_ready
    FROM read_parquet({sql_path(OUTPUT)})
    GROUP BY split
    ORDER BY split
""").fetchdf()

print(qc.to_string(index=False))

db.close()

print("\nBERHASIL MEMBUAT:", OUTPUT)
print("DATASET ASLI TETAP UTUH")
