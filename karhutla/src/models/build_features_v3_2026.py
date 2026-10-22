
from pathlib import Path
from datetime import date, timedelta
import csv
import duckdb

BASE = Path("data")
SOURCE = BASE / "processed/dataset_gabungan_2026.parquet"
GEE_DIR = BASE / "processing/gee_2026_downloads"
HOTSPOT = BASE / "interim/hotspots_daily.parquet"
OUTPUT = BASE / "processed/dataset_fitur_2026_v3.parquet"
TEMP = BASE / "processing/dataset_fitur_2026_v3.building.parquet"

def quote(s):
    return "'" + str(s).replace("'", "''") + "'"

def path_sql(p):
    return quote(p.resolve().as_posix())

def ident(s):
    return '"' + s.replace('"', '""') + '"'

def pick(names, options, required=True):
    lookup = {x.lower().strip(): x for x in names}
    for option in options:
        if option.lower() in lookup:
            return lookup[option.lower()]
    if required:
        raise ValueError(
            f"Kolom tidak ditemukan: {options}\n"
            f"Kolom tersedia: {list(names)}"
        )
    return None

def number(col):
    if col is None:
        return "NULL::DOUBLE"
    x = f"TRY_CAST({ident(col)} AS DOUBLE)"
    return f"NULLIF({x}, -9999)"

def days(a, b):
    x = a
    while x <= b:
        yield x
        x += timedelta(days=1)

gaps = list(days(date(2026, 7, 25), date(2026, 7, 31)))
gaps += list(days(date(2026, 8, 25), date(2026, 8, 31)))
gap_sql = ", ".join(f"DATE '{d.isoformat()}'" for d in gaps)

files = sorted(GEE_DIR.glob("gee26_*.csv"))

for p in [SOURCE, HOTSPOT]:
    if not p.exists():
        raise SystemExit(f"FILE TIDAK DITEMUKAN: {p}")

if not files:
    raise SystemExit("CSV gee26_*.csv tidak ditemukan.")

if OUTPUT.exists() or TEMP.exists():
    raise SystemExit(
        "Output v3 atau file sementara sudah ada. "
        "Tidak ditimpa untuk melindungi hasil lama."
    )

with files[0].open(encoding="utf-8-sig", newline="") as f:
    headers = next(csv.reader(f))

print("CSV GEE ditemukan:", len(files))
print("Kolom CSV GEE:", headers)

gid = pick(headers, ["cell_id"])
gdate = pick(headers, ["date"])
rain = pick(headers, ["precip_mm", "precipitation", "rain_mm"])
temp = pick(headers, ["t2m_c", "temperature_2m_c", "temperature_2m"])
rh = pick(
    headers,
    ["rh", "relative_humidity"],
    required=False
)

dew = pick(
    headers,
    ["dewpoint_temperature_2m"],
    required=False
)

if rh is None and dew is None:
    raise ValueError(
        "Data RH maupun dewpoint tidak ditemukan."
    )
soil = pick(headers, ["sm1", "volumetric_soil_water_layer_1"])
wind = pick(headers, ["wind_ms", "wind_speed"], required=False)

u = pick(headers, ["u_component_of_wind_10m", "u10"], False)
v = pick(headers, ["v_component_of_wind_10m", "v10"], False)

temp_expr = number(temp)
if temp == "temperature_2m":
    temp_expr = f"({temp_expr} - 273.15)"

# Menghitung RH dari suhu dan dewpoint apabila
# kolom RH tidak tersedia pada CSV ERA5-Land.
if rh is not None:
    rh_expr = number(rh)
else:
    dew_c = f"({number(dew)} - 273.15)"

    rh_expr = (
        f"LEAST(100.0, GREATEST(0.0, "
        f"100.0 * EXP("
        f"(17.625 * {dew_c}) / (243.04 + {dew_c}) "
        f"- (17.625 * {temp_expr}) / (243.04 + {temp_expr})"
        f")))"
    )
if wind is not None:
    wind_expr = number(wind)
elif u and v:
    wind_expr = f"SQRT(POWER({number(u)}, 2) + POWER({number(v)}, 2))"
else:
    raise ValueError("Kolom kecepatan angin tidak ditemukan.")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
TEMP.parent.mkdir(parents=True, exist_ok=True)

db = duckdb.connect()
db.execute("SET threads = 4")
db.execute("SET memory_limit = '2GB'")

tempdir = BASE / "processing/duckdb_v3_temp"
tempdir.mkdir(parents=True, exist_ok=True)
db.execute(f"SET temp_directory = {path_sql(tempdir)}")

gee_pattern = path_sql(GEE_DIR / "gee26_*.csv")

print("\n1. MEMBACA CUACA PADA 14 TANGGAL JEDA...")

db.execute(f"""
CREATE TEMP TABLE gee_gap AS
SELECT
    TRY_CAST({ident(gid)} AS BIGINT) AS cell_id,
    TRY_CAST({ident(gdate)} AS DATE) AS tanggal,
    {number(rain)} AS precip_mm,
    {temp_expr} AS t2m_c,
    {rh_expr} AS rh,
    {number(soil)} AS sm1,
    {wind_expr} AS wind_ms
FROM read_csv_auto(
    {gee_pattern},
    union_by_name = true
)
WHERE TRY_CAST({ident(gdate)} AS DATE) IN ({gap_sql})
""")

duplicate = db.execute("""
SELECT COUNT(*) FROM (
    SELECT cell_id, tanggal
    FROM gee_gap
    GROUP BY 1, 2
    HAVING COUNT(*) > 1
)
""").fetchone()[0]

if duplicate:
    raise ValueError(f"CSV GEE memiliki {duplicate} grid-tanggal duplikat.")

hcols = [
    x[0] for x in db.execute(
        f"DESCRIBE SELECT * FROM read_parquet({path_sql(HOTSPOT)})"
    ).fetchall()
]

hid = pick(hcols, ["cell_id"])
hdate = pick(hcols, ["date"])
hcount = pick(hcols, ["n_hotspot", "hotspot_count"])
hfrp = pick(hcols, ["frp_sum", "total_frp"])

print("2. MEMBACA RIWAYAT FIRMS PADA TANGGAL JEDA...")

db.execute(f"""
CREATE TEMP TABLE firms_gap AS
SELECT
    TRY_CAST({ident(hid)} AS BIGINT) AS cell_id,
    TRY_CAST({ident(hdate)} AS DATE) AS tanggal,
    SUM({number(hcount)}) AS n_hotspot,
    SUM({number(hfrp)}) AS frp_sum
FROM read_parquet({path_sql(HOTSPOT)})
WHERE TRY_CAST({ident(hdate)} AS DATE) IN ({gap_sql})
GROUP BY 1, 2
""")

values = ", ".join(f"(DATE '{d.isoformat()}')" for d in gaps)

db.execute(f"""
CREATE TEMP TABLE gap_context AS
SELECT
    g.cell_id,
    d.tanggal,
    w.precip_mm,
    w.t2m_c,
    w.rh,
    w.sm1,
    w.wind_ms,
    COALESCE(h.n_hotspot, 0) AS n_hotspot,
    COALESCE(h.frp_sum, 0) AS frp_sum
FROM (
    SELECT DISTINCT cell_id
    FROM read_parquet({path_sql(SOURCE)})
) g
CROSS JOIN (VALUES {values}) AS d(tanggal)
LEFT JOIN gee_gap w
    ON w.cell_id = g.cell_id AND w.tanggal = d.tanggal
LEFT JOIN firms_gap h
    ON h.cell_id = g.cell_id AND h.tanggal = d.tanggal
""")

n_context, n_rain_missing = db.execute("""
SELECT COUNT(*), COUNT(*) FILTER (WHERE precip_mm IS NULL)
FROM gap_context
""").fetchone()

print("Baris konteks tambahan:", n_context)
print("Hujan kosong pada konteks:", n_rain_missing)

if n_context != 22553 * 14:
    raise ValueError("Jumlah grid konteks tidak sesuai 22.553 x 14 hari.")

if n_rain_missing:
    raise ValueError(
        "Masih ada data hujan kosong pada tanggal jeda. "
        "Periksa header dan cakupan CSV GEE sebelum membuat v3."
    )

print("3. MEMBANGUN RIWAYAT KALENDER LENGKAP...")

db.execute(f"""
CREATE TEMP TABLE histori AS
SELECT
    CAST(cell_id AS BIGINT) AS cell_id,
    CAST(date AS DATE) AS tanggal,
    precip_mm, t2m_c, rh, sm1, wind_ms,
    n_hotspot, frp_sum
FROM read_parquet({path_sql(SOURCE)})

UNION ALL

SELECT
    cell_id, tanggal,
    precip_mm, t2m_c, rh, sm1, wind_ms,
    n_hotspot, frp_sum
FROM gap_context
""")

print("4. MENGHITUNG FITUR HISTORIS DAN MENYIMPAN V3...")

db.execute(f"""
COPY (
WITH rolling AS (
    SELECT
        cell_id, tanggal,

        CASE WHEN COUNT(precip_mm) OVER w7 = 7
            THEN SUM(precip_mm) OVER w7
        END AS rain_prev7_mm,

        CASE WHEN COUNT(precip_mm) OVER w14 = 14
            THEN SUM(precip_mm) OVER w14
        END AS rain_prev14_mm,

        CASE WHEN COUNT(precip_mm) OVER w30 = 30
            THEN SUM(precip_mm) OVER w30
        END AS rain_prev30_mm,

        CASE WHEN COUNT(n_hotspot) OVER w7 = 7
            THEN SUM(n_hotspot) OVER w7
        END AS hotspot_prev7,

        CASE WHEN COUNT(n_hotspot) OVER w14 = 14
            THEN SUM(n_hotspot) OVER w14
        END AS hotspot_prev14,

        CASE WHEN COUNT(frp_sum) OVER w7 = 7
            THEN SUM(frp_sum) OVER w7
        END AS frp_prev7,

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

    FROM histori

    WINDOW
        w7 AS (
            PARTITION BY cell_id ORDER BY tanggal
            RANGE BETWEEN INTERVAL '7 days' PRECEDING
            AND INTERVAL '1 day' PRECEDING
        ),
        w14 AS (
            PARTITION BY cell_id ORDER BY tanggal
            RANGE BETWEEN INTERVAL '14 days' PRECEDING
            AND INTERVAL '1 day' PRECEDING
        ),
        w30 AS (
            PARTITION BY cell_id ORDER BY tanggal
            RANGE BETWEEN INTERVAL '30 days' PRECEDING
            AND INTERVAL '1 day' PRECEDING
        )
)
SELECT
    b.*,
    EXTRACT(MONTH FROM CAST(b.date AS DATE)) AS month_2026,
    r.* EXCLUDE (cell_id, tanggal)

FROM read_parquet({path_sql(SOURCE)}) b
JOIN rolling r
    ON b.cell_id = r.cell_id
    AND CAST(b.date AS DATE) = r.tanggal

) TO {path_sql(TEMP)}
(FORMAT PARQUET, COMPRESSION ZSTD)
""")

print("\n=== QC FITUR V3 ===")

qc = db.execute(f"""
SELECT
    split,
    COUNT(*) AS rows,
    ROUND(100.0 * COUNT(*) FILTER (
        WHERE rain_prev7_mm IS NULL
    ) / COUNT(*), 2) AS missing_rain7_pct,
    ROUND(100.0 * COUNT(*) FILTER (
        WHERE rain_prev14_mm IS NULL
    ) / COUNT(*), 2) AS missing_rain14_pct,
    ROUND(100.0 * COUNT(*) FILTER (
        WHERE temp_prev7_mean IS NULL
    ) / COUNT(*), 2) AS missing_temp7_pct
FROM read_parquet({path_sql(TEMP)})
GROUP BY split
ORDER BY split
""").fetchdf()

print(qc.to_string(index=False))

db.close()
TEMP.replace(OUTPUT)

print("\nBERHASIL:", OUTPUT)
print("DATASET ASLI DAN MODEL LAMA TIDAK DIUBAH")
