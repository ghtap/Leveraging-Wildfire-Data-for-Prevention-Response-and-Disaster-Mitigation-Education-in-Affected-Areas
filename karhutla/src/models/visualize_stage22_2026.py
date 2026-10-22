
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================
# VISUALISASI DIMAS-TI 2026
# =====================================

PRED = Path(
    "reports/orang_b/stage19_predictions_2026.parquet"
)
PROVINCE = Path(
    "reports/orang_b/stage19_province_2026.csv"
)
GRID = Path("data/interim/grid.csv")
SELECTED = Path(
    "results_orang_b/selected_stage16_2026.json"
)

FIG_DIR = Path("figures/orang_b")
REP_DIR = Path("reports/orang_b")
NB_DIR = Path("notebooks")

MAP_IMG = FIG_DIR / "stage22_risk_map_20260930.png"
AP_IMG = FIG_DIR / "stage22_province_pr_auc_2026.png"
ERROR_IMG = FIG_DIR / "stage22_province_errors_2026.png"

MAP_CSV = REP_DIR / "stage22_grid_risk_20260930.csv"
REPORT = REP_DIR / "stage22_visual_summary_2026.json"
NOTEBOOK = NB_DIR / "Visualisasi_Risiko_2026.ipynb"

OUTPUTS = [
    MAP_IMG, AP_IMG, ERROR_IMG,
    MAP_CSV, REPORT, NOTEBOOK
]

for p in [PRED, PROVINCE, GRID, SELECTED]:
    if not p.exists():
        raise SystemExit(f"FILE SUMBER TIDAK ADA: {p}")

for p in OUTPUTS:
    if p.exists():
        raise SystemExit(
            f"Hasil sudah ada: {p}\n"
            "Tidak ditimpa untuk menjaga pekerjaan lama."
        )

for directory in [FIG_DIR, REP_DIR, NB_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

print("=== PAKET 22: VISUALISASI MODEL ===")

with SELECTED.open(encoding="utf-8") as f:
    selected = json.load(f)

threshold = float(selected["threshold_f1"])

if not 0 < threshold < 1:
    raise ValueError("Threshold tidak valid.")

# =====================================
# 1. DATA PREDIKSI SEPTEMBER
# =====================================

print("\n1. MEMBACA PREDIKSI TEST SEPTEMBER...")

pred = pd.read_parquet(
    PRED,
    columns=[
        "cell_id", "date", "province",
        "probability", "predicted_f1"
    ]
)

pred["date"] = pd.to_datetime(
    pred["date"]
).dt.normalize()

target_date = pd.Timestamp("2026-09-30")

daily = pred.loc[
    pred["date"] == target_date
].copy()

if daily.empty:
    raise ValueError(
        "Prediksi 30 September 2026 tidak ditemukan."
    )

if daily["cell_id"].duplicated().any():
    raise ValueError(
        "Ada duplikasi cell_id pada tanggal peta."
    )

if daily["probability"].isna().any():
    raise ValueError("Probabilitas memiliki NaN.")

if not daily["probability"].between(0, 1).all():
    raise ValueError(
        "Probabilitas di luar rentang 0-1."
    )

# Pastikan hasil Stage 19 konsisten
# dengan threshold validation Stage 16.
expected_alert = (
    daily["probability"] >= threshold
).astype(int)

if not np.array_equal(
    expected_alert.to_numpy(),
    daily["predicted_f1"].astype(int).to_numpy()
):
    raise ValueError(
        "Prediksi threshold tidak cocok dengan Stage 19."
    )

print("Baris prediksi:", len(daily))

# =====================================
# 2. KOORDINAT GRID
# =====================================

print("\n2. MENGGABUNGKAN KOORDINAT GRID...")

grid = pd.read_csv(GRID)
grid.columns = grid.columns.str.strip()

def choose_column(columns, candidates):
    lookup = {
        c.lower(): c for c in columns
    }
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise ValueError(
        f"Kolom tidak ditemukan: {candidates}\n"
        f"Kolom tersedia: {list(columns)}"
    )

cell_col = choose_column(
    grid.columns, ["cell_id"]
)
lon_col = choose_column(
    grid.columns,
    ["lon", "longitude", "center_lon",
     "centroid_lon", "x_center"]
)
lat_col = choose_column(
    grid.columns,
    ["lat", "latitude", "center_lat",
     "centroid_lat", "y_center"]
)

coords = grid[
    [cell_col, lon_col, lat_col]
].copy()

coords.columns = ["cell_id", "lon", "lat"]

if coords["cell_id"].duplicated().any():
    raise ValueError("Grid memiliki cell_id duplikat.")

daily["cell_id"] = pd.to_numeric(
    daily["cell_id"], errors="raise"
).astype("int64")

coords["cell_id"] = pd.to_numeric(
    coords["cell_id"], errors="raise"
).astype("int64")

coords["lon"] = pd.to_numeric(
    coords["lon"], errors="coerce"
)
coords["lat"] = pd.to_numeric(
    coords["lat"], errors="coerce"
)

mapped = daily.merge(
    coords,
    on="cell_id",
    how="left",
    validate="one_to_one"
)

if mapped[["lon", "lat"]].isna().any().any():
    raise ValueError(
        "Ada grid tanpa koordinat. Periksa grid.csv."
    )

if not mapped["lon"].between(-180, 180).all():
    raise ValueError("Longitude tidak valid.")

if not mapped["lat"].between(-90, 90).all():
    raise ValueError("Latitude tidak valid.")

if len(mapped) != coords["cell_id"].nunique():
    raise ValueError(
        "Cakupan prediksi harian tidak sama dengan grid."
    )

print("Grid berhasil dipetakan:", len(mapped))

# Tabel yang akan digunakan Orang C
# Tidak memasukkan label aktual masa depan.
export = mapped[
    [
        "cell_id", "date", "province",
        "lon", "lat", "probability"
    ]
].copy()

export["alert_f1_threshold"] = (
    export["probability"] >= threshold
).astype("uint8")

export.to_csv(MAP_CSV, index=False)

# =====================================
# 3. PETA PROBABILITAS HOTSPOT
# =====================================

print("\n3. MEMBUAT PETA RISIKO...")

fig, ax = plt.subplots(figsize=(12, 6))

scatter = ax.scatter(
    mapped["lon"],
    mapped["lat"],
    c=mapped["probability"],
    cmap="YlOrRd",
    vmin=0,
    vmax=1,
    s=7,
    alpha=0.85,
    linewidths=0,
    rasterized=True
)

fig.colorbar(
    scatter,
    ax=ax,
    label="Probabilitas model y_7d"
)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

ax.set_title(
    "Probabilitas Hotspot 7 Hari Berikutnya\n"
    "Grid Penelitian - 30 September 2026"
)

ax.grid(alpha=0.15)

fig.text(
    0.5, 0.02,
    "Peta titik pusat grid historis; "
    "bukan peta kebakaran terkonfirmasi atau prediksi real-time.",
    ha="center", fontsize=9
)

fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(MAP_IMG, dpi=180)
plt.close(fig)

# =====================================
# 4. GRAFIK EVALUASI PER PROVINSI
# =====================================

print("\n4. MEMBUAT GRAFIK PER PROVINSI...")

province = pd.read_csv(PROVINCE)

required = {
    "province", "pr_auc", "positive_rate",
    "tp", "fp", "tn", "fn"
}

missing = required - set(province.columns)
if missing:
    raise ValueError(
        f"Kolom laporan provinsi tidak lengkap: {missing}"
    )

province = province.sort_values(
    "pr_auc", ascending=False
).reset_index(drop=True)

names = province["province"].tolist()
positions = np.arange(len(names))

# 4a. PR-AUC dibanding prevalensi positif
fig, ax = plt.subplots(figsize=(11, 6))

w = 0.38

ax.barh(
    positions - w/2,
    province["pr_auc"],
    height=w,
    label="PR-AUC"
)

ax.barh(
    positions + w/2,
    province["positive_rate"],
    height=w,
    label="Prevalensi positif"
)

ax.set_yticks(positions, names)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("Skor")
ax.set_title(
    "PR-AUC dan Prevalensi Hotspot per Provinsi\n"
    "Test September 2026"
)
ax.legend()

fig.tight_layout()
fig.savefig(AP_IMG, dpi=180)
plt.close(fig)

# 4b. False negative rate dan false positive rate
province["fnr"] = (
    province["fn"] /
    (province["tp"] + province["fn"])
)

province["fpr"] = (
    province["fp"] /
    (province["fp"] + province["tn"])
)

if province[["fnr", "fpr"]].isna().any().any():
    raise ValueError(
        "Ada denominator nol pada metrik provinsi."
    )

fig, ax = plt.subplots(figsize=(11, 6))

ax.barh(
    positions - w/2,
    province["fnr"],
    height=w,
    label="False Negative Rate"
)

ax.barh(
    positions + w/2,
    province["fpr"],
    height=w,
    label="False Positive Rate"
)

ax.set_yticks(positions, names)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("Proporsi")
ax.set_title(
    "Tingkat Kesalahan Model per Provinsi\n"
    "Threshold F1 dari Validation"
)
ax.legend()

fig.tight_layout()
fig.savefig(ERROR_IMG, dpi=180)
plt.close(fig)

# =====================================
# 5. LAPORAN UNTUK ORANG C
# =====================================

print("\n5. MENYIMPAN LAPORAN...")

report = {
    "stage": 22,
    "title": "Peta probabilitas dan evaluasi wilayah",
    "model": selected["model"],
    "map_date": "2026-09-30",
    "prediction_horizon_days": 7,
    "n_grids": int(len(mapped)),
    "n_provinces": int(mapped["province"].nunique()),
    "threshold_f1": threshold,
    "n_alerts": int(
        export["alert_f1_threshold"].sum()
    ),
    "average_probability": float(
        mapped["probability"].mean()
    ),
    "files": {
        "map": str(MAP_IMG),
        "province_ap": str(AP_IMG),
        "province_errors": str(ERROR_IMG),
        "grid_export": str(MAP_CSV),
        "notebook": str(NOTEBOOK)
    },
    "notes": [
        "Peta merupakan hasil historis September 2026",
        "Titik adalah pusat grid, bukan batas kebakaran",
        "Probabilitas model belum diuji kalibrasinya",
        "Threshold dipilih menggunakan validation Agustus",
        "Test September pernah terekspos dalam eksperimen",
        "Fitur cuaca untuk operasional real-time belum diaudit"
    ]
}

REPORT.write_text(
    json.dumps(
        report, indent=2, ensure_ascii=False
    ),
    encoding="utf-8"
)

# =====================================
# 6. BUAT NOTEBOOK VISUALISASI
# =====================================

print("\n6. MEMBUAT NOTEBOOK OTOMATIS...")

def markdown(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True)
    }

def code(text):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True)
    }

cells = [
    markdown(
        "# Visualisasi Risiko Karhutla - Orang B\n\n"
        "Notebook ini menggunakan hasil model yang sudah "
        "tersimpan. Tidak melakukan training ulang.\n\n"
        "**Periode:** September 2026.\n\n"
        "**Catatan:** Probabilitas yang ditampilkan belum "
        "terkalibrasi dan tidak mewakili sistem real-time."
    ),

    code(
        "from pathlib import Path\n"
        "import json\n"
        "import pandas as pd\n"
        "from IPython.display import display, Image\n"
        "ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] "
        "if (p / 'reports/orang_b/stage22_visual_summary_2026.json').exists()), None)\n"
        "assert ROOT is not None, 'Folder proyek tidak ditemukan'\n"
        "report = json.loads((ROOT / 'reports/orang_b/stage22_visual_summary_2026.json').read_text(encoding='utf-8'))\n"
        "print('Tanggal:', report['map_date'])\n"
        "print('Jumlah grid:', report['n_grids'])\n"
    ),

    markdown(
        "## 1. Peta probabilitas hotspot\n\n"
        "Tiap titik mewakili pusat grid penelitian. "
        "Warna menggambarkan probabilitas prediksi model "
        "untuk label hotspot tujuh hari ke depan."
    ),

    code(
        "display(Image(filename=str(ROOT / 'figures/orang_b/stage22_risk_map_20260930.png')))\n"
        "display(pd.read_csv(ROOT / 'reports/orang_b/stage22_grid_risk_20260930.csv').head(10))\n"
    ),

    markdown(
        "## 2. PR-AUC per provinsi\n\n"
        "PR-AUC dibandingkan dengan prevalensi positif sebagai "
        "acuan dasar. Perbedaan prevalensi antarprovinsi perlu "
        "diperhatikan ketika menafsirkan peringkat."
    ),

    code(
        "display(Image(filename=str(ROOT / 'figures/orang_b/stage22_province_pr_auc_2026.png')))\n"
        "display(pd.read_csv(ROOT / 'reports/orang_b/stage19_province_2026.csv'))\n"
    ),

    markdown(
        "## 3. Kesalahan prediksi per provinsi\n\n"
        "FNR = FN / (TP + FN)\n\n"
        "FPR = FP / (FP + TN)\n\n"
        "FNR mengukur proporsi observasi positif yang terlewat. "
        "FPR mengukur proporsi observasi negatif yang menjadi "
        "false alarm."
    ),

    code(
        "display(Image(filename=str(ROOT / 'figures/orang_b/stage22_province_errors_2026.png')))\n"
    ),

    markdown(
        "## 4. SHAP dan evaluasi model\n\n"
        "Untuk grafik SHAP, Precision-Recall Curve, dan confusion "
        "matrix, gunakan notebook `Analisis_Model_2026.ipynb` "
        "yang telah dibuat pada Paket 21.\n\n"
        "SHAP menjelaskan perilaku model, bukan hubungan kausal."
    ),

    markdown(
        "## 5. Keterbatasan\n\n"
        "- Ini peta historis, bukan kondisi terkini.\n"
        "- Skor probabilitas belum melalui kalibrasi.\n"
        "- Model memprediksi hotspot, bukan kebakaran terkonfirmasi.\n"
        "- Generalisasi lintas tahun belum diuji.\n"
        "- Ketersediaan fitur cuaca saat prediksi real-time "
        "belum terverifikasi."
    )
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

NOTEBOOK.write_text(
    json.dumps(
        notebook,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)

print("\n=== VISUALISASI DIMAS-TI 2026 ===")
for p in OUTPUTS:
    print("TERBUAT:", p)

print("\nTIDAK ADA MODEL/DATASET LAMA YANG DIUBAH")
