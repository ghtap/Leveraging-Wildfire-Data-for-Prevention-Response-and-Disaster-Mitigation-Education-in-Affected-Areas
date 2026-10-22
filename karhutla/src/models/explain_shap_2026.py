
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===== LOKASI PROYEK =====
DATA = Path("data/processed/dataset_fitur_2026_v3.parquet")
SELECTED = Path("results_orang_b/selected_stage16_2026.json")

REPORT_DIR = Path("reports/orang_b")
FIG_DIR = Path("figures/orang_b")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_CSV = REPORT_DIR / "stage20_shap_global_2026.csv"
LOCAL_CSV = REPORT_DIR / "stage20_shap_local_2026.csv"
REPORT_JSON = REPORT_DIR / "stage20_shap_2026.json"
REPORT_MD = REPORT_DIR / "stage20_shap_2026.md"

GLOBAL_PNG = FIG_DIR / "stage20_shap_global_2026.png"
GAIN_PNG = FIG_DIR / "stage20_feature_importance_2026.png"
LOCAL_PNG = FIG_DIR / "stage20_shap_local_2026.png"

outputs = [
    GLOBAL_CSV, LOCAL_CSV, REPORT_JSON,
    REPORT_MD, GLOBAL_PNG, GAIN_PNG, LOCAL_PNG
]

for p in [DATA, SELECTED]:
    if not p.exists():
        raise SystemExit(f"File tidak ditemukan: {p}")

for p in outputs:
    if p.exists():
        raise SystemExit(
            f"Output sebelumnya sudah ada: {p}\n"
            "Tidak ditimpa agar eksperimen tetap aman."
        )

with SELECTED.open(encoding="utf-8") as f:
    selection = json.load(f)

FEATURES = selection["features"]
MODEL_PATH = Path(selection["model_path"])

if not MODEL_PATH.exists():
    raise SystemExit(f"Model tidak ditemukan: {MODEL_PATH}")

forbidden = {
    "y_7d", "n_hot_next_7d",
    "date", "split", "cell_id"
}

if forbidden.intersection(FEATURES):
    raise ValueError("Ada fitur terlarang dalam model.")

print("=== PAKET 20: SHAP DAN FEATURE IMPORTANCE ===")
print("Model:", MODEL_PATH)
print("Membaca validation Agustus 2026 saja...")

columns = [
    "cell_id", "date", "province",
    "split", "y_7d"
] + FEATURES

df = pd.read_parquet(
    DATA,
    columns=columns,
    filters=[("split", "==", "validation")]
)

df = df.loc[
    df["split"] == "validation"
].reset_index(drop=True)

if df.empty:
    raise ValueError("Validation kosong.")

if df["y_7d"].isna().any():
    raise ValueError("Label validation kosong.")

# Sampel acak tetap, agar analisis mudah direproduksi.
SAMPLE_SIZE = min(6000, len(df))
sample = df.sample(
    n=SAMPLE_SIZE,
    random_state=42,
    replace=False
).reset_index(drop=True)

X = sample[FEATURES].apply(
    pd.to_numeric, errors="coerce"
).astype("float32")

model = joblib.load(MODEL_PATH)

if not hasattr(model, "booster_"):
    raise TypeError(
        "Model terpilih bukan LightGBM sklearn yang dilatih."
    )

booster = model.booster_

if list(booster.feature_name()) != FEATURES:
    raise ValueError(
        "Urutan nama fitur model berbeda dari dataset."
    )

# ===== 1. HITUNG KONTRIBUSI SHAP =====
print("\n1. Menghitung kontribusi SHAP...")

contributions = booster.predict(
    X,
    pred_contrib=True,
    validate_features=True
)

if not isinstance(contributions, np.ndarray):
    raise TypeError(
        "Format kontribusi SHAP tidak sesuai model biner."
    )

if contributions.shape != (
    SAMPLE_SIZE, len(FEATURES) + 1
):
    raise ValueError(
        f"Dimensi SHAP tidak sesuai: {contributions.shape}"
    )

shap_values = contributions[:, :-1]
base_values = contributions[:, -1]

raw_scores = booster.predict(
    X,
    raw_score=True,
    validate_features=True
)

# Verifikasi sifat aditif SHAP dalam skala log-odds.
error = float(
    np.max(
        np.abs(
            shap_values.sum(axis=1)
            + base_values
            - raw_scores
        )
    )
)

print("Selisih rekonstruksi SHAP maksimum:", error)

if not np.isfinite(error) or error > 1e-4:
    raise ValueError(
        "Kontribusi SHAP tidak merekonstruksi raw score."
    )

probabilities = booster.predict(
    X,
    validate_features=True
)

# ===== 2. FEATURE IMPORTANCE GLOBAL =====
print("\n2. Menghitung feature importance global...")

importance = pd.DataFrame({
    "feature": FEATURES,
    "mean_abs_shap": np.abs(
        shap_values
    ).mean(axis=0),
    "mean_signed_shap": shap_values.mean(axis=0),
    "gain_importance": booster.feature_importance(
        importance_type="gain"
    ),
    "split_importance": booster.feature_importance(
        importance_type="split"
    ),
    "missing_sample_pct": (
        X.isna().mean().to_numpy() * 100
    ),
}).sort_values(
    "mean_abs_shap", ascending=False
).reset_index(drop=True)

print("\n=== PERINGKAT SHAP GLOBAL ===")
print(
    importance[
        ["feature", "mean_abs_shap",
         "gain_importance", "missing_sample_pct"]
    ].to_string(index=False)
)

importance.to_csv(GLOBAL_CSV, index=False)

# Grafik global SHAP
ranked = importance.iloc[::-1]

plt.figure(figsize=(10, 6))
plt.barh(
    ranked["feature"],
    ranked["mean_abs_shap"]
)
plt.xlabel("Rata-rata |SHAP| (log-odds)")
plt.title("Global SHAP Importance - Validation 2026")
plt.tight_layout()
plt.savefig(GLOBAL_PNG, dpi=180)
plt.close()

# Grafik importance menurut gain LightGBM
gain_ranked = importance.sort_values(
    "gain_importance"
)

plt.figure(figsize=(10, 6))
plt.barh(
    gain_ranked["feature"],
    gain_ranked["gain_importance"]
)
plt.xlabel("Total split gain")
plt.title("LightGBM Feature Importance - Gain")
plt.tight_layout()
plt.savefig(GAIN_PNG, dpi=180)
plt.close()

# ===== 3. PENJELASAN SATU PREDIKSI =====
print("\n3. Membuat penjelasan prediksi lokal...")

# Contoh dengan probabilitas risiko tertinggi
# dari sampel validation.
idx = int(np.argmax(probabilities))

record = sample.iloc[idx]
local_shap = shap_values[idx]

local = pd.DataFrame({
    "feature": FEATURES,
    "value": [
        None if pd.isna(X.iloc[idx][c])
        else float(X.iloc[idx][c])
        for c in FEATURES
    ],
    "shap_log_odds": local_shap
})

local["abs_shap"] = local[
    "shap_log_odds"
].abs()

local = local.sort_values(
    "abs_shap", ascending=False
).reset_index(drop=True)

print("\n=== CONTOH PREDIKSI LOKAL ===")
print("Grid:", record["cell_id"])
print("Tanggal:", record["date"])
print("Provinsi:", record["province"])
print("Label sebenarnya:", record["y_7d"])
print("Probabilitas model:", probabilities[idx])
print("Nilai dasar log-odds:", base_values[idx])

print(
    local[
        ["feature", "value", "shap_log_odds"]
    ].to_string(index=False)
)

local.to_csv(LOCAL_CSV, index=False)

local_plot = local.iloc[::-1]

plt.figure(figsize=(10, 6))
plt.barh(
    local_plot["feature"],
    local_plot["shap_log_odds"]
)
plt.axvline(0, linewidth=1)
plt.xlabel("Kontribusi SHAP terhadap log-odds")
plt.title("Local SHAP - Contoh Risiko Tertinggi")
plt.tight_layout()
plt.savefig(LOCAL_PNG, dpi=180)
plt.close()

# ===== 4. SIMPAN LAPORAN =====
print("\n4. Menyimpan laporan...")

report = {
    "stage": 20,
    "method": "LightGBM native SHAP contributions",
    "dataset": str(DATA),
    "model": str(MODEL_PATH),
    "evaluation_split": "validation",
    "sample_size": SAMPLE_SIZE,
    "random_state": 42,
    "sample_positive_rate": float(
        sample["y_7d"].mean()
    ),
    "shap_scale": "raw log-odds",
    "reconstruction_max_error": error,
    "top_features": importance[
        "feature"
    ].head(5).tolist(),
    "local_example": {
        "cell_id": int(record["cell_id"]),
        "date": str(record["date"]),
        "province": str(record["province"]),
        "actual_label": int(record["y_7d"]),
        "predicted_probability": float(
            probabilities[idx]
        ),
        "base_log_odds": float(base_values[idx]),
        "raw_score": float(raw_scores[idx]),
    },
    "limitations": [
        "SHAP menunjukkan kontribusi model, bukan hubungan sebab-akibat",
        "Sampel diambil dari validation yang juga dipakai memilih model",
        "Ketersediaan cuaca secara real-time belum terverifikasi",
        "Hasil tidak membuktikan generalisasi lintas tahun",
    ],
}

REPORT_JSON.write_text(
    json.dumps(
        report, indent=2, ensure_ascii=False
    ),
    encoding="utf-8"
)

lines = [
    "# Interpretasi Model LightGBM - DIMAS-TI 2026",
    "",
    "## Metodologi",
    f"- Sampel validation: {SAMPLE_SIZE:,} observasi",
    "- Metode: LightGBM native SHAP contributions",
    "- Skala kontribusi: log-odds, bukan probabilitas",
    "",
    "## Lima fitur teratas",
]

for i, row in importance.head(5).iterrows():
    lines.append(
        f"{i+1}. {row['feature']}: "
        f"mean |SHAP| = {row['mean_abs_shap']:.5f}"
    )

lines += [
    "",
    "## Contoh prediksi lokal",
    f"- Grid: {record['cell_id']}",
    f"- Tanggal: {record['date']}",
    f"- Provinsi: {record['province']}",
    f"- Probabilitas model: {probabilities[idx]:.4f}",
    "",
    "## Interpretasi",
    "SHAP positif mendorong log-odds hotspot naik; "
    "SHAP negatif mendorongnya turun.",
    "",
    "SHAP menjelaskan perilaku model, "
    "bukan membuktikan penyebab kebakaran.",
    "",
    "## Keterbatasan",
]

for issue in report["limitations"]:
    lines.append(f"- {issue}")

REPORT_MD.write_text(
    "\n".join(lines),
    encoding="utf-8"
)

print("\n=== Menjelaskan prediksi model dengan SHAP ===")
print("Tabel global:", GLOBAL_CSV)
print("Tabel lokal:", LOCAL_CSV)
print("Laporan:", REPORT_JSON)
print("Ringkasan:", REPORT_MD)
print("Grafik SHAP:", GLOBAL_PNG)
print("Grafik gain:", GAIN_PNG)
print("Grafik lokal:", LOCAL_PNG)
print("MODEL DAN DATASET ASLI TIDAK DIUBAH")
