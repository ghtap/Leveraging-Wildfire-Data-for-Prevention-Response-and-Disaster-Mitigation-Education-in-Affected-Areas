
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
)

DATA = Path("data/processed/dataset_fitur_2026_v3.parquet")
SELECTION = Path("results_orang_b/selected_stage16_2026.json")

REPORT_DIR = Path("reports/orang_b")
FIGURE_DIR = Path("figures/orang_b")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / "stage19_evaluation_2026.json"
SUMMARY = REPORT_DIR / "stage19_metrics_2026.csv"
PROVINCE = REPORT_DIR / "stage19_province_2026.csv"
PREDICTIONS = REPORT_DIR / "stage19_predictions_2026.parquet"
PR_FIGURE = FIGURE_DIR / "stage19_pr_curve_2026.png"
CM_FIGURE = FIGURE_DIR / "stage19_confusion_matrix_2026.png"

outputs = [
    REPORT, SUMMARY, PROVINCE,
    PREDICTIONS, PR_FIGURE, CM_FIGURE
]

for p in [DATA, SELECTION]:
    if not p.exists():
        raise SystemExit(f"File belum tersedia: {p}")

for p in outputs:
    if p.exists():
        raise SystemExit(
            f"Output sudah ada: {p}\n"
            "Tidak ditimpa agar hasil sebelumnya aman."
        )

with SELECTION.open(encoding="utf-8") as f:
    selected = json.load(f)

MODEL = Path(selected["model_path"])
FEATURES = selected["features"]

if not MODEL.exists():
    raise SystemExit(f"Model terpilih tidak ditemukan: {MODEL}")

forbidden = {
    "y_7d", "n_hot_next_7d",
    "date", "split", "cell_id"
}

if forbidden.intersection(FEATURES):
    raise ValueError("Ada fitur berpotensi leakage.")

print("=== PAKET 19: EVALUASI DAN ERROR ANALYSIS ===")
print("Model:", MODEL)
print("Membaca data test September 2026...")

columns = [
    "cell_id", "date", "province", "split", "y_7d"
] + FEATURES

df = pd.read_parquet(
    DATA,
    columns=columns,
    filters=[("split", "==", "test")],
)

df = df.loc[df["split"] == "test"].reset_index(drop=True)

if df.empty:
    raise ValueError("Data test kosong.")

df["date"] = pd.to_datetime(df["date"])

if df["date"].min() < pd.Timestamp("2026-09-01"):
    raise ValueError("Test berisi tanggal sebelum September.")

if df["date"].max() > pd.Timestamp("2026-09-30"):
    raise ValueError("Test berisi tanggal setelah September.")

if df["y_7d"].isna().any():
    raise ValueError("Label test memiliki missing value.")

y = df["y_7d"].astype("uint8").to_numpy()

if set(np.unique(y)) != {0, 1}:
    raise ValueError("Label test harus terdiri dari 0 dan 1.")

X = df[FEATURES].apply(
    pd.to_numeric, errors="coerce"
).astype("float32")

model = joblib.load(MODEL)

if hasattr(model, "feature_name_"):
    if list(model.feature_name_) != FEATURES:
        raise ValueError("Urutan fitur berbeda dari model.")

print("Menghitung probabilitas...")
prob = model.predict_proba(X)[:, 1]

if not np.isfinite(prob).all():
    raise ValueError("Model menghasilkan probabilitas tidak valid.")

thresholds = {
    "F1_optimal": selected["threshold_f1"],
}

if selected.get("threshold_recall70") is not None:
    thresholds["Recall_minimum_70"] = (
        selected["threshold_recall70"]
    )

test_pr_auc = float(average_precision_score(y, prob))
test_roc_auc = float(roc_auc_score(y, prob))
prevalence = float(np.mean(y))

def calculate_metrics(y_true, probabilities, threshold):
    predicted = (
        probabilities >= threshold
    ).astype("uint8")

    tn, fp, fn, tp = confusion_matrix(
        y_true, predicted, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "rows": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "precision": float(precision_score(
            y_true, predicted, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, predicted, zero_division=0
        )),
        "f1": float(f1_score(
            y_true, predicted, zero_division=0
        )),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

records = []

for name, threshold in thresholds.items():
    metric = calculate_metrics(y, prob, threshold)
    metric["mode"] = name
    metric["pr_auc"] = test_pr_auc
    metric["roc_auc"] = test_roc_auc
    records.append(metric)

summary = pd.DataFrame(records)

print("\n=== HASIL TEST SEPTEMBER ===")
print("PR-AUC:", round(test_pr_auc, 4))
print("ROC-AUC:", round(test_roc_auc, 4))
print("Prevalensi positif:", round(prevalence, 4))

print(summary[
    ["mode", "threshold", "precision",
     "recall", "f1", "fp", "fn"]
].to_string(index=False))

# Analisis per provinsi menggunakan threshold F1
threshold_main = float(thresholds["F1_optimal"])
pred = (prob >= threshold_main).astype("uint8")

province_rows = []

for province, indices in df.groupby(
    "province", sort=True
).indices.items():

    yt = y[indices]
    pp = prob[indices]

    if len(np.unique(yt)) != 2:
        raise ValueError(
            f"Provinsi {province} hanya memiliki satu kelas."
        )

    met = calculate_metrics(
        yt, pp, threshold_main
    )

    province_rows.append({
        "province": province,
        "pr_auc": float(
            average_precision_score(yt, pp)
        ),
        **met,
    })

province_table = pd.DataFrame(province_rows)

print("\n=== HASIL PER PROVINSI ===")
print(province_table[
    ["province", "rows", "positive_rate",
     "pr_auc", "recall", "f1", "fp", "fn"]
].to_string(index=False))

# Simpan probabilitas untuk kebutuhan analisis Orang C.
prediction_table = df[
    ["cell_id", "date", "province"]
].copy()

prediction_table["y_actual"] = y
prediction_table["probability"] = prob.astype("float32")
prediction_table["predicted_f1"] = pred

prediction_table.to_parquet(
    PREDICTIONS, index=False
)

summary.to_csv(SUMMARY, index=False)
province_table.to_csv(PROVINCE, index=False)

# Grafik Precision-Recall
precision_curve, recall_curve, _ = (
    precision_recall_curve(y, prob)
)

plt.figure(figsize=(8, 5))
plt.plot(
    recall_curve, precision_curve,
    label=f"Model (AP={test_pr_auc:.3f})"
)
plt.axhline(
    prevalence,
    linestyle="--",
    label=f"Prevalensi={prevalence:.3f}"
)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - September 2026")
plt.legend()
plt.tight_layout()
plt.savefig(PR_FIGURE, dpi=170)
plt.close()

# Confusion matrix threshold F1
cm = confusion_matrix(y, pred, labels=[0, 1])

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.colorbar(label="Jumlah observasi")
plt.xticks([0, 1], ["Prediksi 0", "Prediksi 1"])
plt.yticks([0, 1], ["Aktual 0", "Aktual 1"])

for i in range(2):
    for j in range(2):
        plt.text(
            j, i, f"{cm[i, j]:,}",
            ha="center", va="center",
            color="white" if cm[i, j] > cm.max()/2 else "black"
        )

plt.title("Confusion Matrix - Test September 2026")
plt.tight_layout()
plt.savefig(CM_FIGURE, dpi=170)
plt.close()

report = {
    "stage": 19,
    "evaluation_period": "2026-09-01 to 2026-09-30",
    "model": selected["model"],
    "model_path": str(MODEL),
    "features": FEATURES,
    "pr_auc": test_pr_auc,
    "roc_auc": test_roc_auc,
    "positive_prevalence": prevalence,
    "thresholds_source": "validation Agustus 2026",
    "threshold_results": records,
    "test_is_untouched": False,
    "limitations": [
        "Test September pernah diperiksa pada eksperimen baseline",
        "Test tidak boleh dipakai untuk tuning lebih lanjut",
        "Ketersediaan fitur cuaca secara real-time belum diverifikasi",
        "Label FIRMS adalah deteksi hotspot, bukan konfirmasi kebakaran",
        "Generalisasi lintas tahun belum diuji"
    ],
}

REPORT.write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("\n=== Evaluasi test dan analisis kesalahan model ===")
print("Laporan:", REPORT)
print("Tabel metrik:", SUMMARY)
print("Per provinsi:", PROVINCE)
print("Prediksi:", PREDICTIONS)
print("Grafik PR:", PR_FIGURE)
print("Confusion matrix:", CM_FIGURE)
print("Dataset asli dan model lama tidak berubah.")
