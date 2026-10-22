
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

DATA = Path("data/processed/dataset_gabungan_2026.parquet")
OUT = Path("results_orang_b")

FEATURES = [
    "n_hotspot",
    "frp_sum",
    "precip_mm",
    "t2m_c",
    "rh",
    "sm1",
    "wind_ms",
]

MODELS = {
    "LogisticSGD": OUT / "LogisticSGD_2026.joblib",
    "LightGBM": OUT / "baseline_lightgbm_2026.joblib",
    "RandomForest": OUT / "RandomForest_2026.joblib",
}

print("MEMBACA VALIDATION 2026...")

df = pd.read_parquet(
    DATA,
    columns=["split", "y_7d"] + FEATURES,
    filters=[("split", "==", "validation")],
)

if df.empty:
    raise ValueError("Data validation tidak ditemukan.")

X = df[FEATURES].apply(
    pd.to_numeric, errors="coerce"
).astype("float32")

y = df["y_7d"].astype("int8")

if y.nunique() != 2:
    raise ValueError("Validation harus memiliki dua kelas.")

hasil = []

for nama, lokasi in MODELS.items():
    if not lokasi.exists():
        raise FileNotFoundError(lokasi)

    print(f"\nEVALUASI VALIDATION: {nama}")

    model = joblib.load(lokasi)
    prob = model.predict_proba(X)[:, 1]

    precision, recall, thresholds = (
        precision_recall_curve(y, prob)
    )

    f1 = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(
            precision[:-1] + recall[:-1],
            1e-12
        )
    )

    idx = int(np.argmax(f1))

    hasil.append({
        "model": nama,
        "validation_pr_auc": float(
            average_precision_score(y, prob)
        ),
        "validation_roc_auc": float(
            roc_auc_score(y, prob)
        ),
        "validation_f1": float(f1[idx]),
        "validation_precision": float(precision[idx]),
        "validation_recall": float(recall[idx]),
        "threshold": float(thresholds[idx]),
    })

    del model, prob

summary = (
    pd.DataFrame(hasil)
    .sort_values(
        ["validation_pr_auc", "validation_f1"],
        ascending=False
    )
)

print("\n=== PERBANDINGAN VALIDATION ===")
print(summary.to_string(index=False))

summary.to_csv(
    OUT / "validation_comparison_2026.csv",
    index=False
)

terpilih = summary.iloc[0].to_dict()

with open(
    OUT / "selected_candidate_2026.json", "w"
) as f:
    json.dump(terpilih, f, indent=2)

print("\nKANDIDAT MODEL BERDASARKAN VALIDATION:")
print(terpilih["model"])

print("\nSELESAI. DATA ASLI TIDAK DIUBAH.")
