
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
)

DATA = Path("data/processed/dataset_fitur_2026_v3.parquet")
OUT = Path("results_orang_b")
OUT.mkdir(exist_ok=True)

MODEL = OUT / "lightgbm_v3_2026.joblib"
METRICS = OUT / "metrics_lightgbm_v3_validation_2026.json"
COMPARE = OUT / "comparison_v2_v3_validation_2026.csv"

for p in [MODEL, METRICS, COMPARE]:
    if p.exists():
        raise SystemExit(
            f"File sudah tersedia: {p}\n"
            "Tidak ditimpa agar eksperimen lama aman."
        )

if not DATA.exists():
    raise SystemExit("Dataset fitur v3 belum ditemukan.")

# Gunakan fitur yang sama dengan LightGBM v2
FEATURES = [
    "rain_prev7_mm",
    "rain_prev14_mm",
    "hotspot_prev7",
    "hotspot_prev14",
    "frp_prev7",
    "temp_prev7_mean",
    "rh_prev7_mean",
    "soil_prev7_mean",
    "wind_prev7_mean",
    "month_2026",
]

print("MEMBACA DATASET FITUR V3...")

df = pd.read_parquet(
    DATA,
    columns=["split", "y_7d"] + FEATURES,
    filters=[("split", "in", ["train", "validation"])],
)

for col in FEATURES:
    df[col] = pd.to_numeric(
        df[col], errors="coerce"
    ).astype("float32")

if df["y_7d"].isna().any():
    raise ValueError("Label y_7d mengandung missing.")

train = df["split"] == "train"
valid = df["split"] == "validation"

X_train = df.loc[train, FEATURES]
y_train = df.loc[train, "y_7d"].astype("uint8")

X_val = df.loc[valid, FEATURES]
y_val = df.loc[valid, "y_7d"].astype("uint8")

if y_train.nunique() != 2 or y_val.nunique() != 2:
    raise ValueError("Train dan validation harus punya dua kelas.")

print("TRAIN:", len(X_train))
print("VALIDATION:", len(X_val))
print("TEST: TIDAK DIGUNAKAN")

print("\nMISSING FITUR VALIDATION (%):")
print((X_val.isna().mean() * 100).round(2))

# Parameter dibuat sama dengan v2
model = lgb.LGBMClassifier(
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=8,
    min_child_samples=150,
    random_state=42,
    n_jobs=4,
    verbosity=-1,
)

print("\nMULAI TRAINING LIGHTGBM V3...")

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="average_precision",
    callbacks=[
        lgb.early_stopping(40, verbose=False)
    ],
)

print("\nEVALUASI VALIDATION...")

prob = model.predict_proba(X_val)[:, 1]

precision, recall, thresholds = precision_recall_curve(
    y_val, prob
)

f1_scores = (
    2 * precision[:-1] * recall[:-1]
    / np.maximum(
        precision[:-1] + recall[:-1],
        1e-12
    )
)

idx = int(np.argmax(f1_scores))

metrics = {
    "model": "LightGBM_v3",
    "evaluation_split": "validation",
    "train_rows": int(len(X_train)),
    "validation_rows": int(len(X_val)),
    "features": FEATURES,
    "pr_auc": float(
        average_precision_score(y_val, prob)
    ),
    "roc_auc": float(
        roc_auc_score(y_val, prob)
    ),
    "f1": float(f1_scores[idx]),
    "precision": float(precision[idx]),
    "recall": float(recall[idx]),
    "threshold": float(thresholds[idx]),
    "best_iteration": int(model.best_iteration_),
}

print("\n=== HASIL LIGHTGBM V3 ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

# Membandingkan validation v2 dan v3
V2_FILE = OUT / "metrics_lightgbm_v2_validation_2026.json"

if not V2_FILE.exists():
    raise SystemExit(
        "Metrik v2 tidak ditemukan. "
        "Model v3 sudah dilatih tetapi belum disimpan."
    )

with V2_FILE.open(encoding="utf-8") as f:
    v2 = json.load(f)

comparison = pd.DataFrame([
    {
        "model": "LightGBM_v2",
        "pr_auc": v2["pr_auc"],
        "f1": v2["f1"],
        "precision": v2["precision"],
        "recall": v2["recall"],
    },
    {
        "model": "LightGBM_v3",
        "pr_auc": metrics["pr_auc"],
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
    },
])

print("\n=== PERBANDINGAN VALIDATION V2 VS V3 ===")
print(comparison.to_string(index=False))

print("\nSELISIH PR-AUC V3 - V2:",
      round(metrics["pr_auc"] - v2["pr_auc"], 4))

print("SELISIH F1 V3 - V2:",
      round(metrics["f1"] - v2["f1"], 4))

joblib.dump(model, MODEL)

with METRICS.open("w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

comparison.to_csv(COMPARE, index=False)

print("\nSELESAI!")
print("MODEL V3 DAN METRIK TERSIMPAN")
print("DATASET ASLI, V2, DAN MODEL LAMA TETAP AMAN")
