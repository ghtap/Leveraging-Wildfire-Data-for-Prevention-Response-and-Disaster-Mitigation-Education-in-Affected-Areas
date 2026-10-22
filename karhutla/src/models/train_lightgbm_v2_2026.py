
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

DATA = Path("data/processed/dataset_fitur_2026_v2.parquet")
OUT = Path("results_orang_b")
OUT.mkdir(exist_ok=True)

MODEL_PATH = OUT / "lightgbm_v2_2026.joblib"
METRIC_PATH = OUT / "metrics_lightgbm_v2_validation_2026.json"

if not DATA.exists():
    raise SystemExit("Dataset fitur v2 belum dibuat.")

if MODEL_PATH.exists() or METRIC_PATH.exists():
    raise SystemExit(
        "Hasil LightGBM v2 sudah ada. Tidak ditimpa."
    )

# HANYA gunakan fitur yang diketahui sebelum tanggal prediksi.
# rain_prev30_mm sengaja belum digunakan karena riwayat
# 30 hari pada validation/test memiliki jeda tanggal.
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

print("MEMBACA FITUR LIGHTGBM V2...")

df = pd.read_parquet(
    DATA,
    columns=["split", "y_7d"] + FEATURES,
)

df = df[df["split"].isin(["train", "validation"])]

for col in FEATURES:
    df[col] = pd.to_numeric(
        df[col], errors="coerce"
    ).astype("float32")

train = df["split"] == "train"
valid = df["split"] == "validation"

X_train = df.loc[train, FEATURES]
y_train = df.loc[train, "y_7d"].astype("uint8")

X_val = df.loc[valid, FEATURES]
y_val = df.loc[valid, "y_7d"].astype("uint8")

if y_train.nunique() != 2 or y_val.nunique() != 2:
    raise ValueError("Train/validation harus memiliki dua kelas.")

print("TRAIN:", len(X_train))
print("VALIDATION:", len(X_val))
print("TEST: tidak dibaca atau digunakan")

print("\nMULAI TRAINING LIGHTGBM V2...")

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

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="average_precision",
    callbacks=[
        lgb.early_stopping(40, verbose=False),
    ],
)

prob = model.predict_proba(X_val)[:, 1]

precision, recall, thresholds = precision_recall_curve(
    y_val, prob
)

f1_values = (
    2 * precision[:-1] * recall[:-1]
    / np.maximum(
        precision[:-1] + recall[:-1],
        1e-12
    )
)

idx = int(np.argmax(f1_values))

metrics = {
    "model": "LightGBM_v2",
    "evaluation_split": "validation",
    "features": FEATURES,
    "train_rows": int(len(X_train)),
    "validation_rows": int(len(X_val)),
    "pr_auc": float(
        average_precision_score(y_val, prob)
    ),
    "roc_auc": float(
        roc_auc_score(y_val, prob)
    ),
    "f1": float(f1_values[idx]),
    "precision": float(precision[idx]),
    "recall": float(recall[idx]),
    "threshold": float(thresholds[idx]),
    "best_iteration": int(model.best_iteration_),
}

print("\n=== HASIL LIGHTGBM V2 (VALIDATION) ===")
for key, value in metrics.items():
    print(f"{key}: {value}")

# Bandingkan dengan hasil validation v1 yang sudah ada.
comparison = OUT / "validation_comparison_2026.csv"

if comparison.exists():
    old = pd.read_csv(comparison)
    ref = old.loc[
        old["model"] == "LightGBM"
    ]

    if not ref.empty:
        old_ap = float(ref.iloc[0]["validation_pr_auc"])
        old_f1 = float(ref.iloc[0]["validation_f1"])

        print("\n=== V1 VS V2 ===")
        print("PR-AUC v1:", round(old_ap, 4))
        print("PR-AUC v2:", round(metrics["pr_auc"], 4))
        print("F1 v1:", round(old_f1, 4))
        print("F1 v2:", round(metrics["f1"], 4))

joblib.dump(model, MODEL_PATH)

with METRIC_PATH.open("w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("\nMODEL V2 DAN METRIK TERSIMPAN")
print("DATA ASLI DAN MODEL V1 TIDAK BERUBAH")
