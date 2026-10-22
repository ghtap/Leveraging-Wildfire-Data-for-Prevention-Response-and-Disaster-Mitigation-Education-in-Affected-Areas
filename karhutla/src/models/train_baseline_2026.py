
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import pyarrow.parquet as pq

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
)

# 1. Lokasi data dan hasil model
DATA = Path("data/processed/dataset_gabungan_2026.parquet")
OUT = Path("results_orang_b")
OUT.mkdir(exist_ok=True)

if not DATA.exists():
    raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA}")

# 2. Pilih fitur yang tersedia
schema = pq.ParquetFile(DATA).schema_arrow.names

kandidat = [
    "n_hotspot",
    "frp_sum",
    "precip_mm",
    "t2m_c",
    "rh",
    "sm1",
    "wind_ms",
]

fitur = [c for c in kandidat if c in schema]
wajib = {"split", "y_7d"}

if not wajib.issubset(schema):
    raise ValueError(f"Kolom wajib tidak lengkap: {wajib - set(schema)}")

if len(fitur) < 5:
    raise ValueError(f"Fitur yang ditemukan terlalu sedikit: {fitur}")

print("FITUR YANG DIGUNAKAN:", fitur)
print("MEMBACA DATASET...")

df = pd.read_parquet(
    DATA,
    columns=["split", "y_7d"] + fitur
)

# 3. Persiapan data
df["split"] = (
    df["split"].astype(str).str.lower()
    .replace({
        "training": "train",
        "val": "validation",
        "valid": "validation",
        "testing": "test",
    })
)

for col in fitur:
    df[col] = pd.to_numeric(
        df[col], errors="coerce"
    ).astype("float32")

if df["y_7d"].isna().any():
    raise ValueError("Target y_7d mengandung missing value")

df["y_7d"] = df["y_7d"].astype("uint8")

train = df["split"] == "train"
valid = df["split"] == "validation"
test = df["split"] == "test"

if not all([train.any(), valid.any(), test.any()]):
    raise ValueError(
        f"Split tidak lengkap:\n{df['split'].value_counts()}"
    )

X_train, y_train = df.loc[train, fitur], df.loc[train, "y_7d"]
X_val, y_val = df.loc[valid, fitur], df.loc[valid, "y_7d"]
X_test, y_test = df.loc[test, fitur], df.loc[test, "y_7d"]

print("TRAIN:", len(X_train))
print("VALIDATION:", len(X_val))
print("TEST:", len(X_test))

# 4. Latih LightGBM
model = lgb.LGBMClassifier(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=8,
    random_state=42,
    n_jobs=4,
    verbosity=-1,
)

print("\nMULAI TRAINING LIGHTGBM...")

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="average_precision",
    callbacks=[
        lgb.early_stopping(30, verbose=False)
    ],
)

# 5. Pilih threshold berdasarkan VALIDATION
prob_val = model.predict_proba(X_val)[:, 1]

p, r, thresholds = precision_recall_curve(
    y_val, prob_val
)

f1_vals = (
    2 * p[:-1] * r[:-1] /
    np.maximum(p[:-1] + r[:-1], 1e-12)
)

idx = int(np.argmax(f1_vals))
threshold = float(thresholds[idx])

# 6. Evaluasi TEST tanpa tuning ulang
prob_test = model.predict_proba(X_test)[:, 1]
pred_test = (prob_test >= threshold).astype(int)

metrics = {
    "model": "LightGBM",
    "threshold_validation": threshold,
    "features": fitur,
    "train_rows": int(len(X_train)),
    "validation_rows": int(len(X_val)),
    "test_rows": int(len(X_test)),
    "pr_auc": float(average_precision_score(y_test, prob_test)),
    "roc_auc": float(roc_auc_score(y_test, prob_test)),
    "precision": float(precision_score(y_test, pred_test, zero_division=0)),
    "recall": float(recall_score(y_test, pred_test, zero_division=0)),
    "f1": float(f1_score(y_test, pred_test, zero_division=0)),
    "confusion_matrix": confusion_matrix(y_test, pred_test).tolist(),
}

# 7. Simpan model dan hasil evaluasi
joblib.dump(
    model,
    OUT / "baseline_lightgbm_2026.joblib"
)

with open(OUT / "metrics_lightgbm_2026.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n=== HASIL MODEL LIGHTGBM ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

print("\nMODEL DAN METRIK BERHASIL DISIMPAN")
