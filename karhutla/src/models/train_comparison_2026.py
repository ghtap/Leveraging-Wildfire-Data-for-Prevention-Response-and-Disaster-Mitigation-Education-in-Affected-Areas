
from pathlib import Path
import json
import gc
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

DATA = Path("data/processed/dataset_gabungan_2026.parquet")
OUT = Path("results_orang_b")
OUT.mkdir(exist_ok=True)

FEATURES = [
    "n_hotspot", "frp_sum", "precip_mm",
    "t2m_c", "rh", "sm1", "wind_ms"
]

print("MEMBACA DATASET...")
df = pd.read_parquet(
    DATA, columns=["split", "y_7d"] + FEATURES
)

for col in FEATURES:
    df[col] = pd.to_numeric(
        df[col], errors="coerce"
    ).astype("float32")

if df["y_7d"].isna().any():
    raise ValueError("Target mengandung missing value")

train = df["split"] == "train"
valid = df["split"] == "validation"
test = df["split"] == "test"

X_train = df.loc[train, FEATURES].copy()
y_train = df.loc[train, "y_7d"].astype("uint8")

X_val = df.loc[valid, FEATURES].copy()
y_val = df.loc[valid, "y_7d"].astype("uint8")

X_test = df.loc[test, FEATURES].copy()
y_test = df.loc[test, "y_7d"].astype("uint8")

del df
gc.collect()

print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))

def evaluate(name, model):
    print(f"\nEVALUASI {name}...")

    prob_val = model.predict_proba(X_val)[:, 1]

    p, r, thresholds = precision_recall_curve(
        y_val, prob_val
    )
    f1_values = (
        2 * p[:-1] * r[:-1]
        / np.maximum(p[:-1] + r[:-1], 1e-12)
    )
    threshold = float(thresholds[np.argmax(f1_values)])

    prob_test = model.predict_proba(X_test)[:, 1]
    pred = (prob_test >= threshold).astype(int)

    result = {
        "model": name,
        "pr_auc": float(average_precision_score(y_test, prob_test)),
        "roc_auc": float(roc_auc_score(y_test, prob_test)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "threshold_validation": threshold,
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }

    with open(OUT / f"metrics_{name}_2026.json", "w") as f:
        json.dump(result, f, indent=2)

    joblib.dump(model, OUT / f"{name}_2026.joblib")

    print("PR-AUC:", round(result["pr_auc"], 4))
    print("F1:", round(result["f1"], 4))
    print("Recall:", round(result["recall"], 4))
    return result


# MODEL 1: Logistic classifier dengan SGD
# Imputasi dan scaling hanya fit pada train.
logistic = make_pipeline(
    SimpleImputer(strategy="median", add_indicator=True),
    StandardScaler(),
    SGDClassifier(
        loss="log_loss",
        class_weight="balanced",
        max_iter=30,
        tol=1e-3,
        random_state=42,
        n_jobs=4,
    ),
)

print("\nTRAINING LOGISTIC REGRESSION (SGD)...")
logistic.fit(X_train, y_train)
results = [evaluate("LogisticSGD", logistic)]

del logistic
gc.collect()


# MODEL 2: Random Forest
# Sampling training saja untuk membatasi RAM dan waktu.
rng = np.random.default_rng(42)
idx = rng.choice(
    len(X_train),
    size=min(300000, len(X_train)),
    replace=False,
)

forest = make_pipeline(
    SimpleImputer(strategy="median", add_indicator=True),
    RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=4,
    ),
)

print("\nTRAINING RANDOM FOREST...")
print("Sampel train Random Forest:", len(idx))

forest.fit(
    X_train.iloc[idx],
    y_train.iloc[idx],
)

results.append(evaluate("RandomForest", forest))

# Gabungkan metrik LightGBM yang sudah ada
lightgbm_metrics = OUT / "metrics_lightgbm_2026.json"

if lightgbm_metrics.exists():
    with open(lightgbm_metrics) as f:
        results.append(json.load(f))

summary = pd.DataFrame(results)

cols = [
    "model", "pr_auc", "f1",
    "precision", "recall", "roc_auc"
]

print("\n=== PERBANDINGAN MODEL ===")
print(summary[cols].to_string(index=False))

summary.to_csv(
    OUT / "comparison_models_2026.csv",
    index=False
)

print("\nSELESAI. HASIL DISIMPAN DI results_orang_b/")
