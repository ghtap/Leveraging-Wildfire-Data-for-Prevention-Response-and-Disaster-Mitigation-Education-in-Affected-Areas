
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

DATA = Path("data/processed/dataset_fitur_2026_v3.parquet")
OUT = Path("results_orang_b")
OUT.mkdir(exist_ok=True)

BASE_MODEL = OUT / "lightgbm_v3_2026.joblib"
NEW_MODEL = OUT / "lightgbm_tuned_2026.joblib"
REPORT = OUT / "tuning_stage16_2026.csv"
THRESHOLDS = OUT / "threshold_stage16_2026.csv"
SELECTION = OUT / "selected_stage16_2026.json"

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

for path in [DATA, BASE_MODEL]:
    if not path.exists():
        raise SystemExit(f"File tidak ditemukan: {path}")

for path in [NEW_MODEL, REPORT, THRESHOLDS, SELECTION]:
    if path.exists():
        raise SystemExit(
            f"Hasil paket 16 sudah ada: {path}\n"
            "Tidak akan menimpa eksperimen sebelumnya."
        )

print("=== PAKET 16: TUNING DAN THRESHOLD ===")
print("Membaca TRAIN dan VALIDATION saja...")

df = pd.read_parquet(
    DATA,
    columns=["split", "y_7d"] + FEATURES,
    filters=[("split", "in", ["train", "validation"])],
)

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
    raise ValueError("Train dan validation harus berisi dua kelas.")

print("TRAIN:", len(X_train))
print("VALIDATION:", len(X_val))
print("TEST: TIDAK DIGUNAKAN")

COMMON = dict(
    n_estimators=600,
    learning_rate=0.05,
    random_state=42,
    n_jobs=4,
    verbosity=-1,
)

CANDIDATES = {
    "regularized": dict(
        num_leaves=15,
        max_depth=6,
        min_child_samples=300,
        reg_lambda=5.0,
        colsample_bytree=0.85,
    ),
    "weighted": dict(
        num_leaves=31,
        max_depth=8,
        min_child_samples=150,
        reg_lambda=2.0,
        scale_pos_weight=5.0,
    ),
}

def evaluate(name, model):
    prob = model.predict_proba(X_val)[:, 1]

    p, r, thresholds = precision_recall_curve(
        y_val, prob
    )

    f1 = (
        2 * p[:-1] * r[:-1]
        / np.maximum(p[:-1] + r[:-1], 1e-12)
    )

    best_idx = int(np.argmax(f1))

    result = {
        "model": name,
        "pr_auc": float(
            average_precision_score(y_val, prob)
        ),
        "roc_auc": float(
            roc_auc_score(y_val, prob)
        ),
        "f1": float(f1[best_idx]),
        "precision": float(p[best_idx]),
        "recall": float(r[best_idx]),
        "threshold_f1": float(thresholds[best_idx]),
    }

    # Alternatif threshold dengan recall setidaknya 70%.
    target_idx = np.where(r[:-1] >= 0.70)[0]

    if len(target_idx):
        idx = int(target_idx[-1])

        result["threshold_recall70"] = float(
            thresholds[idx]
        )
        result["precision_recall70"] = float(p[idx])
        result["recall_recall70"] = float(r[idx])
    else:
        result["threshold_recall70"] = None
        result["precision_recall70"] = None
        result["recall_recall70"] = None

    print(
        f"{name}: PR-AUC={result['pr_auc']:.4f} "
        f"F1={result['f1']:.4f} "
        f"Recall={result['recall']:.4f}"
    )

    return result


print("\n1. EVALUASI V3 YANG SUDAH ADA")

best_model = joblib.load(BASE_MODEL)
best_result = evaluate("LightGBM_v3", best_model)
best_path = BASE_MODEL

results = [best_result]

print("\n2. MENCOBA PARAMETER BARU")

for name, params in CANDIDATES.items():
    print(f"\nTRAINING {name}...")

    candidate = lgb.LGBMClassifier(
        **COMMON,
        **params,
    )

    candidate.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[
            lgb.early_stopping(40, verbose=False)
        ],
    )

    result = evaluate(name, candidate)
    results.append(result)

    if result["pr_auc"] > best_result["pr_auc"]:
        best_model = candidate
        best_result = result
        best_path = NEW_MODEL

print("\n3. PERBANDINGAN VALIDATION")

table = pd.DataFrame(results).sort_values(
    "pr_auc", ascending=False
)

print(
    table[
        ["model", "pr_auc", "f1", "precision", "recall"]
    ].to_string(index=False)
)

print("\n4. MODEL TERPILIH")
print("Model:", best_result["model"])
print("PR-AUC:", round(best_result["pr_auc"], 4))

print("\n5. OPTIMASI THRESHOLD")

threshold_table = pd.DataFrame([
    {
        "mode": "F1_optimal",
        "threshold": best_result["threshold_f1"],
        "precision_validation": best_result["precision"],
        "recall_validation": best_result["recall"],
    },
    {
        "mode": "Recall_minimum_70",
        "threshold": best_result["threshold_recall70"],
        "precision_validation": best_result["precision_recall70"],
        "recall_validation": best_result["recall_recall70"],
    },
])

print(threshold_table.to_string(index=False))

if best_path == NEW_MODEL:
    joblib.dump(best_model, NEW_MODEL)

table.to_csv(REPORT, index=False)
threshold_table.to_csv(THRESHOLDS, index=False)

selection = {
    "stage": 16,
    "model": best_result["model"],
    "model_path": best_path.as_posix(),
    "features": FEATURES,
    "selection_metric": "validation_pr_auc",
    "validation_pr_auc": best_result["pr_auc"],
    "validation_f1": best_result["f1"],
    "threshold_f1": best_result["threshold_f1"],
    "threshold_recall70": best_result["threshold_recall70"],
    "train_rows": int(len(X_train)),
    "validation_rows": int(len(X_val)),
    "test_used": False,
    "note": (
        "Threshold dan parameter dipilih pada validation. "
        "Hasil ini bukan evaluasi independen final."
    ),
}

with SELECTION.open("w", encoding="utf-8") as f:
    json.dump(selection, f, indent=2)

print("\nTuning dan optimasi threshold selesai.")
print("Model:", best_path)
print("Laporan tuning:", REPORT)
print("Laporan threshold:", THRESHOLDS)
print("Pilihan model:", SELECTION)
print("DATA ASLI DAN MODEL SEBELUMNYA TETAP AMAN")
