
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

DATA = Path("data/processed/dataset_fitur_2026_v3.parquet")
SELECTED = Path("results_orang_b/selected_stage16_2026.json")

MODEL_OUT = Path("results_orang_b/spatial_validation_model_2026.joblib")
REPORT_DIR = Path("reports/orang_b")
REPORT = REPORT_DIR / "spatial_validation_2026.json"
TABLE = REPORT_DIR / "spatial_validation_2026.csv"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

for path in [DATA, SELECTED]:
    if not path.exists():
        raise SystemExit(f"File tidak ditemukan: {path}")

for path in [MODEL_OUT, REPORT, TABLE]:
    if path.exists():
        raise SystemExit(f"File hasil sudah ada, tidak ditimpa: {path}")

with SELECTED.open(encoding="utf-8") as f:
    selected = json.load(f)

FEATURES = selected["features"]

forbidden = {"y_7d", "n_hot_next_7d", "split", "date", "cell_id"}
if forbidden.intersection(FEATURES):
    raise ValueError("Daftar fitur mengandung kolom terlarang")

model_asal = Path(selected["model_path"])
if not model_asal.exists():
    raise SystemExit(f"Model terpilih tidak ditemukan: {model_asal}")

print("=== PAKET 18: SPATIAL VALIDATION ===")
print("Membaca train dan validation saja...")

columns = [
    "cell_id", "province", "lon", "lat",
    "split", "y_7d"
] + FEATURES

df = pd.read_parquet(
    DATA,
    columns=columns,
    filters=[("split", "in", ["train", "validation"])],
)

if df[["cell_id", "province", "lon", "lat"]].isna().any().any():
    raise ValueError("Identitas grid atau koordinat ada yang kosong")

for col in FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

# 1. Susun blok geografis.
# Satu blok berukuran sekitar 0,45 derajat koordinat.
coords = df[
    ["cell_id", "province", "lon", "lat"]
].drop_duplicates("cell_id").copy()

coords["bx"] = np.floor(coords["lon"] / 0.45).astype(int)
coords["by"] = np.floor(coords["lat"] / 0.45).astype(int)

coords["block"] = (
    coords["bx"].astype(str) + "_" +
    coords["by"].astype(str)
)

def stable_key(value):
    return hashlib.sha256(
        ("DIMASTI2026|" + value).encode()
    ).hexdigest()

# Pilih kira-kira 20% blok pada setiap provinsi.
# Jika satu blok melintasi perbatasan, keseluruhan
# blok tetap dikeluarkan dari training.
held_blocks = set()

for province, group in coords.groupby("province"):
    blocks = sorted(group["block"].unique(), key=stable_key)

    if len(blocks) < 2:
        raise ValueError(
            f"Blok spasial terlalu sedikit: {province}"
        )

    n_hold = max(1, round(len(blocks) * 0.20))
    n_hold = min(n_hold, len(blocks) - 1)

    held_blocks.update(blocks[:n_hold])

coords["heldout"] = coords["block"].isin(held_blocks)

held_cells = set(
    coords.loc[coords["heldout"], "cell_id"]
)

heldout = df["cell_id"].isin(held_cells)
is_train = df["split"].eq("train")
is_val = df["split"].eq("validation")

# 2. Train: hanya lokasi bukan holdout.
# Validation: pisahkan lokasi seen dan unseen.
train_mask = is_train & ~heldout
seen_mask = is_val & ~heldout
unseen_mask = is_val & heldout

def get_xy(mask):
    X = df.loc[mask, FEATURES]
    y = df.loc[mask, "y_7d"].astype("uint8")
    if len(X) == 0 or y.nunique() != 2:
        raise ValueError("Subset spasial kosong atau hanya satu kelas")
    return X, y

X_train, y_train = get_xy(train_mask)
X_seen, y_seen = get_xy(seen_mask)
X_unseen, y_unseen = get_xy(unseen_mask)

print("TRAIN lokasi seen:", len(X_train))
print("VALIDATION lokasi seen:", len(X_seen))
print("VALIDATION lokasi unseen:", len(X_unseen))
print("TEST: tidak dibaca")

# 3. Parameter mengikuti model terpilih Paket 16.
# Jumlah tree dikunci agar validation unseen
# tidak digunakan untuk early stopping.
previous = joblib.load(model_asal)
params = previous.get_params()

best_iteration = getattr(previous, "best_iteration_", None)
if best_iteration and best_iteration > 0:
    params["n_estimators"] = int(best_iteration)

params["n_jobs"] = 4
params["random_state"] = 42

model = lgb.LGBMClassifier(**params)

print("\nTRAINING MODEL SPATIAL HOLDOUT...")
model.fit(X_train, y_train)

# 4. Threshold dipilih di validation lokasi seen.
# Lokasi unseen tidak ikut memilih threshold.
prob_seen = model.predict_proba(X_seen)[:, 1]

p, r, thresholds = precision_recall_curve(
    y_seen, prob_seen
)

f1_values = (
    2 * p[:-1] * r[:-1] /
    np.maximum(p[:-1] + r[:-1], 1e-12)
)

threshold = float(thresholds[np.argmax(f1_values)])

def evaluate(name, y, prob):
    pred = (prob >= threshold).astype(int)

    return {
        "subset": name,
        "rows": int(len(y)),
        "positive_rate": float(y.mean()),
        "pr_auc": float(average_precision_score(y, prob)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y, pred, labels=[0, 1]
        ).tolist(),
    }

prob_unseen = model.predict_proba(X_unseen)[:, 1]

results = [
    evaluate("validation_seen", y_seen, prob_seen),
    evaluate("validation_unseen", y_unseen, prob_unseen),
]

summary = pd.DataFrame(results)

print("\n=== HASIL SPATIAL VALIDATION ===")
print(
    summary[
        ["subset", "rows", "positive_rate",
         "pr_auc", "precision", "recall", "f1"]
    ].to_string(index=False)
)

print("\nThreshold dari validation seen:", threshold)
print("Blok holdout:", len(held_blocks))
print("Grid holdout:", len(held_cells))

# 5. Simpan hasil terpisah.
joblib.dump(model, MODEL_OUT)
summary.to_csv(TABLE, index=False)

report = {
    "stage": 18,
    "method": "Geographic block holdout",
    "block_size_degree": 0.45,
    "n_heldout_blocks": len(held_blocks),
    "n_heldout_grids": len(held_cells),
    "threshold_source": "validation_seen",
    "threshold": threshold,
    "model_source": str(model_asal),
    "features": FEATURES,
    "results": results,
    "limitations": [
        "Parameter model sebelumnya telah dipilih dengan validation Agustus",
        "Karena itu audit spasial ini diagnostik, bukan test independen murni",
        "Blok geografis berdekatan masih dapat berbagi kondisi lingkungan",
        "Data test September tidak digunakan dalam script ini",
    ],
}

with REPORT.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("\n=== PAKET 18 SELESAI ===")
print("Model diagnostik:", MODEL_OUT)
print("Tabel:", TABLE)
print("Laporan:", REPORT)
print("Data dan model lama tetap aman")
