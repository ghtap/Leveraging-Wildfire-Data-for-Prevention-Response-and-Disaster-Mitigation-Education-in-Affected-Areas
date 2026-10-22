
from pathlib import Path
import hashlib
import json
import shutil

import joblib
import numpy as np
import pandas as pd

ROOT = Path.cwd()
SELECTED = ROOT / "results_orang_b/selected_stage16_2026.json"
SHAP_LOCAL = ROOT / "reports/orang_b/stage20_shap_local_2026.csv"
SHAP_REPORT = ROOT / "reports/orang_b/stage20_shap_2026.json"

FINAL_DIR = ROOT / "models_final"
NOTEBOOK_DIR = ROOT / "notebooks"

FINAL_MODEL = FINAL_DIR / "lightgbm_candidate_2026.joblib"
METADATA = FINAL_DIR / "model_spec_2026.json"
README = FINAL_DIR / "README_MODEL_2026.md"
NOTEBOOK = NOTEBOOK_DIR / "Analisis_Model_2026.ipynb"

OUTPUTS = [FINAL_MODEL, METADATA, README, NOTEBOOK]

print("=== MODEL DAN NOTEBOOK ===")

for p in [SELECTED, SHAP_LOCAL, SHAP_REPORT]:
    if not p.is_file():
        raise SystemExit(f"File sumber tidak ditemukan: {p}")

for p in OUTPUTS:
    if p.exists():
        raise SystemExit(
            f"File hasil sudah ada: {p}\n"
            "Tidak akan menimpanya."
        )

with SELECTED.open(encoding="utf-8") as f:
    selected = json.load(f)

with SHAP_REPORT.open(encoding="utf-8") as f:
    shap_report = json.load(f)

model_path = Path(selected["model_path"])
if not model_path.is_absolute():
    model_path = ROOT / model_path

if not model_path.is_file():
    raise SystemExit(f"Model tidak ditemukan: {model_path}")

features = selected["features"]

forbidden = {
    "y_7d", "n_hot_next_7d", "split",
    "date", "cell_id"
}
if forbidden.intersection(features):
    raise ValueError("Ada fitur masa depan/identitas yang terlarang.")

model = joblib.load(model_path)

if list(model.booster_.feature_name()) != features:
    raise ValueError("Nama atau urutan fitur berbeda dari model.")

# 1. Periksa prediksi contoh yang sudah dijelaskan oleh SHAP
print("\n1. VERIFIKASI PREDIKSI MODEL...")

local = pd.read_csv(SHAP_LOCAL)

if local["feature"].duplicated().any():
    raise ValueError("Ada nama fitur duplikat pada SHAP lokal.")

local = local.set_index("feature")

missing = set(features) - set(local.index)
if missing:
    raise ValueError(f"Fitur SHAP tidak lengkap: {missing}")

values = {
    feature: pd.to_numeric(
        local.loc[feature, "value"], errors="coerce"
    )
    for feature in features
}

example = pd.DataFrame([values], columns=features).astype("float32")
prob = float(model.predict_proba(example)[0, 1])

expected = float(
    shap_report["local_example"]["predicted_probability"]
)

print("Probabilitas model:", prob)
print("Probabilitas laporan SHAP:", expected)

if not np.isclose(prob, expected, atol=1e-5, rtol=0):
    raise ValueError(
        "Prediksi model tidak sesuai laporan SHAP. "
        "Periksa model dan urutan fitur."
    )

# 2. Verifikasi threshold dan identitas model
print("\n2. MENYIAPKAN KONFIGURASI...")

threshold_f1 = float(selected["threshold_f1"])
threshold_r70 = selected.get("threshold_recall70")

if threshold_r70 is not None:
    threshold_r70 = float(threshold_r70)

for value in [threshold_f1, threshold_r70]:
    if value is not None and not 0 < value < 1:
        raise ValueError("Threshold berada di luar rentang 0 sampai 1.")

digest = hashlib.sha256()
with model_path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)

spec = {
    "model": "LightGBM regularized",
    "status": "RESEARCH_CANDIDATE",
    "tahun": 2026,
    "wilayah": "6 provinsi penelitian DIMAS-TI",
    "target": "y_7d",
    "prediction_horizon_days": 7,
    "feature_order": features,
    "threshold_f1": threshold_f1,
    "threshold_recall70": threshold_r70,
    "validation_pr_auc": selected["validation_pr_auc"],
    "validation_f1": selected["validation_f1"],
    "source_model": str(model_path.relative_to(ROOT)),
    "sha256": digest.hexdigest(),
    "input_requirement": (
        "Seluruh fitur harus dihitung sebelum waktu "
        "prediksi, menggunakan riwayat tanggal yang benar."
    ),
    "missing_policy": (
        "NaN boleh untuk LightGBM; jangan otomatis isi nol."
    ),
    "important_limitations": [
        "Test September sudah pernah terekspos saat eksperimen baseline",
        "Ketersediaan CHIRPS dan ERA5 secara real-time belum diverifikasi",
        "Model baru diuji pada tahun 2026",
        "Model memprediksi hotspot, bukan konfirmasi kebakaran",
    ],
    "verified_shap_example_probability": prob,
}

# 3. Membuat notebook laporan secara otomatis
print("\n3. MEMBUAT NOTEBOOK ANALISIS...")

def cell_markdown(content):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": content.splitlines(keepends=True),
    }

def cell_code(content):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": content.splitlines(keepends=True),
    }

cells = [
    cell_markdown(
        "# Analisis Pemodelan Karhutla DIMAS-TI 2026\n\n"
        "Notebook ini menyajikan **metodologi dan hasil Orang B**. "
        "Training dan perhitungan SHAP tetap berada dalam skrip "
        "Python terpisah agar dapat direproduksi.\n\n"
        "Urutan: dataset → perbandingan model → tuning → "
        "spatial validation → evaluasi → SHAP."
    ),
    cell_code(
        "from pathlib import Path\n"
        "import json\n"
        "import pandas as pd\n"
        "from IPython.display import display, Image, Markdown\n"
        "ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] "
        "if (p / 'models_final/model_spec_2026.json').exists()), None)\n"
        "assert ROOT is not None, 'Buka notebook dari folder proyek karhutla'\n"
        "spec = json.loads((ROOT / 'models_final/model_spec_2026.json')"
        ".read_text(encoding='utf-8'))\n"
        "print('Model:', spec['model'])\n"
        "print('Target:', spec['target'])\n"
        "print('Fitur:', spec['feature_order'])\n"
    ),
    cell_markdown(
        "## 1. Perbandingan LightGBM v2 dan v3\n\n"
        "Feature engineering v3 menggunakan tanggal jeda sebagai "
        "sumber riwayat, tanpa menjadikannya baris training."
    ),
    cell_code(
        "path = ROOT / 'results_orang_b/comparison_v2_v3_validation_2026.csv'\n"
        "display(pd.read_csv(path))\n"
    ),
    cell_markdown(
        "## 2. Hyperparameter tuning\n\n"
        "Pemilihan kandidat dilakukan dengan **PR-AUC validation**, "
        "bukan berdasarkan data test."
    ),
    cell_code(
        "display(pd.read_csv(ROOT / 'results_orang_b/tuning_stage16_2026.csv'))\n"
        "display(pd.read_csv(ROOT / 'results_orang_b/threshold_stage16_2026.csv'))\n"
    ),
    cell_markdown(
        "## 3. Spatial Validation\n\n"
        "Membandingkan lokasi yang digunakan saat training "
        "dengan lokasi geografis yang ditahan dari training."
    ),
    cell_code(
        "display(pd.read_csv(ROOT / 'reports/orang_b/spatial_validation_2026.csv'))\n"
    ),
    cell_markdown(
        "## 4. Evaluasi Test September 2026\n\n"
        "Hasil test ini **bukan untouched holdout**, karena September "
        "sudah pernah dilihat dalam eksperimen baseline."
    ),
    cell_code(
        "display(pd.read_csv(ROOT / 'reports/orang_b/stage19_metrics_2026.csv'))\n"
        "display(Image(filename=str(ROOT / 'figures/orang_b/stage19_pr_curve_2026.png')))\n"
        "display(Image(filename=str(ROOT / 'figures/orang_b/stage19_confusion_matrix_2026.png')))\n"
    ),
    cell_markdown(
        "## 5. Metode SHAP\n\n"
        "Metode SHAP menjelaskan kontribusi fitur terhadap skor "
        "model. Pada LightGBM biner, kontribusi di sini menggunakan "
        "skala **log-odds**:\n\n"
        "$$z=f_0+\\sum_j\\phi_j$$\n\n"
        "$$P(y=1)=1/(1+e^{-z})$$\n\n"
        "SHAP menjelaskan perilaku model, **bukan membuktikan "
        "hubungan sebab-akibat**."
    ),
    cell_code(
        "display(pd.read_csv(ROOT / 'reports/orang_b/stage20_shap_global_2026.csv'))\n"
        "display(Image(filename=str(ROOT / 'figures/orang_b/stage20_shap_global_2026.png')))\n"
        "display(Image(filename=str(ROOT / 'figures/orang_b/stage20_feature_importance_2026.png')))\n"
    ),
    cell_markdown(
        "## 6. Penjelasan Satu Prediksi dengan SHAP\n\n"
        "SHAP positif meningkatkan log-odds hotspot menurut model. "
        "Nilai negatif menurunkannya."
    ),
    cell_code(
        "display(pd.read_csv(ROOT / 'reports/orang_b/stage20_shap_local_2026.csv'))\n"
        "display(Image(filename=str(ROOT / 'figures/orang_b/stage20_shap_local_2026.png')))\n"
    ),
    cell_markdown(
        "## 7. Keterbatasan Penelitian\n\n"
        "- Perubahan distribusi musiman antara train dan validation.\n"
        "- Beberapa nilai ERA5 masih kosong.\n"
        "- Keterlambatan publikasi GEE belum diaudit secara operasional.\n"
        "- Generalisasi lintas tahun belum dibuktikan.\n"
        "- Interpretasi SHAP bukan hubungan kausal."
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

readme = """# Model Risiko Hotspot 2026 (Orang C)

Status: RESEARCH CANDIDATE (belum tervalidasi operasional)

## Input
Gunakan urutan fitur persis seperti dalam model_spec_2026.json.
Fitur rolling harus dihitung dari tanggal sebelum prediksi.
Jangan masukkan y_7d atau n_hot_next_7d sebagai fitur.

## Output
predict_proba(X)[:, 1] menghasilkan probabilitas model
untuk label hotspot tujuh hari berikutnya.

## Threshold
threshold_f1: prioritas keseimbangan precision dan recall.
threshold_recall70: alternatif sensitivitas lebih tinggi,
dengan risiko false alarm lebih banyak.

## Penting
Model ini belum membuktikan ketersediaan data cuaca
secara real-time, dan bukan alat konfirmasi kebakaran.
"""

# 4. Simpan hasil tanpa mengganti artefak lama
print("\n4. MENYIMPAN MODEL DAN NOTEBOOK...")

FINAL_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

shutil.copy2(model_path, FINAL_MODEL)

METADATA.write_text(
    json.dumps(spec, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

README.write_text(readme, encoding="utf-8")

NOTEBOOK.write_text(
    json.dumps(notebook, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("\n=== Final dan Konfigurasi ===")
for p in OUTPUTS:
    print("TERBUAT:", p)

print("\nDataset, hasil Orang A, dan model lama tidak diubah.")
