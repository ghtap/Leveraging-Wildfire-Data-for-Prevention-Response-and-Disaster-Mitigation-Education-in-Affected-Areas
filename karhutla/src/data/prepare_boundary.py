import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SUMBER = ROOT / "data/raw/geoBoundaries-IDN-ADM1-all.zip"
OUTPUT = ROOT / "data/interim/prov_boundary.geojson"

PROVINSI = {
    "Central Kalimantan": "Kalimantan Tengah",
    "South Kalimantan": "Kalimantan Selatan",
    "West Kalimantan": "Kalimantan Barat",
    "South Sumatra": "Sumatera Selatan",
    "Jambi": "Jambi",
    "Riau": "Riau",
}

# 1. Membaca arsip asli tanpa mengubahnya
with zipfile.ZipFile(SUMBER) as arsip:
    data = json.loads(
        arsip.read("geoBoundaries-IDN-ADM1.geojson")
    )

# 2. Memilih hanya enam provinsi penelitian
hasil = []

for fitur in data["features"]:
    nama = fitur["properties"]["shapeName"]

    if nama in PROVINSI:
        fitur["properties"]["ADM1_NAME"] = PROVINSI[nama]
        hasil.append(fitur)

# 3. Memvalidasi jumlah provinsi
if len(hasil) != 6:
    raise ValueError(
        f"Seharusnya 6 provinsi, ditemukan {len(hasil)}"
    )

# 4. Menyimpan data baru ke folder interim
data["features"] = hasil

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

OUTPUT.write_text(
    json.dumps(data, ensure_ascii=False),
    encoding="utf-8"
)

print("Berhasil menyiapkan 6 provinsi!")
print(f"File tersimpan di: {OUTPUT}")