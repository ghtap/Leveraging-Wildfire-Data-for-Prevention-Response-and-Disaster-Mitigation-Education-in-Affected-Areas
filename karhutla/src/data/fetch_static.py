"""Fitur statis per sel: gambut, elevasi/lereng, tutupan lahan, jarak ke permukiman.

- Gambut: poligon lokal (config static.peat_file), fraksi luas per sel.
- Elevasi, lereng, tutupan lahan (ESA WorldCover 2021): ekspor GEE per sel.
- Jarak permukiman: raster fraksi built-up 0.01 deg dari GEE, jarak dihitung di Python.
"""
import argparse

import numpy as np
import pandas as pd

from .common import load_config, rpath

LC_CLASSES = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built",
              60: "bare", 80: "water", 90: "wetland", 95: "mangrove"}


# ---------------------------------------------------------------- GEE
def export_gee_static(cfg: dict):
    """Mulai 2 task: static_gee (CSV per sel) dan built_frac_001deg (GeoTIFF)."""
    import ee
    c, d = cfg["climate"], cfg["grid"]["cell_deg"]
    ee.Initialize(project=c["gee_project"])
    grid = pd.read_csv(rpath(cfg["paths"]["interim"]) / "grid.csv")
    h = d / 2
    cells = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Rectangle([x - h, y - h, x + h, y + h], "EPSG:4326", False), {"cell_id": int(i)})
        for i, x, y in zip(grid.cell_id, grid.lon, grid.lat)])

    dem = ee.Image("USGS/SRTMGL1_003")
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    stack = ee.Image.cat([dem.rename("elev_m"), ee.Terrain.slope(dem).rename("slope_deg")] +
                         [wc.eq(k).rename(f"lc_{v}") for k, v in LC_CLASSES.items()])
    names = ["elev_m", "slope_deg"] + [f"lc_{v}" for v in LC_CLASSES.values()]
    fc = stack.reduceRegions(collection=cells, reducer=ee.Reducer.mean(), scale=100)
    ee.batch.Export.table.toDrive(collection=fc, description="static_gee", folder=c["gee_folder"],
                                  fileFormat="CSV", selectors=["cell_id"] + names).start()

    # raster fraksi built-up 0.01 deg untuk jarak ke permukiman
    x0, x1 = np.floor((grid.lon.min() - 1) * 100) / 100, np.ceil((grid.lon.max() + 1) * 100) / 100
    y0, y1 = np.floor((grid.lat.min() - 1) * 100) / 100, np.ceil((grid.lat.max() + 1) * 100) / 100
    tf = [0.01, 0, float(x0), 0, -0.01, float(y1)]
    built = (wc.eq(50).unmask(0).reduceResolution(ee.Reducer.mean(), maxPixels=16384)
             .reproject(crs="EPSG:4326", crsTransform=tf))
    region = ee.Geometry.Rectangle([float(x0), float(y0), float(x1), float(y1)], "EPSG:4326", False)
    ee.batch.Export.image.toDrive(image=built.rename("built_frac").toFloat(), description="built_frac_001deg",
                                  folder=c["gee_folder"], region=region, crs="EPSG:4326",
                                  crsTransform=tf, maxPixels=1e9).start()
    print("task dimulai: static_gee, built_frac_001deg")


DW_BANDS = ["water", "trees", "grass", "flooded_vegetation", "crops", "shrub_and_scrub", "built", "bare"]


def export_gee_dw(cfg: dict):
    """Rata-rata probabilitas Dynamic World per tahun kalender per sel (satu task)."""
    import ee
    from .fetch_climate import _cells
    c = cfg["climate"]
    ee.Initialize(project=c["gee_project"])
    cells = _cells(cfg, ee)
    y0, y1 = pd.Timestamp(cfg["dates"]["start"]).year - 1, pd.Timestamp(cfg["dates"]["end"]).year - 1

    def per_year(y):
        y = ee.Number(y)
        img = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
               .filterDate(ee.Date.fromYMD(y, 1, 1), ee.Date.fromYMD(y.add(1), 1, 1)).select(DW_BANDS).mean())
        return img.reduceRegions(collection=cells, reducer=ee.Reducer.mean(), scale=100).map(lambda f: f.set("year", y))

    fc = ee.FeatureCollection(ee.List.sequence(y0, y1).map(per_year)).flatten()
    ee.batch.Export.table.toDrive(collection=fc, description="dw_yearly", folder=c["gee_folder"],
                                  fileFormat="CSV", selectors=["cell_id", "year"] + DW_BANDS).start()
    print("task dimulai: dw_yearly", y0, "-", y1)


def load_dw(cfg: dict):
    """dw_yearly.csv -> (cell_id, year, dw_<kelas>). None bila belum diunduh."""
    f = rpath(cfg["static"]["gee_dir"]) / "dw_yearly.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    df["cell_id"] = df["cell_id"].round().astype("int64")
    df["year"] = df["year"].round().astype("int64")
    df = df.rename(columns={b: f"dw_{b}" for b in DW_BANDS})
    return df.drop_duplicates(["cell_id", "year"]).astype({f"dw_{b}": "float32" for b in DW_BANDS})


# ---------------------------------------------------------------- Python
def peat_fraction(grid: pd.DataFrame, peat_path, d: float) -> pd.Series:
    """Fraksi luas sel yang berupa gambut (hitung di CRS luas-sama EPSG:6933)."""
    import geopandas as gpd
    from .build_grid import grid_polygons
    cells = grid_polygons(grid[["cell_id", "lon", "lat", "province"]], d).to_crs(6933)
    peat = gpd.read_file(peat_path).to_crs(6933)[["geometry"]]
    peat["geometry"] = peat.geometry.make_valid()
    inter = gpd.overlay(cells[["cell_id", "geometry"]], peat.dissolve(), how="intersection", keep_geom_type=True)
    area = inter.geometry.area.groupby(inter["cell_id"]).sum()
    frac = (area / cells.set_index("cell_id").geometry.area).reindex(grid["cell_id"]).fillna(0.0).clip(0, 1)
    return frac.set_axis(grid["cell_id"].to_numpy()).rename("peat_frac")


RASTER_EXT, VECTOR_EXT = {".tif", ".tiff"}, {".gpkg", ".shp", ".geojson", ".json"}


def find_peat(cfg: dict):
    """File gambut: static.peat_file bila ada, kalau tidak file pertama bertipe didukung di foldernya."""
    from .common import rpath
    f = rpath(cfg["static"]["peat_file"])
    if f.exists():
        return f
    cands = sorted(p for p in f.parent.glob("*") if p.suffix.lower() in RASTER_EXT | VECTOR_EXT) if f.parent.exists() else []
    return cands[0] if cands else None


def peat_fraction_raster(grid: pd.DataFrame, tif_path, d: float, peat_values=(1, 2), dominant_values=(1,)) -> pd.DataFrame:
    """Raster Global Peatland Map 2.0 (1 = didominasi gambut, 2 = gambut dalam mosaik tanah).
    peat_frac = fraksi piksel bernilai 1 atau 2 dalam sel; peat_dom_frac = fraksi bernilai 1. Nodata dihitung bukan gambut."""
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    h = d / 2
    out_all, out_dom = [], []
    with rasterio.open(tif_path) as src:
        for x, y in zip(grid.lon, grid.lat):
            b = transform_bounds("EPSG:4326", src.crs, x - h, y - h, x + h, y + h)
            w = from_bounds(*b, transform=src.transform)
            a = src.read(1, window=w, boundless=True, fill_value=0)
            if a.size == 0:
                out_all.append(0.0); out_dom.append(0.0); continue
            out_all.append(float(np.isin(a, peat_values).mean()))
            out_dom.append(float(np.isin(a, dominant_values).mean()))
    return pd.DataFrame({"peat_frac": out_all, "peat_dom_frac": out_dom}, index=grid["cell_id"].to_numpy(), dtype="float32")


def peat_features(grid, path, d) -> pd.DataFrame:
    if path.suffix.lower() in RASTER_EXT:
        return peat_fraction_raster(grid, path, d)
    return peat_fraction(grid, path, d).to_frame()


def dist_to_built(grid: pd.DataFrame, tif_path, threshold: float) -> pd.Series:
    """Jarak (km) dari titik tengah sel ke piksel permukiman terdekat (fraksi built >= threshold)."""
    import rasterio
    from scipy.spatial import cKDTree
    with rasterio.open(tif_path) as src:
        a = src.read(1)
        rows, cols = np.nonzero(a >= threshold)
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
    if len(xs) == 0:
        raise ValueError("tidak ada piksel permukiman; turunkan static.built_frac_threshold")

    def to_km(lon, lat):  # proyeksi ekuirektangular lokal, cukup untuk jarak puluhan km
        lat0 = np.deg2rad(np.mean(grid.lat))
        return np.c_[np.asarray(lon) * 111.32 * np.cos(lat0), np.asarray(lat) * 110.57]

    dist, _ = cKDTree(to_km(xs, ys)).query(to_km(grid.lon, grid.lat))
    return pd.Series(dist.astype("float32"), index=grid["cell_id"].to_numpy(), name="dist_built_km")


def build_static(cfg: dict) -> pd.DataFrame:
    s, d = cfg["static"], cfg["grid"]["cell_deg"]
    grid = pd.read_csv(rpath(cfg["paths"]["interim"]) / "grid.csv")
    out = grid[["cell_id"]].copy()
    parts = []
    gee_csv = rpath(s["gee_dir"]) / "static_gee.csv"
    if gee_csv.exists():
        g = pd.read_csv(gee_csv)
        g["cell_id"] = g["cell_id"].round().astype("int64")
        parts.append(g.drop_duplicates("cell_id").set_index("cell_id"))
    else:
        print("PERINGATAN: static_gee.csv tidak ada -> tanpa elevasi/lereng/tutupan lahan")
    pf = find_peat(cfg)
    if pf is not None:
        print("gambut dari:", pf)
        parts.append(peat_features(grid, pf, d))
    else:
        print("PERINGATAN: file gambut tidak ada -> tanpa peat_frac")
    if rpath(s["built_tif"]).exists():
        parts.append(dist_to_built(grid, rpath(s["built_tif"]), s["built_frac_threshold"]).to_frame())
    else:
        print("PERINGATAN: raster built-up tidak ada -> tanpa dist_built_km")
    for p in parts:
        out = out.merge(p, left_on="cell_id", right_index=True, how="left")
    return out.astype({c: "float32" for c in out.columns if c != "cell_id"})


def main(cfg=None):
    cfg = cfg or load_config()
    st = build_static(cfg)
    st.to_parquet(rpath(cfg["paths"]["interim"]) / "static_features.parquet", index=False)
    dw = load_dw(cfg)
    if dw is not None:
        dw.to_parquet(rpath(cfg["paths"]["interim"]) / "dw_yearly.parquet", index=False)
        print(f"Dynamic World: {len(dw)} baris, tahun {dw.year.min()}-{dw.year.max()}")
    else:
        print("PERINGATAN: dw_yearly.csv tidak ada -> tanpa fitur Dynamic World")
    print(f"statis: {len(st)} sel, kolom {[c for c in st.columns if c != 'cell_id']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    (export_gee_static(cfg), export_gee_dw(cfg)) if a.export else main(cfg)
