"""Bangun grid sel (cell_id, lon, lat, province) untuk provinsi target."""
import argparse

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from .common import (ID_BASE, cell_index, id_to_center, load_config, norm_name, rpath)


def build_grid(cfg: dict, boundary: gpd.GeoDataFrame | None = None) -> pd.DataFrame:
    d = cfg["grid"]["cell_deg"]
    if boundary is None:
        boundary = gpd.read_file(rpath(cfg["grid"]["boundary_file"]))
    g = boundary.to_crs(4326)
    want = {norm_name(p): p for p in cfg["provinces"]}
    g = g.assign(province=g[cfg["grid"]["province_col"]].map(lambda s: want.get(norm_name(s))))
    g = g[g["province"].notna()]
    missing = set(cfg["provinces"]) - set(g["province"])
    if missing:
        raise ValueError(f"Provinsi tidak ditemukan di boundary: {sorted(missing)}")

    minx, miny, maxx, maxy = g.total_bounds
    ix0, iy0 = cell_index(minx, miny, d)
    ix1, iy1 = cell_index(maxx, maxy, d)
    IX, IY = np.meshgrid(np.arange(ix0, ix1 + 1), np.arange(iy0, iy1 + 1))
    cid = (IY * ID_BASE + IX).ravel()
    lon, lat = id_to_center(cid, d)
    pts = gpd.GeoDataFrame({"cell_id": cid, "lon": lon, "lat": lat},
                           geometry=gpd.points_from_xy(lon, lat), crs=4326)
    j = gpd.sjoin(pts, g[["province", "geometry"]], predicate="within", how="inner")
    out = (j.drop_duplicates("cell_id")[["cell_id", "lon", "lat", "province"]]
           .sort_values("cell_id").reset_index(drop=True))
    out["lon"] = out["lon"].round(6)
    out["lat"] = out["lat"].round(6)
    return out


def grid_polygons(grid: pd.DataFrame, d: float) -> gpd.GeoDataFrame:
    h = d / 2
    geom = [box(x - h, y - h, x + h, y + h) for x, y in zip(grid.lon, grid.lat)]
    return gpd.GeoDataFrame(grid.copy(), geometry=geom, crs=4326)


def main(cfg=None):
    cfg = cfg or load_config()
    out_dir = rpath(cfg["paths"]["interim"])
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = build_grid(cfg)
    grid.to_csv(out_dir / "grid.csv", index=False)
    grid_polygons(grid, cfg["grid"]["cell_deg"]).to_file(out_dir / "grid.geojson", driver="GeoJSON")
    print(f"grid: {len(grid)} sel\n{grid.province.value_counts().to_string()}")
    return grid


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
