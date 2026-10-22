"""Rakit dataset_v1.parquet (sel-hari) + skema kolom + laporan QC untuk Orang B.

Pemakaian:  python -m src.data.make_dataset [--steps grid,firms,climate,static,features]
Default hanya `features` (asumsi langkah sebelumnya sudah jalan).
"""
import argparse
import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import build_grid, fetch_climate, fetch_firms, fetch_static
from .build_features import LABEL_COLS, META_COLS, FeatureBuilder
from .common import load_config, rpath


def load_inputs(cfg: dict) -> dict:
    d = rpath(cfg["paths"]["interim"])
    st = d / "static_features.parquet"
    return dict(
        grid=pd.read_csv(d / "grid.csv"),
        hotspots=pd.read_parquet(d / "hotspots_daily.parquet"),
        climate=pd.read_parquet(d / "climate_daily.parquet"),
        enso=pd.read_csv(d / "enso_monthly.csv", parse_dates=["month"]),
        last_firms_date=json.loads((d / "firms_meta.json").read_text())["last_date"],
        static=pd.read_parquet(st) if st.exists() else None,
        vi=pd.read_parquet(d / "vi_composites.parquet") if (d / "vi_composites.parquet").exists() else None,
        dw=pd.read_parquet(d / "dw_yearly.parquet") if (d / "dw_yearly.parquet").exists() else None,
    )


def make_dataset(cfg: dict, inp: dict, out_path) -> dict:
    fb = FeatureBuilder(cfg, inp["grid"], inp["hotspots"], inp["climate"], inp["enso"], inp["last_firms_date"],
                        vi=inp.get("vi"), dw=inp.get("dw"))
    static = inp.get("static")
    writer, cols, n_rows, qc_year, pos_by_prov = None, None, 0, {}, {}
    # Rekap missingness lintas seluruh blok, bukan hanya blok terakhir.
    missing_count, missing_total = {}, {}
    try:
        for blk in fb.iter_blocks():
            if blk.empty:
                continue
            if static is not None:
                blk = blk.merge(static, on="cell_id", how="left", validate="many_to_one")
                blk = blk[[c for c in blk.columns if c not in LABEL_COLS] + LABEL_COLS]
            if writer is None:
                cols = list(blk.columns)
                writer = pq.ParquetWriter(out_path, pa.Schema.from_pandas(blk, preserve_index=False), compression="zstd")
            writer.write_table(pa.Table.from_pandas(blk[cols], schema=writer.schema, preserve_index=False))
            n_rows += len(blk)
            year_stats = blk.groupby(blk["date"].dt.year)["y_7d"].agg(["sum", "size"])
            for year, stat in year_stats.iterrows():
                acc = qc_year.setdefault(int(year), [0, 0])
                acc[0] += int(stat["sum"])
                acc[1] += int(stat["size"])
            for p, v in blk.groupby("province")["y_7d"].agg(["sum", "size"]).iterrows():
                acc = pos_by_prov.setdefault(p, [0, 0])
                acc[0] += int(v["sum"])
                acc[1] += int(v["size"])
            for col in (c for c in blk.columns if c not in META_COLS + LABEL_COLS):
                missing_count[col] = missing_count.get(col, 0) + int(blk[col].isna().sum())
                missing_total[col] = missing_total.get(col, 0) + len(blk)
    finally:
        if writer is not None:
            writer.close()
    if cols is None:
        raise ValueError("Tidak ada blok fitur yang dapat ditulis. Periksa data dan rentang tanggal.")
    feat = [c for c in cols if c not in META_COLS + LABEL_COLS]
    report = dict(
        rows=n_rows, cells=fb.N, cells_dropped_no_climate=fb.dropped_cells,
        date_min=str(fb.dates[fb.t_first].date()), date_max=str(fb.dates[fb.t_last].date()),
        pos_rate_by_year={str(y): {"rows": n, "pos_rate": round(pos / n, 4)}
                          for y, (pos, n) in sorted(qc_year.items())},
        pos_rate_by_province={p: round(a[0] / a[1], 4) for p, a in pos_by_prov.items()},
        missing_pct_all_data={c: round(100 * missing_count[c] / missing_total[c], 2)
                              for c in feat if missing_count[c]},
        config_snapshot=dict(horizon_days=fb.H, cell_deg=cfg["grid"]["cell_deg"], sensors=cfg["firms"]["sensors"],
                             obs_latency_days=cfg["features"]["obs_latency_days"]),
    )
    schema = dict(meta_cols=META_COLS, label_cols=LABEL_COLS, feature_cols=feat,
                  categorical_cols=["province"], group_col="cell_id", time_col="date")
    base = out_path.with_suffix("")
    base.with_name(base.name + "_schema.json").write_text(json.dumps(schema, indent=2))
    base.with_name(base.name + "_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main(steps=("features",), cfg=None):
    cfg = cfg or load_config()
    run = dict(grid=build_grid.main, firms=fetch_firms.main, climate=fetch_climate.main, static=fetch_static.main)
    for s in steps:
        if s in run:
            run[s](cfg)
    if "features" in steps:
        out = rpath(cfg["paths"]["processed"]); out.mkdir(parents=True, exist_ok=True)
        rep = make_dataset(cfg, load_inputs(cfg), out / "dataset_v1.parquet")
        print(json.dumps({k: v for k, v in rep.items() if k != "missing_pct_all_data"}, indent=2, ensure_ascii=False))
        print("kolom dengan NaN pada seluruh dataset (%):", rep["missing_pct_all_data"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="features")
    main(tuple(ap.parse_args().steps.split(",")))
