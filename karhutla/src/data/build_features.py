"""Fitur lag + label 7 hari per sel-hari.

Aturan anti-kebocoran: fitur pada hari t hanya memakai data sampai hari t
(dikurangi obs_latency_days untuk iklim); label = ada hotspot pada t+1..t+H.
"""
import copy

import numpy as np
import pandas as pd

from .common import ID_BASE, cell_index

META_COLS = ["date", "cell_id", "province", "lon", "lat"]
LABEL_COLS = ["y_7d", "n_hot_next_7d"]


class Roll:
    """Jumlah bergulir O(1) per jendela lewat cumsum; jendela yang memuat NaN -> NaN."""

    def __init__(self, arr: np.ndarray):
        T, N = arr.shape
        self.cs = np.zeros((T + 1, N), np.float64)
        np.cumsum(np.nan_to_num(arr, nan=0.0), axis=0, out=self.cs[1:])
        self.nn = np.zeros((T + 1, N), np.int32)
        np.cumsum(np.isnan(arr), axis=0, out=self.nn[1:])

    def sum(self, t: np.ndarray, w: int) -> np.ndarray:
        lo = t + 1 - w
        lo_c = np.maximum(lo, 0)
        s = self.cs[t + 1] - self.cs[lo_c]
        bad = (self.nn[t + 1] - self.nn[lo_c]) > 0
        bad |= (lo < 0)[:, None]
        s[bad] = np.nan
        return s.astype(np.float32)

    def mean(self, t, w):
        return self.sum(t, w) / np.float32(w)


def run_length(mask: np.ndarray) -> np.ndarray:
    """Panjang run True berturut-turut yang berakhir di tiap baris."""
    out = np.zeros(mask.shape, np.int16)
    for i in range(mask.shape[0]):
        out[i] = np.where(mask[i], (out[i - 1] if i else 0) + 1, 0)
    return out


def days_since(event: np.ndarray, cap: int) -> np.ndarray:
    out = np.empty(event.shape, np.int16)
    prev = np.full(event.shape[1], cap, np.int32)
    for i in range(event.shape[0]):
        prev = np.where(event[i], 0, np.minimum(prev + 1, cap))
        out[i] = prev
    return out


def to_array(df, cell_ids, dates, col, fill=np.nan) -> np.ndarray:
    """Tabel panjang (cell_id, date, col) -> array (T, N)."""
    ti = dates.get_indexer(df["date"])
    ci = pd.Index(cell_ids).get_indexer(df["cell_id"])
    ok = (ti >= 0) & (ci >= 0)
    arr = np.full((len(dates), len(cell_ids)), fill, np.float32)
    arr[ti[ok], ci[ok]] = df[col].to_numpy(np.float32)[ok]
    return arr


def neighbor_table(grid: pd.DataFrame, d: float) -> np.ndarray:
    """(N, 8) indeks tetangga; tetangga di luar grid -> N (kolom nol)."""
    ix, iy = cell_index(grid["lon"], grid["lat"], d)
    ids = pd.Index(grid["cell_id"].to_numpy())
    cols = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx or dy:
                pos = ids.get_indexer((iy + dy) * ID_BASE + (ix + dx))
                cols.append(np.where(pos < 0, len(ids), pos))
    return np.stack(cols, axis=1)


class FeatureBuilder:
    def __init__(self, cfg, grid, hotspots, climate, enso, last_firms_date, vi=None, dw=None):
        F, self.cfg = cfg["features"], cfg
        self.F, self.H = copy.deepcopy(F), cfg["label"]["horizon_days"]
        self.latency = F["obs_latency_days"]
        self.min_hot = cfg["label"]["min_hotspots"]
        self.dates = pd.date_range(cfg["dates"]["start"], cfg["dates"]["end"])
        grid = grid.sort_values("cell_id").reset_index(drop=True)

        clim = {c: to_array(climate, grid["cell_id"], self.dates, c)
                for c in ["precip_mm", "t2m_c", "rh", "sm1", "wind_ms"]}
        keep = (np.isfinite(clim["precip_mm"]).mean(0) >= 0.5) & (np.isfinite(clim["t2m_c"]).mean(0) >= 0.5)
        self.dropped_cells = int((~keep).sum())
        self.grid = grid[keep].reset_index(drop=True)
        self.N = len(self.grid)
        clim = {k: v[:, keep] for k, v in clim.items()}

        ids = self.grid["cell_id"]
        n_hot = to_array(hotspots, ids, self.dates, "n_hotspot", fill=0.0)
        frp = to_array(hotspots, ids, self.dates, "frp_sum", fill=0.0)
        self.last_idx = int(self.dates.get_indexer([min(pd.Timestamp(last_firms_date), self.dates[-1])])[0])
        self.R = {"hot": Roll(n_hot), "frp": Roll(frp)}
        for k, name in [("precip_mm", "precip"), ("t2m_c", "t2m"), ("rh", "rh"), ("sm1", "sm1"), ("wind_ms", "wind")]:
            self.R[name] = Roll(clim[k])
        self.sm1_raw, self.precip_raw = clim["sm1"], clim["precip_mm"]
        self.dry = run_length(np.nan_to_num(clim["precip_mm"], nan=np.inf) < F["dry_day_mm"])
        self.since = days_since(n_hot > 0, F["max_days_since_hotspot"])
        self.nbr = neighbor_table(self.grid, cfg["grid"]["cell_deg"])

        self.vi = self._vi_arrays(vi, ids) if vi is not None else None
        self.dw = None
        if dw is not None:
            cols = [c for c in dw.columns if c.startswith("dw_")]
            piv = {c: dw.pivot(index="year", columns="cell_id", values=c).reindex(columns=ids) for c in cols}
            self.dw_y0 = int(dw["year"].min())
            self.dw = {c: v.reindex(range(self.dw_y0, int(dw["year"].max()) + 1)).to_numpy(np.float32) for c, v in piv.items()}

        lag = F["enso_lag_months"]
        m = (self.dates.to_period("M") - lag).to_timestamp()
        e = enso.set_index("month").reindex(m)
        self.oni, self.dmi = e["oni"].to_numpy(np.float32), e["dmi"].to_numpy(np.float32)

        wins = (F["precip_windows"] + F["temp_windows"] + F["rh_windows"] + F["soil_windows"]
                + F["wind_windows"] + F["hotspot_windows"])
        self.t_first = max(wins) + self.latency
        self.t_last = self.last_idx - self.H

    def _vi_arrays(self, vi, ids):
        """Komposit -> deret harian: nilai baru muncul pada tanggal awal + vi_availability_days,
        lalu di-forward-fill (maks vi_max_ffill_days). Tidak ada nilai yang 'dari masa depan'."""
        F = self.F
        shifted = vi.assign(date=vi["date"] + pd.Timedelta(days=F["vi_availability_days"]))
        out = {}
        for col, name in [("NDVI", "ndvi_last"), ("EVI", "evi_last")]:
            a = to_array(shifted, ids, self.dates, col)
            out[name] = pd.DataFrame(a).ffill(limit=F["vi_max_ffill_days"]).to_numpy(np.float32)
        return out

    # ---------------------------------------------------------------
    def block(self, t: np.ndarray) -> pd.DataFrame:
        t = np.asarray(t)
        n, N, F = len(t), self.N, self.F
        tc = t - self.latency  # hari observasi iklim terakhir yang tersedia
        f = {}
        for w in F["precip_windows"]:
            f[f"precip_sum_{w}d"] = self.R["precip"].sum(tc, w)
        f["dry_days"] = np.where(np.isnan(self.precip_raw[tc]), np.nan, self.dry[tc]).astype(np.float32)
        for w in F["temp_windows"]:
            f[f"t2m_mean_{w}d"] = self.R["t2m"].mean(tc, w)
        for w in F["rh_windows"]:
            f[f"rh_mean_{w}d"] = self.R["rh"].mean(tc, w)
        f["sm1_now"] = self.sm1_raw[tc]
        for w in F["soil_windows"]:
            f[f"sm1_mean_{w}d"] = self.R["sm1"].mean(tc, w)
        for w in F["wind_windows"]:
            f[f"wind_mean_{w}d"] = self.R["wind"].mean(tc, w)
        for w in F["hotspot_windows"]:
            f[f"hot_{w}d"] = self.R["hot"].sum(t, w)
        f["frp_sum_7d"] = self.R["frp"].sum(t, 7)
        f["days_since_hot"] = self.since[t].astype(np.float32)
        h7 = np.concatenate([self.R["hot"].sum(t, 7), np.zeros((n, 1), np.float32)], axis=1)
        f["hot_nbr_7d"] = h7[:, self.nbr].sum(axis=2)

        if self.vi is not None:
            for name, arr in self.vi.items():
                f[name] = arr[t]
            tp = np.maximum(t - F["vi_change_days"], 0)
            f["ndvi_chg"] = self.vi["ndvi_last"][t] - self.vi["ndvi_last"][tp]
        if self.dw is not None:  # komposit tahun sebelumnya (tahun berjalan memuat data setelah t)
            idx = self.dates[t].year.to_numpy() - 1 - self.dw_y0
            ok = ((idx >= 0) & (idx < next(iter(self.dw.values())).shape[0]))[:, None]
            for name, arr in self.dw.items():
                f[name] = np.where(ok, arr[np.clip(idx, 0, arr.shape[0] - 1)], np.nan).astype(np.float32)

        f["oni_lag"] = np.broadcast_to(self.oni[t][:, None], (n, N))
        f["dmi_lag"] = np.broadcast_to(self.dmi[t][:, None], (n, N))
        doy = self.dates[t].dayofyear.to_numpy()
        ang = 2 * np.pi * doy / 365.25
        f["doy_sin"] = np.broadcast_to(np.sin(ang).astype(np.float32)[:, None], (n, N))
        f["doy_cos"] = np.broadcast_to(np.cos(ang).astype(np.float32)[:, None], (n, N))
        f["month"] = np.broadcast_to(self.dates[t].month.to_numpy(np.int8)[:, None], (n, N))

        cs = self.R["hot"].cs
        fut = cs[t + self.H + 1] - cs[t + 1]
        out = {
            "date": np.repeat(self.dates[t].to_numpy(), N),
            "cell_id": np.tile(self.grid["cell_id"].to_numpy(), n),
            "province": np.tile(self.grid["province"].to_numpy(), n),
            "lon": np.tile(self.grid["lon"].to_numpy(np.float32), n),
            "lat": np.tile(self.grid["lat"].to_numpy(np.float32), n),
        }
        out.update({k: np.ascontiguousarray(v).ravel() for k, v in f.items()})
        out["y_7d"] = (fut >= self.min_hot).astype(np.int8).ravel()
        out["n_hot_next_7d"] = fut.astype(np.int16).ravel()
        return pd.DataFrame(out)

    def iter_blocks(self):
        """Blok bulanan agar peak RAM lebih rendah; rolling tetap memakai riwayat penuh."""
        if self.t_first > self.t_last:
            return
        ts = np.arange(self.t_first, self.t_last + 1)
        months = self.dates[ts].to_period("M")
        for month in months.unique():
            yield self.block(ts[np.asarray(months == month)])
