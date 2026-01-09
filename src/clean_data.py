"""
Layer 1 builder:
- Input: one CSV with a datetime column + BAS points
- Output:
  (1) layer1_values.csv              : y_use(t) on fixed grid (NaN where unusable)
  (2) layer1_quality_exceptions.csv  : sparse interval log of exceptions (missing/rejected/interp)
  (3) qa_summary.csv + qa_report.png : QA report

Usage:
  python layer1_build.py --input your.csv --outdir out --date_col date --freq 1min

Notes:
- Thresholds below are sane *starting* defaults. You should tune them once you see QA plots.
- R is a nominal measurement variance placeholder. If you use EKF later, you can refine R.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# CONFIG (EDIT THESE)
# -------------------------
INPUT_CSV = r"D:\Trust-Aware Virtual Sensing and Supervisory Control for Smart Buildings\config\raw_data\bldg59_layer0_zone_062_1min.csv"
OUT_DIR   = r"D:\Trust-Aware Virtual Sensing and Supervisory Control for Smart Buildings\config"

DATE_COL = "date"                       # your datetime column name
FREQ = "1min"                           # "1min" or "5min" etc.

# If True: keep last value for duplicate timestamps
KEEP_LAST_DUPLICATE_TIMESTAMP = True

# Columns expected (your list)
EXPECTED_COLS = [
    "occ_forth_south",
    "rtu_003_econ_stpt_tn",
    "rtu_003_fltrd_gnd_plenum_press_tn",
    "rtu_003_fltrd_lvl2_plenum_press_tn",
    "rtu_003_ma_temp",
    "rtu_003_oa_damper",
    "rtu_003_oa_fr",
    "rtu_003_pa_static_stpt_tn",
    "rtu_003_ra_temp",
    "rtu_003_rf_vfd_spd_fbk_tn",
    "rtu_003_sa_temp",
    "rtu_003_sf_vfd_spd_fbk_tn",
    "wifi_fourth_south",
    "zone_062_co2",
    "zone_062_cooling_sp",
    "zone_062_fan_spd",
    "zone_062_heating_sp",
    "zone_062_temp",
]

# -------------------------
# Point contracts (EDIT/REFINE THESE THRESHOLDS)
# -------------------------

@dataclass
class PointSpec:
    bounds: Tuple[float, float]                 # (min, max)
    roc_limit: Optional[float] = None           # max |Δy| per sample (on the resampled grid)
    stuck_window: Optional[int] = None          # rolling window length in samples
    stuck_eps: Optional[float] = None           # rolling std < eps => stuck
    fill_method: str = "none"                   # "none" | "ffill" | "interp"
    fill_max_gap: int = 0                       # fill only if consecutive NaNs <= this
    R_nominal: float = 1.0                      # nominal variance (placeholder for Layer 2 weighting)


POINT_SPECS: Dict[str, PointSpec] = {
    # -------- unitless counts --------
    "occ_forth_south": PointSpec(
        bounds=(0, 500), roc_limit=150, fill_method="ffill", fill_max_gap=5, R_nominal=25.0
    ),
    "wifi_fourth_south": PointSpec(
        bounds=(0, 3000), roc_limit=500, fill_method="none", fill_max_gap=0, R_nominal=100.0
    ),

    # -------- temperatures in C --------
    "rtu_003_ma_temp": PointSpec(
        bounds=(-30, 60), roc_limit=6.0, stuck_window=60, stuck_eps=0.05,
        fill_method="interp", fill_max_gap=2, R_nominal=0.25
    ),
    "rtu_003_ra_temp": PointSpec(
        bounds=(-10, 50), roc_limit=4.0, stuck_window=60, stuck_eps=0.05,
        fill_method="interp", fill_max_gap=2, R_nominal=0.25
    ),
    "rtu_003_sa_temp": PointSpec(
        bounds=(-10, 60), roc_limit=6.0, stuck_window=60, stuck_eps=0.05,
        fill_method="interp", fill_max_gap=2, R_nominal=0.25
    ),

    "zone_062_temp": PointSpec(
        bounds=(10, 35), roc_limit=2.0, stuck_window=120, stuck_eps=0.03,
        fill_method="interp", fill_max_gap=2, R_nominal=0.10
    ),
    "zone_062_cooling_sp": PointSpec(
        bounds=(10, 35), roc_limit=2.0, fill_method="ffill", fill_max_gap=10, R_nominal=0.10
    ),
    "zone_062_heating_sp": PointSpec(
        bounds=(5, 30), roc_limit=2.0, fill_method="ffill", fill_max_gap=10, R_nominal=0.10
    ),

    "rtu_003_econ_stpt_tn": PointSpec(
        bounds=(-30, 60), roc_limit=5.0, fill_method="ffill", fill_max_gap=60, R_nominal=0.25
    ),

    # -------- fractions (0..1) --------
    "rtu_003_oa_damper": PointSpec(
        bounds=(0.0, 1.0), roc_limit=0.7, stuck_window=120, stuck_eps=0.01,
        fill_method="ffill", fill_max_gap=5, R_nominal=0.02
    ),
    "rtu_003_rf_vfd_spd_fbk_tn": PointSpec(
        bounds=(0.0, 1.0), roc_limit=0.7, stuck_window=120, stuck_eps=0.01,
        fill_method="ffill", fill_max_gap=2, R_nominal=0.01
    ),
    "rtu_003_sf_vfd_spd_fbk_tn": PointSpec(
        bounds=(0.0, 1.0), roc_limit=0.7, stuck_window=120, stuck_eps=0.01,
        fill_method="ffill", fill_max_gap=2, R_nominal=0.01
    ),
    "zone_062_fan_spd": PointSpec(
        bounds=(0.0, 1.0), roc_limit=0.9, fill_method="ffill", fill_max_gap=10, R_nominal=0.02
    ),

    # -------- airflow --------
    "rtu_003_oa_fr": PointSpec(
        bounds=(0, 200000), roc_limit=50000, stuck_window=120, stuck_eps=5.0,
        fill_method="none", fill_max_gap=0, R_nominal=1e6
    ),

    # -------- pressures in psi --------
    "rtu_003_fltrd_gnd_plenum_press_tn": PointSpec(
        bounds=(-1.0, 1.0), roc_limit=0.10, stuck_window=120, stuck_eps=0.0005,
        fill_method="interp", fill_max_gap=2, R_nominal=0.0004
    ),
    "rtu_003_fltrd_lvl2_plenum_press_tn": PointSpec(
        bounds=(-1.0, 1.0), roc_limit=0.10, stuck_window=120, stuck_eps=0.0005,
        fill_method="interp", fill_max_gap=2, R_nominal=0.0004
    ),
    "rtu_003_pa_static_stpt_tn": PointSpec(
        bounds=(-1.0, 1.0), roc_limit=0.20, fill_method="ffill", fill_max_gap=60, R_nominal=0.0004
    ),

    # -------- CO2 --------
    "zone_062_co2": PointSpec(
        bounds=(300, 5000), roc_limit=500, stuck_window=120, stuck_eps=2.0,
        fill_method="none", fill_max_gap=0, R_nominal=2500.0
    ),
}


# -------------------------
# Helpers
# -------------------------

def make_canonical_grid(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    df = df.sort_index()
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    return df.reindex(full_idx)

def fill_short_gaps(s: pd.Series, method: str, max_gap: int) -> Tuple[pd.Series, pd.Series]:
    if method == "none" or max_gap <= 0:
        return s, pd.Series(False, index=s.index)

    if method == "ffill":
        filled = s.ffill(limit=max_gap)
    elif method == "interp":
        filled = s.interpolate(method="time", limit=max_gap, limit_area="inside")
    else:
        raise ValueError(f"Unknown fill_method: {method}")

    was_filled = s.isna() & filled.notna()
    return filled, was_filled

def mask_to_intervals(mask: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp, int]]:
    mask = mask.fillna(False).astype(bool)
    if mask.empty or not mask.any():
        return []

    pos = np.where(mask.values)[0]
    breaks = np.where(np.diff(pos) != 1)[0]

    intervals = []
    start_i = 0
    for b in breaks:
        run = pos[start_i:b+1]
        intervals.append((mask.index[run[0]], mask.index[run[-1]], len(run)))
        start_i = b + 1
    run = pos[start_i:]
    intervals.append((mask.index[run[0]], mask.index[run[-1]], len(run)))
    return intervals


def clean_point(s_raw: pd.Series, spec: PointSpec):
    s = s_raw.copy()

    m_missing = s.isna()

    # bounds
    lo, hi = spec.bounds
    m_oob = (~m_missing) & ((s < lo) | (s > hi))
    s[m_oob] = np.nan

    # roc spikes
    if spec.roc_limit is not None:
        ds = s.diff().abs()
        m_spike = (ds > spec.roc_limit) & s.notna()
        s[m_spike] = np.nan
    else:
        m_spike = pd.Series(False, index=s.index)

    # stuck
    if spec.stuck_window and spec.stuck_eps is not None and spec.stuck_window > 2:
        roll_std = s.rolling(spec.stuck_window, min_periods=spec.stuck_window).std()
        m_stuck = (roll_std < spec.stuck_eps) & s.notna()
        s[m_stuck] = np.nan
    else:
        m_stuck = pd.Series(False, index=s.index)

    # fill short gaps
    s_use, m_filled = fill_short_gaps(s, spec.fill_method, spec.fill_max_gap)

    reason_masks = {
        "missing": m_missing,
        "out_of_bounds": m_oob,
        "roc_spike": m_spike,
        "stuck": m_stuck,
        "interpolated": m_filled,
    }
    return s_use, reason_masks


def build_exceptions(point: str, spec: PointSpec, reason_masks: Dict[str, pd.Series]) -> List[dict]:
    rows: List[dict] = []
    reject_reasons = ["missing", "out_of_bounds", "roc_spike", "stuck"]

    for reason in reject_reasons:
        for start, end, n in mask_to_intervals(reason_masks[reason]):
            rows.append({
                "point": point,
                "reason": reason,
                "start": start,
                "end": end,
                "n_samples": n,
                "mask": 0,
                "trust": 0.0,
                "R": np.nan,
            })

    for start, end, n in mask_to_intervals(reason_masks["interpolated"]):
        rows.append({
            "point": point,
            "reason": "interpolated",
            "start": start,
            "end": end,
            "n_samples": n,
            "mask": 1,
            "trust": 0.7,
            "R": spec.R_nominal * 2.0,
        })

    return rows


def main():
    outdir = Path(OUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    if DATE_COL not in df.columns:
        raise ValueError(f"DATE_COL='{DATE_COL}' not found. Columns: {list(df.columns)}")

    # datetime index
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).copy()
    df = df.set_index(DATE_COL).sort_index()

    # duplicates
    if KEEP_LAST_DUPLICATE_TIMESTAMP:
        df = df[~df.index.duplicated(keep="last")]
    else:
        df = df[~df.index.duplicated(keep="first")]

    # check expected cols
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        print(f"[WARN] Missing columns (will skip): {missing}")

    cols_present = [c for c in EXPECTED_COLS if c in df.columns]
    df = df[cols_present].copy()

    # numeric coercion
    for c in cols_present:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # canonical time grid
    df_grid = make_canonical_grid(df, FREQ)

    # clean
    y_use = pd.DataFrame(index=df_grid.index)
    exceptions_all: List[dict] = []
    qa_rows: List[dict] = []

    for c in cols_present:
        spec = POINT_SPECS.get(c)
        if spec is None:
            # if you forgot to define a spec, keep as-is
            spec = PointSpec(bounds=(-np.inf, np.inf), fill_method="none", fill_max_gap=0, R_nominal=1.0)

        s_use, reason_masks = clean_point(df_grid[c], spec)
        y_use[c] = s_use

        exceptions_all.extend(build_exceptions(c, spec, reason_masks))

        n_total = len(df_grid)
        n_missing = int(reason_masks["missing"].sum())
        n_oob = int(reason_masks["out_of_bounds"].sum())
        n_spike = int(reason_masks["roc_spike"].sum())
        n_stuck = int(reason_masks["stuck"].sum())
        n_interp = int(reason_masks["interpolated"].sum())
        n_reject = n_missing + n_oob + n_spike + n_stuck

        qa_rows.append({
            "point": c,
            "n_total": n_total,
            "n_missing": n_missing,
            "n_out_of_bounds": n_oob,
            "n_roc_spike": n_spike,
            "n_stuck": n_stuck,
            "n_interpolated": n_interp,
            "pct_missing": n_missing / n_total if n_total else np.nan,
            "pct_rejected": n_reject / n_total if n_total else np.nan,
            "pct_interpolated": n_interp / n_total if n_total else np.nan,
        })

    # write values (date + points)
    values_path = outdir / "layer1_values.csv"
    y_use_out = y_use.reset_index().rename(columns={"index": DATE_COL})
    y_use_out.to_csv(values_path, index=False)

    # write exceptions
    qual_path = outdir / "layer1_quality_exceptions.csv"
    qual_df = pd.DataFrame(exceptions_all)
    if not qual_df.empty:
        qual_df["start"] = pd.to_datetime(qual_df["start"])
        qual_df["end"] = pd.to_datetime(qual_df["end"])
        qual_df = qual_df.sort_values(["point", "start", "reason"])
    qual_df.to_csv(qual_path, index=False)

    # write QA summary
    qa_path = outdir / "qa_summary.csv"
    qa_df = pd.DataFrame(qa_rows).sort_values("pct_rejected", ascending=False)
    qa_df.to_csv(qa_path, index=False)

    # QA plot (bar chart)
    plot_path = outdir / "qa_report.png"
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(qa_df))
    ax.bar(x, qa_df["pct_missing"].values, label="missing")
    ax.bar(
        x,
        (qa_df["pct_rejected"].values - qa_df["pct_missing"].values),
        bottom=qa_df["pct_missing"].values,
        label="rejected (non-missing)"
    )
    ax.bar(x, qa_df["pct_interpolated"].values, alpha=0.6, label="interpolated")
    ax.set_xticks(x)
    ax.set_xticklabels(qa_df["point"].values, rotation=45, ha="right")
    ax.set_ylabel("Fraction of samples")
    ax.set_title("Layer 1 QA Summary (missing / rejected / interpolated)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    print("[DONE] Wrote:")
    print(f"  {values_path}")
    print(f"  {qual_path}")
    print(f"  {qa_path}")
    print(f"  {plot_path}")


if __name__ == "__main__":
    main()
