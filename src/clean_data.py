"""
Layer 1: Trust-aware data QA and preparation for estimation/control

Purpose
-------
Convert a standardized BAS time-series table (Layer 0 output) into an estimator-ready dataset by
(i) enforcing basic physical plausibility checks
(ii) optionally repairing very short gaps
(iii) producing an auditable quality log that downstream estimators/controllers can use to
mask or down-weight low-quality samples.

Inputs
------
A single table (CSV) containing:
- a datetime column (e.g., "date")
- one column per canonical BAS point (already unit-converted and named consistently)

Processing (per point)
----------------------
1) Align to a canonical time grid (e.g., 1 minute) via reindex/resample (Layer 0 may already do this).
2) Create QA masks on the grid:
   - missing: raw NaN on the grid
   - out_of_bounds: violates point-specific bounds (physical plausibility)
   - roc_spike: exceeds point-specific rate-of-change limit (spike detection)
   - unresponsive (optional): context-aware flag or reject (only if enabled in PointSpec)
3) Apply masking rules to produce a masked signal s_masked(t) (invalid samples set to NaN).
4) Optional short-gap repair (policy-driven):
   - ffill for sample-and-hold points (e.g., setpoints/modes)
   - interp for very short gaps on slow sensors (if explicitly allowed)
   - none for most measured sensors (recommended default)
   The repair step yields s_use(t) and a mask "interpolated" indicating where repaired values were inserted.

Outputs
-------
(1) layer1_values.csv
    Estimator-facing values on the canonical grid:
    - same columns as input
    - NaN where samples are unusable (masked)
    - repaired values only where the point’s fill policy permits it

(2) layer1_quality_exceptions.csv
    Sparse, interval-based log of QA events (audit + estimator weighting):
    - point, reason, start, end, n_samples
    - mask (0 = do not use as measurement, 1 = usable but possibly down-weighted)
    - trust (0..1) and nominal measurement variance R (for EKF/KF/MHE weighting)

(3) qa_summary.csv and qa_report.png
    Per-point summary statistics and a compact visualization to guide threshold tuning.

Notes
-----------------------
- The default stance is conservative: do not fabricate sensor measurements unless explicitly justified.
  Forward-fill is appropriate for sample-and-hold variables (e.g., setpoints), while interpolation on
  measured sensors should be limited to very short gaps and clearly logged (interpolated mask).
- All thresholds (bounds, ROC limits, gap limits, nominal R) are initial engineering priors. They should
  be refined using the QA report and by checking against known equipment behavior and BAS conventions.
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
    bounds: Tuple[float, float]
    roc_limit: Optional[float] = None

    # gap fill
    fill_method: str = "none"      # "none" | "ffill" | "interp"
    fill_max_gap: int = 0

    # Layer 2 hook (optional)
    R_nominal: float = 1.0

    # -------- context-aware "unresponsive" detection --------
    # Only evaluate flatline when system is active:
    active_point: Optional[str] = None      # e.g., "rtu_003_sf_vfd_spd_fbk_tn"
    active_threshold: Optional[float] = None # e.g., 0.10 (frac)

    # Only evaluate flatline when a driver is changing:
    driver_point: Optional[str] = None      # e.g., "rtu_003_sf_vfd_spd_fbk_tn"
    driver_window: Optional[int] = None     # samples
    driver_min_std: Optional[float] = None  # std threshold

    # Define what "flat" means:
    flat_window: Optional[int] = None       # samples
    flat_eps: Optional[float] = None        # range threshold (max-min)

    # What to do if unresponsive is detected:
    unresp_action: str = "flag"             # "flag" or "reject"



POINT_SPECS: Dict[str, PointSpec] = {
    # -------- unitless counts --------
    "occ_forth_south": PointSpec(
        bounds=(0, 500),
        roc_limit=150,
        fill_method="ffill",
        fill_max_gap=5,
        R_nominal=25.0,
        # no context checks on occupancy
    ),
    "wifi_fourth_south": PointSpec(
        bounds=(0, 3000),
        roc_limit=500,
        fill_method="none",
        fill_max_gap=0,
        R_nominal=100.0,
        # no context checks here either (it’s sparse)
    ),

    # -------- temperatures in C --------
    # Keep temps simple: bounds + roc; NO "flat=bad" logic
    "rtu_003_ma_temp": PointSpec(
        bounds=(-30, 60),
        roc_limit=6.0,
        fill_method="none",      # recommend none for sensors; if you insist, interp gap<=2 is okay
        fill_max_gap=0,
        R_nominal=0.25,
    ),
    "rtu_003_ra_temp": PointSpec(
        bounds=(-10, 50),
        roc_limit=4.0,
        fill_method="none",
        fill_max_gap=0,
        R_nominal=0.25,
    ),
    "rtu_003_sa_temp": PointSpec(
        bounds=(-10, 60),
        roc_limit=6.0,
        fill_method="none",
        fill_max_gap=0,
        R_nominal=0.25,
    ),
    "zone_062_temp": PointSpec(
        bounds=(10, 35),
        roc_limit=2.0,
        fill_method="none",
        fill_max_gap=0,
        R_nominal=0.10,
    ),

    # -------- setpoints in C (sample-and-hold) --------
    "zone_062_cooling_sp": PointSpec(
        bounds=(10, 35),
        roc_limit=2.0,
        fill_method="ffill",
        fill_max_gap=10**9,   # effectively hold forever
        R_nominal=0.10,
    ),
    "zone_062_heating_sp": PointSpec(
        bounds=(5, 30),
        roc_limit=2.0,
        fill_method="ffill",
        fill_max_gap=10**9,
        R_nominal=0.10,
    ),
    "rtu_003_econ_stpt_tn": PointSpec(
        bounds=(-30, 60),
        roc_limit=5.0,
        fill_method="ffill",
        fill_max_gap=10**9,
        R_nominal=0.25,
    ),

    # -------- fractions (0..1) --------
    # These can be steady legitimately; don’t reject flatness.
    "rtu_003_oa_damper": PointSpec(
        bounds=(0.0, 1.0),
        roc_limit=0.7,
        fill_method="ffill",
        fill_max_gap=5,
        R_nominal=0.02,
    ),
    "rtu_003_rf_vfd_spd_fbk_tn": PointSpec(
        bounds=(0.0, 1.0),
        roc_limit=0.7,
        fill_method="ffill",
        fill_max_gap=2,
        R_nominal=0.01,
    ),
    "rtu_003_sf_vfd_spd_fbk_tn": PointSpec(
        bounds=(0.0, 1.0),
        roc_limit=0.7,
        fill_method="ffill",
        fill_max_gap=2,
        R_nominal=0.01,
    ),
    "zone_062_fan_spd": PointSpec(
        bounds=(0.0, 1.0),
        roc_limit=0.9,
        fill_method="ffill",
        fill_max_gap=10,
        R_nominal=0.02,
    ),

    # -------- airflow (CFM) --------
    # Context-aware: OA flow should not be perfectly flat when fan is ON and damper is moving
    "rtu_003_oa_fr": PointSpec(
        bounds=(0, 200000),
        roc_limit=50000,
        fill_method="none",
        fill_max_gap=0,
        R_nominal=1e6,

        active_point="rtu_003_sf_vfd_spd_fbk_tn",
        active_threshold=0.15,          # fan on

        driver_point="rtu_003_oa_damper",
        driver_window=30,              # 30 min window
        driver_min_std=0.03,           # damper is actually modulating

        flat_window=30,
        flat_eps=200.0,                # if OA flow range <= 200 CFM while damper modulates => suspicious
        unresp_action="flag",          # start by flagging only
    ),

    # -------- pressures in psi --------
    # Context-aware: pressure should respond when fan speed is changing (modulating)
    "rtu_003_fltrd_gnd_plenum_press_tn": PointSpec(
        bounds=(-1.0, 1.0),
        roc_limit=0.10,
        fill_method="none",
        fill_max_gap=0,
        R_nominal=0.0004,

        active_point="rtu_003_sf_vfd_spd_fbk_tn",
        active_threshold=0.15,

        driver_point="rtu_003_sf_vfd_spd_fbk_tn",
        driver_window=30,
        driver_min_std=0.03,           # fan changing

        flat_window=30,
        flat_eps=0.002,                # psi range ~0 while fan modulates => suspicious
        unresp_action="flag",
    ),
    "rtu_003_fltrd_lvl2_plenum_press_tn": PointSpec(
        bounds=(-1.0, 1.0),
        roc_limit=0.10,
        fill_method="none",
        fill_max_gap=0,
        R_nominal=0.0004,

        active_point="rtu_003_sf_vfd_spd_fbk_tn",
        active_threshold=0.15,

        driver_point="rtu_003_sf_vfd_spd_fbk_tn",
        driver_window=30,
        driver_min_std=0.03,

        flat_window=30,
        flat_eps=0.002,
        unresp_action="flag",
    ),
    "rtu_003_pa_static_stpt_tn": PointSpec(
        bounds=(-1.0, 1.0),
        roc_limit=0.20,
        fill_method="ffill",
        fill_max_gap=10**9,  # setpoint should hold
        R_nominal=0.0004,
    ),

    # -------- CO2 (ppm) --------
    # Context-aware: CO2 being perfectly flat while occupancy changes can be suspicious.
    "zone_062_co2": PointSpec(
        bounds=(300, 5000),
        roc_limit=500,
        fill_method="none",
        fill_max_gap=0,
        R_nominal=2500.0,

        active_point="occ_forth_south",
        active_threshold=5,            # “occupied enough to matter”

        driver_point="occ_forth_south",
        driver_window=60,              # 1 hr
        driver_min_std=5,              # occupancy actually changing

        flat_window=120,               # 2 hr flatline
        flat_eps=20.0,                 # <= 20 ppm range over 2 hr while occ changes => suspicious
        unresp_action="flag",
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

def clean_point_with_context(
    df_grid: pd.DataFrame,          # <-- whole dataframe on canonical grid
    point: str,
    spec: PointSpec
):
    s_raw = df_grid[point].copy()

    # 1) Missing
    m_missing = s_raw.isna()

    # 2) Bounds
    lo, hi = spec.bounds
    m_oob = (~m_missing) & ((s_raw < lo) | (s_raw > hi))
    s_masked = s_raw.copy()
    s_masked[m_oob] = np.nan

    # 3) ROC spikes (optional)
    if spec.roc_limit is not None:
        ds = s_masked.diff().abs()
        m_spike = (ds > spec.roc_limit) & s_masked.notna()
        s_masked[m_spike] = np.nan
    else:
        m_spike = pd.Series(False, index=s_raw.index)

    # 4) Context-aware unresponsive (optional)
    m_unresp = pd.Series(False, index=s_raw.index)

    if (
        spec.active_point and spec.active_threshold is not None and
        spec.driver_point and spec.driver_window and spec.driver_min_std is not None and
        spec.flat_window and spec.flat_eps is not None
    ):
        a = pd.to_numeric(df_grid[spec.active_point], errors="coerce")
        d = pd.to_numeric(df_grid[spec.driver_point], errors="coerce")

        # Active condition (e.g., fan on)
        active = a > spec.active_threshold

        # Driver excitation condition (driver is actually changing)
        d_std = d.rolling(spec.driver_window, min_periods=spec.driver_window).std()
        excited = d_std > spec.driver_min_std

        # Flatness of the target point (use rolling range)
        s_for_flat = s_masked.copy()
        s_max = s_for_flat.rolling(spec.flat_window, min_periods=spec.flat_window).max()
        s_min = s_for_flat.rolling(spec.flat_window, min_periods=spec.flat_window).min()
        s_range = (s_max - s_min)

        flat = (s_range <= spec.flat_eps) & s_for_flat.notna()

        m_unresp = active & excited & flat

        if spec.unresp_action == "reject":
            s_masked[m_unresp] = np.nan
        elif spec.unresp_action == "flag":
            # keep values, only flag
            pass
        else:
            raise ValueError("unresp_action must be 'flag' or 'reject'")

    # 5) Fill short gaps (optional, AFTER masking)
    s_use, m_filled = fill_short_gaps(s_masked, spec.fill_method, spec.fill_max_gap)

    reason_masks = {
        "missing": m_missing,
        "out_of_bounds": m_oob,
        "roc_spike": m_spike,
        "unresponsive": m_unresp,
        "filled": m_filled,
    }
    return s_use, s_masked, reason_masks

def build_exceptions(point: str, spec: PointSpec, reason_masks: Dict[str, pd.Series]) -> List[dict]:
    rows: List[dict] = []

    def add_intervals(reason: str, mask_series: pd.Series, mask_val: int, trust: float, R):
        for start, end, n in mask_to_intervals(mask_series):
            rows.append({
                "point": point,
                "reason": reason,
                "start": start,
                "end": end,
                "n_samples": n,
                "mask": mask_val,
                "trust": trust,
                "R": R,
            })

    # ---- hard rejects ----
    for reason in ["missing", "out_of_bounds", "roc_spike"]:
        m = reason_masks.get(reason)
        if m is not None:
            add_intervals(reason, m, mask_val=0, trust=0.0, R=np.nan)

    # ---- unresponsive (flag or reject depending on spec) ----
    m_unresp = reason_masks.get("unresponsive")
    if m_unresp is not None:
        if getattr(spec, "unresp_action", "flag") == "reject":
            add_intervals("unresponsive", m_unresp, mask_val=0, trust=0.0, R=np.nan)
        else:
            add_intervals("unresponsive", m_unresp, mask_val=1, trust=0.3, R=spec.R_nominal * 5.0)

    # ---- filled/interpolated (always usable but down-weighted) ----
    m_fill = reason_masks.get("interpolated")
    if m_fill is None:
        m_fill = reason_masks.get("filled")  # fallback if you kept the name "filled"

    if m_fill is not None:
        add_intervals("interpolated", m_fill, mask_val=1, trust=0.7, R=spec.R_nominal * 2.0)

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

        s_use, s_masked, reason_masks = clean_point_with_context(df_grid, c, spec)
        y_use[c] = s_use

        exceptions_all.extend(build_exceptions(c, spec, reason_masks))

        n_total = len(df_grid.index)

        # ---- stage metrics (KEEP these) ----
        missing_raw = int(df_grid[c].isna().sum())  # raw NaNs on canonical grid
        nan_pre_fill = int(s_masked.isna().sum())  # NaNs after QA masking (before fill)
        missing_final = int(s_use.isna().sum())  # NaNs in final output
        n_filled = int((s_masked.isna() & s_use.notna()).sum())  # repaired by fill
        rejected_added = nan_pre_fill - missing_raw  # extra NaNs introduced by QA rules

        # ---- reason metrics (KEEP these, optional but useful) ----
        n_oob = int(reason_masks["out_of_bounds"].sum())
        n_spike = int(reason_masks["roc_spike"].sum())
        n_stuck = int(reason_masks.get("stuck", pd.Series(False, index=df_grid.index)).sum())
        n_unresp = int(reason_masks.get("unresponsive", pd.Series(False, index=df_grid.index)).sum())

        qa_rows.append({
            "point": c,
            "n_total": n_total,

            # stage (these are the ones you interpret)
            "missing_raw": missing_raw,
            "rejected_added": rejected_added,
            "filled": n_filled,
            "missing_final": missing_final,

            "pct_missing_raw": missing_raw / n_total,
            "pct_rejected_added": rejected_added / n_total,
            "pct_filled": n_filled / n_total,
            "pct_missing_final": missing_final / n_total,

            # reasons (debug only)
            "n_out_of_bounds": n_oob,
            "n_roc_spike": n_spike,
            "n_stuck": n_stuck,
            "n_unresponsive": n_unresp,
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
    qa_df = pd.DataFrame(qa_rows).sort_values("pct_missing_final", ascending=False)
    qa_df.to_csv(qa_path, index=False)

    # Plot
    plot_path = outdir / "qa_report.png"
    fig, ax = plt.subplots(figsize=(14, 6))
    qa_df = qa_df.sort_values("pct_missing_final", ascending=False).reset_index(drop=True)
    x = np.arange(len(qa_df))
    w = 0.22  # bar width
    ax.bar(x - 1.5 * w, qa_df["pct_missing_raw"].values, width=w, label="missing_raw")
    ax.bar(x - 0.5 * w, qa_df["pct_rejected_added"].values, width=w, label="rejected_added")
    ax.bar(x + 0.5 * w, qa_df["pct_filled"].values, width=w, label="filled")
    ax.set_xticks(x)
    ax.set_xticklabels(qa_df["point"].values, rotation=45, ha="right")
    ax.set_ylabel("Fraction of samples")
    ax.set_title("Layer 1 QA (grouped metrics)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    ax.set_xticks(x)
    ax.set_xticklabels(qa_df["point"].values, rotation=45, ha="right")
    ax.set_ylabel("Fraction of samples")
    ax.set_title("Layer 1 QA (overlaid bars)")
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
