from __future__ import annotations

from pathlib import Path
import pandas as pd
import yaml

# ==========================
# Path File
# ==========================
DATA_DIR = Path(r"D:\Trust-Aware Virtual Sensing and Supervisory Control for Smart Buildings\Building_59_dataset\Building_59\Bldg59_clean data")
YAML_PATH = Path(r"D:\Trust-Aware Virtual Sensing and Supervisory Control for Smart Buildings\make_point_map\point_contract.yaml")
OUT_DIR = Path(r"D:\Trust-Aware Virtual Sensing and Supervisory Control for Smart Buildings\make_point_map")
# ==========================

# also build a “full” table using every mapped Brick point in YAML
BUILD_FULL_FROM_BRICK_POINTS = True

# Export a clean Excel mapping table for Layer 0 review/audit
EXPORT_LAYER0_CONTRACT_EXCEL = True


def _parse_time(series: pd.Series, tz: str) -> pd.DatetimeIndex:
    """Parse timestamps (tz handling is done in Layer 1+; Layer 0 keeps it simple)."""
    ts = pd.to_datetime(series, errors="coerce")
    ts = ts[ts.notna()]
    return pd.DatetimeIndex(ts)


def _load_one_series(csv_path: Path, time_col: str, value_col: str, tz: str) -> pd.Series:
    df = pd.read_csv(csv_path, usecols=[time_col, value_col])
    ts = _parse_time(df[time_col], tz)

    good_mask = pd.to_datetime(df[time_col], errors="coerce").notna().values
    vals = pd.to_numeric(df.loc[good_mask, value_col], errors="coerce")

    s = pd.Series(vals.values, index=ts, name=value_col)

    # If duplicate timestamps exist, average them
    s = s.groupby(s.index).mean()
    return s


def _auto_percent_to_frac(s: pd.Series) -> pd.Series:
    """If values look like 0–100, convert to 0–1."""
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return s
    q95 = float(x.quantile(0.95))
    if 1.5 < q95 <= 110.0:
        return s / 100.0
    return s

def _apply_convert(s: pd.Series, convert: str | None) -> pd.Series:
    """
    Apply a single conversion rule to a numeric Series.

    Supported (from your table):
      - F_to_C
      - percent_to_frac
      - '' / None  -> no-op

    Notes:
      - CFM->CFM, psi->psi, ppm->ppm are no-ops (convert empty)
      - If s contains non-numeric values, you should coerce before calling this.
    """
    rule = (convert or "").strip()

    if rule == "" or rule == "/":
        return s

    if rule == "F_to_C":
        # (°F - 32) * 5/9
        return (s - 32.0) * (5.0 / 9.0)

    if rule == "percent_to_frac":
        # 0–100 (%) -> 0–1 (frac)
        return s / 100.0

    # For safety
    raise ValueError(f"Unknown convert rule: '{rule}'")


def _export_layer0_contract_excel(
    canon_specs: dict,
    df0: pd.DataFrame,
    out_xlsx: Path,
    data_dir: Path,
    tz: str,
    dt: str,
    zone: str,
):
    """
    Export Layer 0 contract (from YAML) + basic data stats (from df0) to Excel.
    This is ONLY for human review; Layer0 CSV remains the actual pipeline output.
    """
    # ---- Contract table (from YAML specs) ----
    rows = []
    for canon_name, spec in canon_specs.items():
        file = spec.get("file", "")
        csv_path = data_dir / file if file else None

        row = {
            "signal": canon_name,
            "file": file,
            "time_col": spec.get("time_col", ""),
            "value_col": spec.get("value_col", ""),
            # Optional metadata keys (only exist in your enriched YAML)
            "description": spec.get("description", ""),
            "unit": spec.get("unit", ""),
            "sampling_frequency": spec.get("sampling_frequency", ""),
            "missing_rate_raw": spec.get("missing_rate_raw", ""),
            "available_period": spec.get("available_period", ""),
            # Layer0 semantics
            "role": spec.get("role", ""),
            "resample": spec.get("resample", ""),
            "units_in": spec.get("units_in", ""),
            "units_out": spec.get("units_out", ""),
            "convert": spec.get("convert", ""),
            "sign": spec.get("sign", ""),
            "sign_reason": spec.get("sign_reason", ""),
            # File existence check
            "csv_exists": bool(csv_path.exists()) if csv_path else False,
        }
        rows.append(row)

    contract_df = pd.DataFrame(rows).sort_values(["file", "signal"], kind="stable")

    # ---- Data stats table (from df0) ----
    miss = df0.isna().mean().rename("missing_frac")
    desc = df0.describe(percentiles=[0.05, 0.5, 0.95]).T
    desc = desc.rename(
        columns={"min": "min", "5%": "p05", "50%": "p50", "95%": "p95", "max": "max", "count": "count"}
    )

    # First/last valid timestamp per column
    first_valid = {}
    last_valid = {}
    for c in df0.columns:
        s = df0[c]
        idx = s.first_valid_index()
        first_valid[c] = idx
        idx2 = s.last_valid_index()
        last_valid[c] = idx2

    stats_df = pd.DataFrame(
        {
            "signal": df0.columns,
            "missing_frac": miss.values,
            "count": desc["count"].values if "count" in desc.columns else None,
            "min": desc["min"].values,
            "p05": desc["p05"].values,
            "p50": desc["p50"].values,
            "p95": desc["p95"].values,
            "max": desc["max"].values,
            "first_valid_time": [first_valid[c] for c in df0.columns],
            "last_valid_time": [last_valid[c] for c in df0.columns],
        }
    )

    # Merge stats into contract 
    merged_df = contract_df.merge(stats_df, how="left", left_on="signal", right_on="signal")

    # Add global context columns
    merged_df.insert(0, "zone", zone)
    merged_df.insert(1, "timezone", tz)
    merged_df.insert(2, "sampling", dt)

    # ---- Write Excel ----
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        merged_df.to_excel(xw, sheet_name="contract", index=False)

        # also include a separate sheet sorted by missing fraction (useful)
        stats_sorted = stats_df.sort_values("missing_frac", ascending=False)
        stats_sorted.to_excel(xw, sheet_name="data_stats", index=False)

    print("Contract Excel :", out_xlsx)


def build_layer0():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(open(YAML_PATH, "r", encoding="utf-8"))

    tz = cfg.get("timezone", "America/Los_Angeles")
    dt = cfg.get("sampling", "1min")
    zone = cfg.get("zone", "unknown_zone")

    signals_cfg = cfg.get("signals", {})
    brick_points_cfg = cfg.get("brick_points", {})

    if BUILD_FULL_FROM_BRICK_POINTS:
        canon_specs = {
            k: v for k, v in brick_points_cfg.items()
            if isinstance(v, dict) and "file" in v and "value_col" in v
        }
    else:
        canon_specs = {}
        for canon_name, spec in signals_cfg.items():
            if isinstance(spec, dict) and "file" in spec and "value_col" in spec:
                canon_specs[canon_name] = spec
            elif isinstance(spec, str) and spec in brick_points_cfg:
                canon_specs[canon_name] = brick_points_cfg[spec]

    if not canon_specs:
        raise ValueError("No usable specs found in YAML (signals/brick_points).")

    # Load each signal
    series_list = []
    used = []

    for canon_name, spec in canon_specs.items():
        file = spec["file"]
        time_col = spec.get("time_col", "timestamp")
        value_col = spec.get("value_col")

        csv_path = DATA_DIR / file
        if not csv_path.exists():
            print(f"[WARN] Missing file for {canon_name}: {csv_path}")
            continue

        # load raw
        s = _load_one_series(csv_path, time_col, value_col, tz)
        s.name = canon_name

        # apply sign convention (default +1)
        sign = spec.get("sign", 1)
        try:
            s = float(sign) * s
        except Exception:
            pass

        # apply unit conversion (canonical)
        s = _apply_convert(s, spec.get("convert", ""))

        series_list.append(s)
        used.append(canon_name)

    if not series_list:
        raise RuntimeError("No signals were loaded. Check DATA_DIR, YAML file names, and columns.")

    df0 = pd.concat(series_list, axis=1).sort_index()

    # Resample to fixed cadence (NOTE: for true correctness, resample should be per-signal using spec['resample'])
    df0 = df0.resample(dt).mean()

    # Optional derived: fan_on
    if "fan_on" not in df0.columns:
        for cand in ["sf_speed", "fan_speed_hz", "fan_speed", "sf_vfd_speed"]:
            if cand in df0.columns:
                df0["fan_on"] = (df0[cand] > 5).astype(int)
                break

    out_csv = OUT_DIR / f"bldg59_layer0_{zone}_{dt}.csv"
    df0.to_csv(out_csv)

    print("\n===== LAYER 0 DONE =====")
    print("Output csv    :", out_csv)
    print("Timezone      :", tz)
    print("Sampling      :", dt)
    print("Columns used  :", used)

    # Export Excel contract
    if EXPORT_LAYER0_CONTRACT_EXCEL:
        out_xlsx = OUT_DIR / f"layer0_contract_{zone}_{dt}.xlsx"
        _export_layer0_contract_excel(
            canon_specs=canon_specs,
            df0=df0,
            out_xlsx=out_xlsx,
            data_dir=DATA_DIR,
            tz=tz,
            dt=dt,
            zone=zone,
        )

    print("\nMissing fraction (top 10):")
    print(df0.isna().mean().sort_values(ascending=False).head(10))

    print("\nQuick ranges:")
    desc = df0.describe(percentiles=[0.05, 0.5, 0.95]).T
    print(desc[["min", "5%", "50%", "95%", "max"]].head(12))


if __name__ == "__main__":
    build_layer0()

