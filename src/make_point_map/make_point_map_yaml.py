"""
Build Building 59 Signal Contract YAML

Reads:
- Brick model (.ttl) to discover zone/RTU/floor points
- Clean-data CSV headers to map Brick point names -> (file, time_col, value_col)
- Data description Excel to attach metadata (description, unit, sampling, missingness, availability)

Writes:
- point_contract.yaml containing:
    - signals: small canonical set for the CO2 app (with units/convert/role/resample/sign)
    - brick_points: full mapped point catalog with the same contract fields
"""

from pathlib import Path
import re
import difflib
import pandas as pd
import yaml
import numpy as np
from rdflib import Graph
from rdflib.namespace import RDF


DATA_DIR = Path(r"D:\Trust-Aware Virtual Sensing and Supervisory Control for Smart Buildings\Building_59_dataset\Building_59")
TTL_PATH  = DATA_DIR / "Bldg59_w_occ Brick model.ttl"

EXCEL_DESC = Path(r"D:\Trust-Aware Virtual Sensing and Supervisory Control for Smart Buildings\Building_59_dataset\data_description_table_3year_clean_data.xlsx")

ZONE_NAME  = "zone_062"
TIMEZONE   = "America/Los_Angeles"
SAMPLING   = "1min"

# YAML 
OUT_YAML   = Path(r"D:\Trust-Aware Virtual Sensing and Supervisory Control for Smart Buildings\make_point_map\point_contract.yaml")


# Prefer clean-data folder if it exists
CLEAN_DIR = DATA_DIR / "Bldg59_clean data"
CSV_ROOT = CLEAN_DIR if CLEAN_DIR.exists() else DATA_DIR

TIME_COL_CANDIDATES = ["timestamp", "time", "ts", "datetime", "date_time", "DateTime", "Date", "Time"]

INFER_SIGN_FOR_PRESSURE = True
SIGN_SAMPLE_ROWS = 200_000

# -------------------------
# Helpers
# -------------------------
def local_name(x) -> str:
    s = str(x)
    if "#" in s:
        return s.split("#")[-1]
    return s.rsplit("/", 1)[-1]

def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def pred_name(p) -> str:
    return local_name(p)

def find_uri_by_local_name(g: Graph, name: str):
    target = name.lower()
    for x in set(list(g.subjects()) + list(g.objects())):
        try:
            if local_name(x).lower() == target:
                return x
        except Exception:
            pass
    return None

def triples_with_pred_local(g: Graph, subj=None, pred_local=None, obj=None):
    for s, p, o in g.triples((subj, None, obj)):
        if pred_local is None or pred_name(p) == pred_local:
            yield s, p, o

def points_of_entity(g: Graph, entity_uri):
    pts = set()
    for s, p, o in triples_with_pred_local(g, subj=entity_uri, pred_local="hasPoint"):
        pts.add(o)
    for s, p, o in triples_with_pred_local(g, subj=None, pred_local="isPointOf", obj=entity_uri):
        pts.add(s)
    return pts

def type_locals(g: Graph, uri):
    out = []
    for t in g.objects(uri, RDF.type):
        out.append(local_name(t))
    return out

def get_zone_bundle(g: Graph, zone_name: str):
    zone_uri = find_uri_by_local_name(g, zone_name)
    if zone_uri is None:
        raise ValueError(f"Zone '{zone_name}' not found in Brick TTL")

    vavs = [s for s, p, o in triples_with_pred_local(g, subj=None, pred_local="feeds", obj=zone_uri)]

    rtus = []
    for v in vavs:
        rtus += [o for s, p, o in triples_with_pred_local(g, subj=v, pred_local="isFedBy", obj=None)]

    floors = [o for s, p, o in triples_with_pred_local(g, subj=zone_uri, pred_local="isPartOf", obj=None)]

    zone_pts = points_of_entity(g, zone_uri)
    rtu_pts = set()
    for r in rtus:
        rtu_pts |= points_of_entity(g, r)
    floor_pts = set()
    for f in floors:
        floor_pts |= points_of_entity(g, f)

    def typed_list(pts):
        return [{"name": local_name(p), "types": type_locals(g, p)} for p in pts]

    return {
        "zone": local_name(zone_uri),
        "vavs": [local_name(v) for v in vavs],
        "rtus": [local_name(r) for r in rtus],
        "floors": [local_name(f) for f in floors],
        "zone_points": typed_list(zone_pts),
        "rtu_points": typed_list(rtu_pts),
        "floor_points": typed_list(floor_pts),
    }

def scan_csv_headers(data_dir: Path):
    csvs = sorted(data_dir.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found under {data_dir}")

    file_cols = {}
    for p in csvs:
        try:
            df = pd.read_csv(p, nrows=0)
            file_cols[p.name] = [c.strip() for c in df.columns]
        except Exception as e:
            print(f"[WARN] skip {p.name}: {e}")
    return file_cols

def infer_time_col(cols):
    for c in cols:
        if c in TIME_COL_CANDIDATES or c.lower() in [x.lower() for x in TIME_COL_CANDIDATES]:
            return c
    return cols[0] if cols else "timestamp"

def match_point_to_column(point_name: str, file_cols: dict):
    best_file = None
    best_col = None
    best_score = -1.0
    pn = norm(point_name)

    for fname, cols in file_cols.items():
        for c in cols:
            cn = norm(c)
            if pn == cn:
                score = 100.0
            elif pn and pn in cn:
                score = 88.0 - 0.02 * (len(cn) - len(pn))
            else:
                score = 60.0 * difflib.SequenceMatcher(None, pn, cn).ratio()

            if score > best_score:
                best_score = score
                best_file = fname
                best_col = c

    if best_score < 45 or best_file is None or best_col is None:
        return None
    return {"file": best_file, "value_col": best_col}

def pick_canonical_signals(bundle, mapped_brick_points):
    def find_first(points, want_type):
        for p in points:
            if want_type in p["types"] and p["name"] in mapped_brick_points:
                return p["name"]
        return None

    signals = {}
    co2 = find_first(bundle["zone_points"], "CO2_Sensor")
    if co2: signals["co2_ppm"] = co2

    zt = find_first(bundle["zone_points"], "Zone_Air_Temperature_Sensor")
    if zt: signals["zone_temp"] = zt

    flow = find_first(bundle["zone_points"], "Supply_Air_Flow_Sensor")
    if flow: signals["supply_flow"] = flow

    oa = find_first(bundle["rtu_points"], "Outdoor_Air_Damper")
    if oa: signals["oa_damper"] = oa

    sf = find_first(bundle["rtu_points"], "Supply_Air_Fan_Speed")
    if sf: signals["sf_speed"] = sf

    occ = find_first(bundle["floor_points"], "Occupant_Count")
    if occ: signals["occ_floor"] = occ

    return signals

# -------------------------
# Excel lookup + enrichment
# -------------------------
def load_excel_lut(excel_path: Path):
    df = pd.read_excel(excel_path)
    df["File name"] = df["File name"].ffill()

    lut = {}
    for _, r in df.iterrows():
        f = str(r.get("File name", "")).strip()
        if not f or f.lower() == "nan":
            continue
        f = Path(f).name

        # take the first non-empty unit/desc per file (or override rules you want)
        unit = str(r.get("Unit", "")).strip()
        desc = str(r.get("Description", "")).strip()
        samp = str(r.get("Sampling frequency", "")).strip()
        miss = str(r.get("Missing rate of the raw data (2018 - 2020)", "")).strip()
        period = str(r.get("Specific available time period (default is 3 years from 2018 to 2020)", "")).strip()

        if f not in lut:
            lut[f] = {
                "unit": unit,
                "description": desc,
                "sampling_frequency": samp,
                "missing_rate_raw": miss,
                "available_period": period,
            }
        else:
            # fill blanks only
            if not lut[f]["unit"] and unit: lut[f]["unit"] = unit
            if not lut[f]["description"] and desc: lut[f]["description"] = desc

    return lut


def infer_role(colname: str, desc: str) -> str:
    s = f"{colname} {desc}".lower()
    if "setpoint" in s or "stpt" in s or "_sp" in s:
        return "sp"
    if "cmd" in s or "command" in s:
        return "cmd"
    if "fbk" in s or "feedback" in s:
        return "fbk"
    return "meas"

def default_resample(role: str) -> str:
    if role in ("sp", "cmd"):
        return "ffill"
    if role == "fbk":
        return "last"
    return "mean"

def units_and_convert(unit: str):
    u = unit.strip().lower()
    if u in ("f", "degf", "°f"):
        return ("F", "C", "F_to_C")
    if u in ("c", "degc", "°c"):
        return ("C", "C", "")
    if u in ("%", "percent", "percentage"):
        return ("percent", "frac", "percent_to_frac")
    if u in ("pa",):
        return ("Pa", "Pa", "")
    if u in ("inh2o", "in_h2o", "inchesh2o", "in water"):
        return ("inH2O", "Pa", "inH2O_to_Pa")
    if u in ("w",):
        return ("W", "W", "")
    if u in ("kw",):
        return ("kW", "kW", "")
    if u in ("ppm",):
        return ("ppm", "ppm", "")
    return (unit, unit, "")

PRESSURE_KEYS = ["dp", "press", "pressure", "static", "plenum", "filter", "fltr"]

def is_pressure_like(colname: str, desc: str, unit: str) -> bool:
    s = f"{colname} {desc} {unit}".lower()
    return any(k in s for k in PRESSURE_KEYS)

def infer_sign_from_data(csv_path: Path, time_col: str, value_col: str):
    # Heuristic: if median is negative, propose sign=-1
    try:
        df = pd.read_csv(csv_path, usecols=[time_col, value_col], nrows=SIGN_SAMPLE_ROWS)
        x = pd.to_numeric(df[value_col], errors="coerce").dropna()
        if x.empty:
            return +1, "no_numeric_data"
        med = float(x.median())
        if med < 0:
            return -1, f"median<0 ({med:.3g})"
        return +1, f"median>=0 ({med:.3g})"
    except Exception as e:
        return +1, f"infer_error: {e}"

# -------------------------
# Main
# -------------------------
def main():
    # Load Brick
    g = Graph()
    g.parse(str(TTL_PATH), format="turtle")
    bundle = get_zone_bundle(g, ZONE_NAME)

    # Scan headers in CSV_ROOT (prefer clean data)
    file_cols = scan_csv_headers(CSV_ROOT)
    # Excel LUT (file+col → unit/desc/etc.)
    excel_lut = load_excel_lut(EXCEL_DESC)
    # Candidate points from Brick topology
    all_points = bundle["zone_points"] + bundle["rtu_points"] + bundle["floor_points"]
    all_point_names = sorted(set(p["name"] for p in all_points))
    # Map Brick point → (file,time_col,value_col)
    brick_points = {}
    unmapped = []

    for pn in all_point_names:
        m = match_point_to_column(pn, file_cols)
        if m is None:
            unmapped.append(pn)
            continue
        base = {"file": m["file"], "time_col": infer_time_col(file_cols[m["file"]]), "value_col": m["value_col"]}
        # Enrich from Excel (if row exists)
        print(base["value_col"])
        meta = excel_lut.get((base["file"]))
        if meta:
            desc = meta.get("description", "")
            unit = meta.get("unit", "")
            role = infer_role(base["value_col"], desc)
            resample = default_resample(role)

            u_in, u_out, conv = units_and_convert(unit)

            sign = +1
            sign_reason = "default +1"
            if INFER_SIGN_FOR_PRESSURE and is_pressure_like(base["value_col"], desc, unit):
                csv_path = CSV_ROOT / base["file"]
                if csv_path.exists():
                    sign, sign_reason = infer_sign_from_data(csv_path, base["time_col"], base["value_col"])
                else:
                    sign_reason = "csv_missing_for_sign_infer"

            base.update({
                "description": desc,
                "unit": unit,
                "sampling_frequency": meta.get("sampling_frequency", ""),
                "missing_rate_raw": meta.get("missing_rate_raw", ""),
                "available_period": meta.get("available_period", ""),
                "role": role,
                "resample": resample,
                "units_in": u_in,
                "units_out": u_out,
                "convert": conv,
                "sign": int(sign),
                "sign_reason": sign_reason,
            })
        else:
            # Not found in Excel: keep mapping only
            base.update({
                "role": "unknown",
                "resample": "mean",
                "units_in": "",
                "units_out": "",
                "convert": "",
                "sign": 1,
                "sign_reason": "no_excel_row",
            })

        brick_points[pn] = base

    # Canonical signals (your “app” signals)
    signals_from_types = pick_canonical_signals(bundle, brick_points)

    # Store signals as full specs (copied from brick_points)
    signals = {}
    for canon_name, brick_name in signals_from_types.items():
        signals[canon_name] = dict(brick_points[brick_name])

    # Final YAML contract
    out = {
        "timezone": TIMEZONE,
        "sampling": SAMPLING,
        "timestamp": {
            "source_tz": "local",        # change to "UTC" if your CSV timestamps are UTC
            "local_tz": TIMEZONE,
        },
        "zone": ZONE_NAME,
        "topology": {
            "vavs": bundle["vavs"],
            "rtus": bundle["rtus"],
            "floors": bundle["floors"],
        },
        "signals": signals,           # small, app-facing signals with full contract fields
        "brick_points": brick_points, # full mapping+contract for everything mapped
        "unmapped_points": unmapped,
    }

    OUT_YAML.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    print(f"\nWrote FINAL YAML -> {OUT_YAML}")
    print(f"Mapped points: {len(brick_points)} | Unmapped: {len(unmapped)}")
    print("Signals keys:", list(signals.keys()))
    print("CSV root used:", CSV_ROOT)

if __name__ == "__main__":
    main()

