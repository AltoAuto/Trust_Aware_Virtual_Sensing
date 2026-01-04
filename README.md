# Trust-Aware Virtual Sensing + Supervisory DCV (Building 59)

Turn noisy CO₂ + BAS signals into **trusted estimates + confidence**, then use them in a **constraint-based ventilation supervisor** to meet IAQ limits with minimum energy — robust to sensor drift + missing data.

## Why?
Smart-building sensing for control runs into practical blockers:

- occupant sensing accuracy is often insufficient for closed-loop control
- “peel-and-stick” sensing (easy-to-deploy add-on sensors) hits power/wiring/maintenance limits
- virtual sensing must propagate uncertainty from sensor errors into control decisions

This project is built to survive those realities

## Objective 
Estimate **CO₂ generation rate** G(t) (preferred over headcount) and related latent ventilation effectiveness/rate (as needed), with **uncertainty**, then use a **constraint-based DCV supervisor** to:
- keep CO₂ below limits (with confidence-aware constraints)
- minimize ventilation energy
- outperform rule baselines (raw CO₂-threshold DCV and schedule-based ventilation)

under sensor drift, missing data, and messy BAS signals.

## Dataset
Uses the **LBNL Building 59** dataset (3 years of BAS-style data) 
The dataset includes cleaned time-series CSVs plus a Brick model (.ttl) and metadata JSON (three-layer structure)
"https://datadryad.org/dataset/doi:10.7941/D1N33Q"

## Roadmap (Layers)
- Layer 0 — point contract: map Brick points → (file, time_col, value_col) and export a canonical table
- Layer 1 — DataQA/Trust: missingness, drift detection, range checks, unit checks, timestamp sanity, point identity (cmd/fbk/meas)
- Layer 2 — Minimal zone CO₂ model: define dynamics + inputs from BAS
- Layer 3 — Estimation (primary): MHE (and EKF/UKF baselines) for G(t) under noise/missingness
- Layer 4 — Supervisory DCV: constraint handling with uncertainty-aware margins
- Layer 5 — Evaluation: stress tests (drift, dropouts), IAQ violations, energy proxies, estimator calibration

## Outputs (what this repo will produce)
- `point_contract.yaml` (signals + brick_points with metadata + preprocessing rules)
- `layer0.csv` (canonical time-aligned table)
- Layer 1 reports: missingness heatmap, drift flags, trust scores
- Layer 3 estimates: G(t) with uncertainty bands
- Layer 4 results: CO₂ constraint satisfaction vs energy proxy + baseline comparisons

## Citation
"https://datadryad.org/dataset/doi:10.7941/D1N33Q"
