import argparse, json
import numpy as np
import pandas as pd
import yaml
from typing import Dict

def _minmax(series: pd.Series) -> np.ndarray:
    """Safe min-max scaler → [0,1], handles constant series."""
    x = series.to_numpy(dtype=float)
    lo = np.nanmin(x) if x.size else 0.0
    hi = np.nanmax(x) if x.size else 1.0
    denom = (hi - lo) if (hi - lo) > 0 else 1.0
    return (x - lo) / denom

def _get(cfg: Dict, path: str, default):
    """Nested get: _get(cfg, 'field.length_m', 165.0)."""
    cur = cfg
    for key in path.split("."):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur

def main():
    ap = argparse.ArgumentParser(description="Compute AFL workload metrics from tracking CSV.")
    ap.add_argument("--tracking", required=True, help="Path to tracking.csv")
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    ap.add_argument("--out-json", required=True, help="Output JSON path")
    ap.add_argument("--out-csv", default="", help="(Optional) Output CSV path")
    args = ap.parse_args()

    # Load config
    cfg = yaml.safe_load(open(args.config))
    fps = float(_get(cfg, "fps", 25.0))
    L   = float(_get(cfg, "field.length_m", 165.0))
    W   = float(_get(cfg, "field.width_m", 135.0))
    hsr_kmh = float(_get(cfg, "thresholds.hsr_kmh", 18.0))
    vmax_ms = float(_get(cfg, "thresholds.vmax_ms", 12.0))  # optional; not in YAML by default
    weights = _get(cfg, "weights", {"distance":0.5, "hsr":0.4, "work_rest":0.1})
    risk_bands = _get(cfg, "risk_bands", [
        {"max": 40, "label": "Low"},
        {"max": 70, "label": "Moderate"},
        {"max": 999, "label": "High"},
    ])

    # Load tracking data
    df = pd.read_csv(args.tracking).sort_values(["player_id", "frame_id"]).reset_index(drop=True)

    # Basic column check
    required = {"frame_id", "player_id", "timestamp_s", "cx", "cy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # --- Scale pixel coords to metres (simple affine based on data extents) ---
    sx = W / max(df["cx"].max() - df["cx"].min(), 1e-9)
    sy = L / max(df["cy"].max() - df["cy"].min(), 1e-9)
    df["cx_m"] = (df["cx"] - df["cx"].min()) * sx
    df["cy_m"] = (df["cy"] - df["cy"].min()) * sy

    # --- Per-step distance with clamp (anti-teleport) ---
    dx = df.groupby("player_id")["cx_m"].diff()
    dy = df.groupby("player_id")["cy_m"].diff()
    step = np.hypot(dx, dy)
    df["step_dist_m"] = np.where(np.isfinite(step) & (step <= 0.5), step, 0.0).astype(float)

    # --- Frame delta time & speed ---
    dt = df.groupby("player_id")["frame_id"].diff() / fps
    df["dt_s"] = np.where((dt > 0) & np.isfinite(dt), dt, 0.0).astype(float)

    speed_mps = np.where(df["dt_s"] > 0, df["step_dist_m"] / df["dt_s"], np.nan)
    df["speed_mps"] = np.clip(speed_mps, 0.0, vmax_ms)

    # --- High-Speed Running metres (HSR) ---
    kmh = (df["speed_mps"] * 3.6).astype(float)
    df["hsr_m"] = np.where(kmh >= hsr_kmh, df["step_dist_m"], 0.0).astype(float)

    # --- Work/Rest ratio: (time in HSR) / (time not in HSR) ---
    # Measure time in seconds using dt_s, not row counts.
    high_time_s = np.where(kmh >= hsr_kmh, df["dt_s"], 0.0)
    per_player_high_s = pd.Series(high_time_s).groupby(df["player_id"]).sum()
    per_player_total_s = df.groupby("player_id")["dt_s"].sum()
    # Align to group keys later
    # Note: work_rest = high / (total - high); clip denom to avoid /0
    # We'll materialize after we build the per-player table.

    # --- Per-player summary ---
    g = (df.groupby("player_id")
           .agg(distance_m=("step_dist_m", "sum"),
                hsr_m=("hsr_m", "sum"),
                mean_speed_mps=("speed_mps", "mean"),
                max_speed_mps=("speed_mps", "max"),
                total_time_s=("dt_s", "sum"))
           .reset_index())

    # Work/rest ratio
    high_aligned = per_player_high_s.reindex(g["player_id"]).fillna(0.0).to_numpy()
    total_aligned = g["total_time_s"].to_numpy(dtype=float)
    denom = np.clip(total_aligned - high_aligned, 1e-9, None)
    g["work_rest_ratio"] = (high_aligned / denom)

    # --- Workload score (0–100) ---
    # Only use keys that exist to avoid KeyError if config has extra weights you haven't computed yet.
    parts = []
    if "distance" in weights:
        parts.append(weights["distance"] * _minmax(g["distance_m"]))
    if "hsr" in weights:
        parts.append(weights["hsr"] * _minmax(g["hsr_m"]))
    if "work_rest" in weights:
        parts.append(weights["work_rest"] * _minmax(g["work_rest_ratio"]))
    # Sum parts, clamp to [0,1], scale to 0–100
    score = 100.0 * np.clip(np.sum(parts, axis=0) if parts else np.zeros(len(g)), 0.0, 1.0)
    g["workload_score"] = score

    # --- Risk label from risk_bands ---
    def band_label(s: float) -> str:
        for b in risk_bands:
            if s <= float(b["max"]):
                return str(b["label"])
        return str(risk_bands[-1]["label"])
    g["fatigue_risk"] = g["workload_score"].apply(band_label)

    # --- Optional CSV ---
    if args.out_csv:
        g.to_csv(args.out_csv, index=False)

    # --- Build JSON payload (guard against NaN/inf) ---
    def clean_float(v) -> float:
        try:
            fv = float(v)
            return fv if np.isfinite(fv) else 0.0
        except Exception:
            return 0.0

    payload = {
        "meta": {
            "fps": fps,
            "field": {"length_m": L, "width_m": W},
            "hsr_kmh": hsr_kmh,
        },
        "players": [
            {
                "player_id": int(r.player_id),
                "distance_m": clean_float(r.distance_m),
                "hsr_m": clean_float(r.hsr_m),
                "mean_speed_mps": clean_float(r.mean_speed_mps),
                "max_speed_mps": clean_float(r.max_speed_mps),
                "total_time_s": clean_float(r.total_time_s),
                "work_rest_ratio": clean_float(r.work_rest_ratio),
                "workload_score": clean_float(r.workload_score),
                "fatigue_risk": str(r.fatigue_risk),
            }
            for _, r in g.iterrows()
        ],
    }

    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2)

if __name__ == "__main__":
    main()