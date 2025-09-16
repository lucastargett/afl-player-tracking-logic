#!/usr/bin/env python3
import argparse, json
import numpy as np
import pandas as pd
import yaml
from typing import Dict, List, Any

def _minmax(series: pd.Series) -> np.ndarray:
    x = series.to_numpy(dtype=float)
    if x.size == 0:
        return x
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    denom = (hi - lo) if (hi - lo) > 0 else 1.0
    return (x - lo) / denom

def _get(cfg: Dict, path: str, default):
    cur = cfg
    for key in path.split("."):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur

def _clean_float(v) -> float:
    try:
        fv = float(v)
        return fv if np.isfinite(fv) else 0.0
    except Exception:
        return 0.0

def _band_label(s: float, bands: List[Dict[str, Any]]) -> str:
    for b in bands:
        if s <= float(b["max"]):
            return str(b["label"])
    return str(bands[-1]["label"])

def _ensure_timestamps_and_quarters(df: pd.DataFrame, fps: float, edges_s: list[int|float]) -> pd.DataFrame:
    """
    Ensures df has a reliable 'timestamp_s' and a 'quarter' column.
    - If timestamp_s is missing or non-monotonic per player, synthesise from frame_id/fps.
    - Assign quarters by edges_s; if still impossible, fall back to Q1.
    """
    df = df.copy()

    def _is_mono_increasing(x: pd.Series) -> bool:
        # allow tiny jitter; treat NaNs as bad
        if x.isna().any(): 
            return False
        d = x.diff().dropna()
        if d.empty:
            return False
        # require 95% of diffs >= 0 (some feeds can duplicate frames)
        return (d >= -1e-6).mean() >= 0.95

    use_estimated = False
    has_ts = "timestamp_s" in df.columns

    if not has_ts:
        use_estimated = True
    else:
        # check monotonicity per player
        ok = True
        for _, g in df.groupby("player_id", sort=False):
            if not _is_mono_increasing(g["timestamp_s"]):
                ok = False
                break
        if not ok:
            use_estimated = True

    if use_estimated:
        # Synthesize timestamps from frame_id per player (start at 0s for each player)
        df["timestamp_s"] = (
            (df["frame_id"] - df.groupby("player_id")["frame_id"].transform("min")) / float(max(fps, 1e-9))
        )

    # Assign quarters using edges_s
    try:
        bins = [-np.inf] + list(edges_s) + [np.inf]
        df["quarter"] = pd.cut(df["timestamp_s"], bins=bins, labels=[1,2,3,4]).astype(int)
    except Exception:
        # last resort: put everything in Q1
        df["quarter"] = 1

    return df, ("estimated" if use_estimated else "provided")

def main():
    ap = argparse.ArgumentParser(description="Compute AFL workload metrics (game total + per quarter).")
    ap.add_argument("--tracking", required=True, help="Path to tracking.csv")
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    ap.add_argument("--out-json", required=True, help="Output JSON path")
    ap.add_argument("--out-csv", default="", help="(Optional) Output CSV path for game totals")
    args = ap.parse_args()

    # --- Config ---
    cfg = yaml.safe_load(open(args.config))
    fps = float(_get(cfg, "fps", 25.0))
    L   = float(_get(cfg, "field.length_m", 165.0))
    W   = float(_get(cfg, "field.width_m", 135.0))
    hsr_kmh = float(_get(cfg, "thresholds.hsr_kmh", 18.0))
    vmax_ms = float(_get(cfg, "thresholds.vmax_ms", 12.0))  # optional
    weights = _get(cfg, "weights", {"distance":0.5, "hsr":0.4, "work_rest":0.1})
    risk_bands = _get(cfg, "risk_bands", [
        {"max": 40, "label": "Low"},
        {"max": 70, "label": "Moderate"},
        {"max": 999, "label": "High"},
    ])

    # Quarter edges in seconds from game start (20 min quarters by default)
    quarter_edges_s = _get(cfg, "quarters.edges_s", [1200, 2400, 3600])

    # --- Load tracking ---
    df = pd.read_csv(args.tracking).sort_values(["player_id", "frame_id"]).reset_index(drop=True)
    required = {"frame_id", "player_id", "cx", "cy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # --- Scale pixels → metres (simple affine by extents) ---
    sx = W / max(df["cx"].max() - df["cx"].min(), 1e-9)
    sy = L / max(df["cy"].max() - df["cy"].min(), 1e-9)
    df["cx_m"] = (df["cx"] - df["cx"].min()) * sx
    df["cy_m"] = (df["cy"] - df["cy"].min()) * sy

    # --- Per-step distance with clamp (anti-teleport) ---
    dx = df.groupby("player_id")["cx_m"].diff()
    dy = df.groupby("player_id")["cy_m"].diff()
    step = np.hypot(dx, dy)
    df["step_dist_m"] = np.where(np.isfinite(step) & (step <= 0.5), step, 0.0).astype(float)

    # --- dt & speed ---
    dt = df.groupby("player_id")["frame_id"].diff() / fps
    df["dt_s"] = np.where((dt > 0) & np.isfinite(dt), dt, 0.0).astype(float)
    speed_mps = np.where(df["dt_s"] > 0, df["step_dist_m"] / df["dt_s"], np.nan)
    df["speed_mps"] = np.clip(speed_mps, 0.0, vmax_ms)
    kmh = (df["speed_mps"] * 3.6).astype(float)

    # --- Sprint detection (events) ---
    sprint_kmh = float(_get(cfg, "thresholds.sprint_kmh", 24.0))
    min_sprint_dur_s = float(_get(cfg, "thresholds.min_sprint_dur_s", 1.0))
    rs_window_s = float(_get(cfg, "thresholds.repeat_sprint_window_s", 60.0))

    df["is_sprint"] = kmh >= sprint_kmh

    # Run-length encode per player to get sprint segments
    def _sprint_segments(g: pd.DataFrame):
        # g is sorted by frame_id already
        mask = g["is_sprint"].to_numpy()
        if mask.size == 0:
            return pd.DataFrame(columns=["player_id","seg_id","t_start","t_end","dur_s",
                                        "dist_m","mean_speed_mps"])
        # segment ids
        boundaries = np.where(np.diff(mask.astype(int)) != 0)[0] + 1
        seg_ids = np.zeros(len(mask), dtype=int)
        seg_ids[boundaries] = 1
        seg_ids = np.cumsum(seg_ids)
        g = g.copy()
        g["seg_id"] = seg_ids

        # keep only sprint segments (mask True)
        sprint_rows = g[mask]
        if sprint_rows.empty:
            return pd.DataFrame(columns=["player_id","seg_id","t_start","t_end","dur_s",
                                        "dist_m","mean_speed_mps"])

        agg = (sprint_rows.groupby("seg_id")
                .agg(player_id=("player_id","first"),
                    t_start=("timestamp_s","min"),
                    t_end=("timestamp_s","max"),
                    dur_s=("dt_s","sum"),
                    dist_m=("step_dist_m","sum"),
                    mean_speed_mps=("speed_mps","mean"))
                .reset_index(drop=True))
        # keep segments that last long enough
        agg = agg[agg["dur_s"] >= min_sprint_dur_s].reset_index(drop=True)
        return agg

    sprint_segments = (
        df.groupby("player_id", group_keys=False)
        .apply(_sprint_segments)
        .reset_index(drop=True)
    )

    # --- Repeated Sprint Bouts (clusters of sprints within rs_window_s) ---
    def _count_repeated_bouts(agg: pd.DataFrame) -> pd.DataFrame:
        if agg.empty:
            return pd.DataFrame({"player_id": [], "num_sprints": [], "repeated_sprint_bouts": []})
        out = []
        for pid, gseg in agg.groupby("player_id"):
            gseg = gseg.sort_values("t_start")
            num_sprints = len(gseg)

            # cluster by start-time gaps
            starts = gseg["t_start"].to_numpy()
            gaps = np.diff(starts)
            # new cluster whenever gap > window
            cluster_break = np.insert(gaps > rs_window_s, 0, True)
            cluster_id = np.cumsum(cluster_break)

            sizes = pd.Series(cluster_id).value_counts()
            repeated_bouts = int((sizes >= 2).sum())  # clusters with >=2 sprints

            out.append({
                "player_id": int(pid),
                "num_sprints": int(num_sprints),
                "repeated_sprint_bouts": repeated_bouts,
                "mean_sprint_speed_mps": float(gseg["mean_speed_mps"].mean()) if num_sprints else 0.0,
                "mean_sprint_dist_m": float(gseg["dist_m"].mean()) if num_sprints else 0.0,
            })
        return pd.DataFrame(out)

    sprint_summary = _count_repeated_bouts(sprint_segments)

    # --- HSR metres ---
    df["hsr_m"] = np.where(kmh >= hsr_kmh, df["step_dist_m"], 0.0).astype(float)

    # --- Quarter segmentation (robust) ---
    df, ts_mode = _ensure_timestamps_and_quarters(df, fps=fps, edges_s=quarter_edges_s) 

    # --- GAME TOTAL (per player) ---
    high_time_s = np.where(kmh >= hsr_kmh, df["dt_s"], 0.0)
    per_player_high_s = pd.Series(high_time_s).groupby(df["player_id"]).sum()
    per_player_total_s = df.groupby("player_id")["dt_s"].sum()

    g = (df.groupby("player_id")
           .agg(distance_m=("step_dist_m", "sum"),
                hsr_m=("hsr_m", "sum"),
                mean_speed_mps=("speed_mps", "mean"),
                max_speed_mps=("speed_mps", "max"),
                total_time_s=("dt_s", "sum"))
           .reset_index())
    
    # Merge sprint summary into game totals
    g = g.merge(sprint_summary, on="player_id", how="left").fillna({
        "num_sprints": 0,
        "repeated_sprint_bouts": 0,
        "mean_sprint_speed_mps": 0.0,
        "mean_sprint_dist_m": 0.0
    })

    high_aligned = per_player_high_s.reindex(g["player_id"]).fillna(0.0).to_numpy()
    total_aligned = g["total_time_s"].to_numpy(dtype=float)
    denom = np.clip(total_aligned - high_aligned, 1e-9, None)
    g["work_rest_ratio"] = (high_aligned / denom)

    # Score & risk (game-total)
    parts = []
    if "distance" in weights:   
        parts.append(weights["distance"]  * _minmax(g["distance_m"]))
    if "hsr" in weights:        
        parts.append(weights["hsr"]       * _minmax(g["hsr_m"]))
    if "work_rest" in weights:  
        parts.append(weights["work_rest"] * _minmax(g["work_rest_ratio"]))
    if "repeat_sprints" in weights and "num_sprints" in g:
        parts.append(weights["repeat_sprints"] * _minmax(g["num_sprints"]))

    score_game = 100.0 * np.clip(
        np.sum(parts, axis=0) if parts else np.zeros(len(g)), 0.0, 1.0
    )
    g["workload_score"] = score_game
    g["fatigue_risk"] = g["workload_score"].apply(lambda s: _band_label(s, risk_bands))

    # --- PER QUARTER (per player, per quarter) ---
    # Aggregations
    gq = (df.groupby(["player_id","quarter"])
            .agg(distance_m=("step_dist_m","sum"),
                 hsr_m=("hsr_m","sum"),
                 mean_speed_mps=("speed_mps","mean"),
                 max_speed_mps=("speed_mps","max"),
                 total_time_s=("dt_s","sum"))
            .reset_index())

    # Work/rest per quarter
    high_time_q = pd.Series(np.where(kmh >= hsr_kmh, df["dt_s"], 0.0)).groupby([df["player_id"], df["quarter"]]).sum()
    total_time_q = df.groupby(["player_id","quarter"])["dt_s"].sum()
    wr_q = (high_time_q / (total_time_q - high_time_q).clip(lower=1e-9)).reset_index(name="work_rest_ratio")
    gq = gq.merge(wr_q, on=["player_id","quarter"], how="left").fillna(0.0)

    # Quarter scores are computed across ALL (player, quarter) rows so they’re comparable
    parts_q = []
    if "distance" in weights:   parts_q.append(weights["distance"]  * _minmax(gq["distance_m"]))
    if "hsr" in weights:        parts_q.append(weights["hsr"]       * _minmax(gq["hsr_m"]))
    if "work_rest" in weights:  parts_q.append(weights["work_rest"] * _minmax(gq["work_rest_ratio"]))
    score_q = 100.0 * np.clip(np.sum(parts_q, axis=0) if parts_q else np.zeros(len(gq)), 0.0, 1.0)
    gq["workload_score"] = score_q
    gq["fatigue_risk"] = gq["workload_score"].apply(lambda s: _band_label(s, risk_bands))

    # Optional CSV of game totals
    if args.out_csv:
        g.to_csv(args.out_csv, index=False)

    # --- Build JSON payload with quarters nested under each player ---
    players_payload = []
    for pid, row in g.set_index("player_id").iterrows():
        # quarters for this player
        qrows = gq[gq["player_id"] == pid].sort_values("quarter")
        quarters_payload = [
            {
                "q":          int(qr.quarter),
                "distance_m": _clean_float(qr.distance_m),
                "hsr_m":      _clean_float(qr.hsr_m),
                "mean_speed_mps": _clean_float(qr.mean_speed_mps),
                "max_speed_mps":  _clean_float(qr.max_speed_mps),
                "total_time_s":   _clean_float(qr.total_time_s),
                "work_rest_ratio": _clean_float(qr.work_rest_ratio),
                "workload_score":  _clean_float(qr.workload_score),
                "fatigue_risk":    str(qr.fatigue_risk),
            }
            for _, qr in qrows.iterrows()
        ]

        players_payload.append({
            "player_id": int(pid),
            "game_total": {
                "distance_m":     _clean_float(row.distance_m),
                "hsr_m":          _clean_float(row.hsr_m),
                "mean_speed_mps": _clean_float(row.mean_speed_mps),
                "max_speed_mps":  _clean_float(row.max_speed_mps),
                "total_time_s":   _clean_float(row.total_time_s),
                "work_rest_ratio": _clean_float(row.work_rest_ratio),
                "workload_score":  _clean_float(row.workload_score),
                "fatigue_risk":    str(row.fatigue_risk),
                "num_sprints":          int(row.get("num_sprints", 0)),
                "repeated_sprint_bouts": int(row.get("repeated_sprint_bouts", 0)),
                "mean_sprint_speed_mps": _clean_float(row.get("mean_sprint_speed_mps", 0.0)),
                "mean_sprint_dist_m":    _clean_float(row.get("mean_sprint_dist_m", 0.0)),
            },
            "quarters": quarters_payload
        })

    payload = {
    "meta": {
        "fps": fps,
        "field": {"length_m": L, "width_m": W},
        "hsr_kmh": hsr_kmh,
        "quarters": {"edges_s": quarter_edges_s, "labels": [1,2,3,4], "timestamp_mode": ts_mode}
    },
    "players": players_payload
}

    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2)

if __name__ == "__main__":
    main()