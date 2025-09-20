#!/usr/bin/env python3
import argparse
import pandas as pd
import yaml
import os

# import the pipeline from your heatmaps/ package
from heatmaps import pipeline, render

def main():
    ap = argparse.ArgumentParser(description="Generate heatmaps from tracking data.")
    ap.add_argument(
        "--tracking",
        default="examples/tracking2.0.csv",
        help="Path to tracking2.0.csv (default: examples/tracking2.0.csv)"
    )
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    ap.add_argument("--out-dir", required=True, help="Directory for saving heatmaps")
    ap.add_argument("--group-by", default="player_id", help="Column to group by (e.g. player_id or team_id)")
    args = ap.parse_args()

    # load config
    cfg = yaml.safe_load(open(args.config))

    # load tracking
    df = pd.read_csv(args.tracking)

    # ensure output dir exists
    os.makedirs(args.out_dir, exist_ok=True)

    # run pipeline
    heatmaps = pipeline.compute_heatmaps(df, cfg, group_by=args.group_by)

    # render images
    for key, grid in heatmaps.items():
        out_path = os.path.join(args.out_dir, f"heatmap_{key}.png")
        render.save_heatmap(grid, out_path)
        print(f"Saved heatmap for {key} → {out_path}")

if __name__ == "__main__":
    main()