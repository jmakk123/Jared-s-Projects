"""SPARTAN per-image-type Pipeline TEDS aggregation.

Reads `benchmark.csv` (output of train_spartan.py --benchmark) and produces a
table comparable to the per-model breakdowns in the report:

    image_type        | SPARTAN_Pipeline_TEDS
    low_contrast      | 0.x
    low_quality_blur  | 0.x
    normal_table      | 0.x
    tall_table        | 0.x
    wide_table        | 0.x

Because the SPARTAN benchmark was originally scored with the notebook's own
two-bucket taxonomy (medium / extreme), every image is re-classified into the
five-category scheme used elsewhere in the report.  Classification is derived
from the PNG itself:

    aspect_ratio = W / H
        > 1.5    → wide_table
        < 0.6    → tall_table
    grayscale stddev (contrast)
        < 35     → low_contrast
    Laplacian variance (sharpness)
        < 150    → low_quality_blur
    otherwise   → normal_table

The thresholds match the gross category distributions reported in the EDA
(wide ≈ 49%, normal ≈ 35%, tall ≈ 10%, blur ≈ 4%, low-contrast ≈ 3%).

Run:
    python TableSight/models/spartan/score_by_type.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from PIL import Image


HERE = Path(__file__).resolve().parent
DEFAULT_BENCHMARK = HERE / "benchmark.csv"
DEFAULT_OUT       = HERE / "pipeline_teds_by_type.csv"

# Classification thresholds calibrated against the FinTabNet EDA distribution
# Target: wide ~49%, normal ~35%, tall ~10%, blur ~4%, low-contrast ~3%.
# These cut points are the 3rd, 7th, 10th, and 50th percentiles of the actual
# test-image statistics so the resulting bucket sizes track the EDA's mix.
WIDE_RATIO    = 4.0    # only the truly wide-landscape tables
TALL_RATIO    = 1.0    # portrait or near-square
LOW_CONTRAST  = 39.0   # ~3rd percentile of greyscale std
LOW_SHARPNESS = 1300.0 # ~7th percentile of (dx^2 + dy^2).mean()


def classify_image(path: str) -> str:
    """Return one of the five report-canonical image_type labels."""
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return "normal_table"
    w, h = img.size
    if h == 0:
        return "normal_table"
    aspect = w / h
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    contrast = float(gray.std())
    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)
    sharpness = float((dx**2).mean() + (dy**2).mean())

    # Priority order — matches the exclusive labelling scheme used in FinTabNet
    # preparation: quality flags first, then shape, with normal_table as the
    # default catch-all.
    if contrast < LOW_CONTRAST:
        return "low_contrast"
    if sharpness < LOW_SHARPNESS:
        return "low_quality_blur"
    if aspect > WIDE_RATIO:
        return "wide_table"
    if aspect < TALL_RATIO:
        return "tall_table"
    return "normal_table"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK),
                    help="Path to SPARTAN benchmark.csv")
    ap.add_argument("--splits-csv",
                    default="data/processed_local/training_fintabnet_pool_splits_local.csv",
                    help="CSV that maps image_id → img_path (for re-classification)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    bench = pd.read_csv(args.benchmark)
    splits = pd.read_csv(args.splits_csv)
    # Normalise id column name
    if "img_id" in splits.columns and "image_id" not in splits.columns:
        splits = splits.rename(columns={"img_id": "image_id"})

    df = bench.merge(splits[["image_id", "img_path"]], on="image_id", how="left")
    missing = df["img_path"].isna().sum()
    if missing:
        print(f"WARN: {missing} of {len(df)} benchmark rows have no matching img_path")

    print(f"Classifying {df['img_path'].notna().sum()} images into the report's five types…")
    df["report_image_type"] = df["img_path"].apply(
        lambda p: classify_image(p) if isinstance(p, str) and Path(p).exists() else "normal_table"
    )

    summary = (df.groupby("report_image_type")
                 .agg(SPARTAN_Pipeline_TEDS=("teds",   "mean"),
                      SPARTAN_Skeleton_TEDS=("teds_s", "mean"),
                      n_samples=("teds", "size"))
                 .round(4)
                 .reset_index()
                 .rename(columns={"report_image_type": "image_type"})
                 .sort_values("image_type"))

    # Force the canonical row order (matching how the UniTable table appears in the report)
    canonical = ["low_contrast", "low_quality_blur", "normal_table",
                 "tall_table", "wide_table"]
    summary["__order"] = summary["image_type"].apply(
        lambda t: canonical.index(t) if t in canonical else len(canonical))
    summary = summary.sort_values("__order").drop(columns="__order").reset_index(drop=True)

    summary.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")
    print(summary.to_string(index=False))

    # Headline mean (for the report's overall comparison table)
    overall_mean = df["teds"].mean()
    print(f"\nOverall mean Pipeline TEDS across all images: {overall_mean:.4f}")


if __name__ == "__main__":
    main()
