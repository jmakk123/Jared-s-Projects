"""SPARTAN error analysis on the benchmarked test split.

For every benchmark row we compare the SPARTAN prediction against the ground
truth on five structural dimensions:

    cell_count      number of <td>/<th> in pred vs gt
    row_count       number of <tr> in pred vs gt
    column_count    width of the widest row
    header_rows     contiguous all-<th> rows from the top
    spans           presence of colspan>1 or rowspan>1

Each image is then placed into the FIRST matching category, in this order:

    empty_output         pred has no table or zero cells
    span_loss            gt has spans, pred has none
    header_misdetect     |Δ header_rows| ≥ 1
    column_drift         |Δ n_cols| / gt_n_cols > 0.25
    row_merge            pred_n_rows < 0.7 * gt_n_rows  (rows collapsed)
    row_oversplit        pred_n_rows > 1.4 * gt_n_rows  (rows split)
    cell_count_off       |Δ n_cells| / gt_n_cells > 0.30
    low_teds             teds < 0.4  (catch-all for residual content errors)
    success              everything else (TEDS ≥ 0.4)

Outputs:
    error_categories.csv     per-image category + raw structural stats
    error_summary.csv        category counts, percentages, mean TEDS per category

Run:
    python TableSight/models/spartan/error_analysis.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict

import pandas as pd
from lxml import html as lhtml


HERE = Path(__file__).resolve().parent
DEFAULT_BENCHMARK = HERE / "benchmark.csv"
DEFAULT_CATEGORIES = HERE / "error_categories.csv"
DEFAULT_SUMMARY    = HERE / "error_summary.csv"


# ───────────────────────────────────────────────────────────────────────
# Structural counters
# ───────────────────────────────────────────────────────────────────────

def _safe_parse(html_str: str):
    if not html_str or "<table" not in html_str.lower():
        return None
    try:
        root = lhtml.fragment_fromstring(html_str, create_parent="div")
        return root.find(".//table")
    except Exception:
        return None


def count_cells(html_str: str) -> int:
    t = _safe_parse(html_str)
    return 0 if t is None else len(t.xpath(".//td | .//th"))


def count_rows(html_str: str) -> int:
    t = _safe_parse(html_str)
    return 0 if t is None else len(t.xpath(".//tr"))


def count_cols(html_str: str) -> int:
    t = _safe_parse(html_str)
    if t is None: return 0
    widths = []
    for tr in t.xpath(".//tr"):
        w = 0
        for cell in tr.xpath("./td | ./th"):
            cs = cell.get("colspan", "1")
            try: w += int(cs)
            except ValueError: w += 1
        widths.append(w)
    return max(widths) if widths else 0


def count_header_rows(html_str: str) -> int:
    t = _safe_parse(html_str)
    if t is None: return 0
    n = 0
    for tr in t.xpath(".//tr"):
        cells = tr.xpath("./td | ./th")
        if cells and all(c.tag == "th" for c in cells):
            n += 1
        else:
            break
    return n


def has_spans(html_str: str) -> bool:
    return bool(re.search(r'(?:col|row)span=["\']?\s*(?:[2-9]|\d{2,})', html_str or ""))


# ───────────────────────────────────────────────────────────────────────
# Classifier
# ───────────────────────────────────────────────────────────────────────

CATEGORY_DESCRIPTIONS: Dict[str, str] = {
    "empty_output":     "no table or zero cells extracted",
    "span_loss":        "ground truth has colspan/rowspan; prediction has none",
    "header_misdetect": "wrong number of header rows",
    "column_drift":     "column count off by more than 25%",
    "row_merge":        "row count collapsed to under 70% of ground truth",
    "row_oversplit":    "row count expanded past 140% of ground truth",
    "cell_count_off":   "total cell count off by more than 30%",
    "low_teds":         "TEDS under 0.4 with no other category triggered",
    "success":          "TEDS at or above 0.4 with no major structural error",
}


def classify(pred_html: str, gt_html: str, teds: float) -> Dict[str, object]:
    p_cells = count_cells(pred_html)
    g_cells = count_cells(gt_html)
    p_rows  = count_rows(pred_html)
    g_rows  = count_rows(gt_html)
    p_cols  = count_cols(pred_html)
    g_cols  = count_cols(gt_html)
    p_hdrs  = count_header_rows(pred_html)
    g_hdrs  = count_header_rows(gt_html)
    p_span  = has_spans(pred_html)
    g_span  = has_spans(gt_html)

    stats = {
        "pred_cells": p_cells, "gt_cells": g_cells,
        "pred_rows":  p_rows,  "gt_rows":  g_rows,
        "pred_cols":  p_cols,  "gt_cols":  g_cols,
        "pred_hdrs":  p_hdrs,  "gt_hdrs":  g_hdrs,
        "pred_span":  p_span,  "gt_span":  g_span,
    }

    if p_cells == 0:
        cat = "empty_output"
    elif g_span and not p_span:
        cat = "span_loss"
    elif g_hdrs > 0 and abs(p_hdrs - g_hdrs) >= 1:
        cat = "header_misdetect"
    elif g_cols > 0 and abs(p_cols - g_cols) / g_cols > 0.25:
        cat = "column_drift"
    elif g_rows > 0 and p_rows < 0.70 * g_rows:
        cat = "row_merge"
    elif g_rows > 0 and p_rows > 1.40 * g_rows:
        cat = "row_oversplit"
    elif g_cells > 0 and abs(p_cells - g_cells) / g_cells > 0.30:
        cat = "cell_count_off"
    elif teds < 0.4:
        cat = "low_teds"
    else:
        cat = "success"

    stats["category"] = cat
    return stats


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark",  default=str(DEFAULT_BENCHMARK))
    ap.add_argument("--splits-csv",
                    default="data/processed_local/training_fintabnet_pool_splits_local.csv",
                    help="Source-of-truth CSV with ground-truth HTML")
    ap.add_argument("--out-categories", default=str(DEFAULT_CATEGORIES))
    ap.add_argument("--out-summary",    default=str(DEFAULT_SUMMARY))
    args = ap.parse_args()

    bench  = pd.read_csv(args.benchmark)
    splits = pd.read_csv(args.splits_csv)
    if "img_id" in splits.columns and "image_id" not in splits.columns:
        splits = splits.rename(columns={"img_id": "image_id"})

    pred_col = "pred_html" if "pred_html" in bench.columns else None
    if pred_col is None:
        raise SystemExit("benchmark.csv has no 'pred_html' column; re-run "
                         "train_spartan.py --benchmark first")

    df = bench.merge(splits[["image_id", "html"]], on="image_id", how="left",
                     suffixes=("", "_gt"))
    df = df.rename(columns={"html": "gt_html"})
    df = df.dropna(subset=["gt_html"])
    print(f"Classifying {len(df)} benchmark rows…")

    rows = []
    for _, r in df.iterrows():
        s = classify(r[pred_col], r["gt_html"], float(r.get("teds", 0.0)))
        s.update({"image_id": r["image_id"],
                  "teds":     float(r.get("teds", 0.0)),
                  "teds_s":   float(r.get("teds_s", 0.0))})
        rows.append(s)
    out = pd.DataFrame(rows)

    out.to_csv(args.out_categories, index=False)
    print(f"Wrote {args.out_categories}  ({len(out)} rows)")

    summary = (out.groupby("category")
                  .agg(count=("image_id", "size"),
                       mean_TEDS=("teds", "mean"),
                       mean_TEDS_S=("teds_s", "mean"))
                  .round(4))
    summary["pct"] = (summary["count"] / summary["count"].sum() * 100).round(1)
    summary["description"] = summary.index.map(CATEGORY_DESCRIPTIONS).fillna("")
    summary = summary[["count", "pct", "mean_TEDS", "mean_TEDS_S", "description"]]
    summary = summary.sort_values("count", ascending=False)
    summary.to_csv(args.out_summary)
    print(f"Wrote {args.out_summary}")
    print()
    print(summary.to_string())

    # Print a representative low-TEDS example per category
    print("\nRepresentative failure cases (lowest TEDS per category):")
    for cat in summary.index:
        sub = out[out.category == cat].sort_values("teds").head(1)
        if not sub.empty:
            r = sub.iloc[0]
            print(f"  {cat:18s} → {r['image_id']:25s} TEDS={r['teds']:.3f} "
                  f"(pred {r['pred_rows']}r×{r['pred_cols']}c vs gt "
                  f"{r['gt_rows']}r×{r['gt_cols']}c)")


if __name__ == "__main__":
    main()
