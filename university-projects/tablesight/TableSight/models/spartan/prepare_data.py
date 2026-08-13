"""
Build a splits CSV + image directory locally from the apoidea/fintabnet-html
HuggingFace dataset and/or the raw FinTabNet_train.jsonl.

Output:
    <out_dir>/
        training_fintabnet_pool_splits_local.csv   # img_id, img_path, html, image_type, split, phase
        images/<img_id>.png                         # all images written here

Use this when the team's Drive-resident CSV isn't synced to your local machine.

Usage:
    python TableSight/models/spartan/prepare_local_data.py \\
        --out-dir data/processed_local \\
        --n-samples 1300 \\
        --hard-examples-only

`--hard-examples-only` keeps only tables with colspan>1 or rowspan>1
(the "hierarchical" subset that matters for our project).
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path
from typing import Iterator

import pandas as pd
from PIL import Image
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────
def _has_span(html_str: str) -> bool:
    """True if the HTML table contains any colspan>1 or rowspan>1."""
    if not html_str:
        return False
    return bool(re.search(r'(?:col|row)span\s*=\s*["\']?\s*(?:[2-9]|\d{2,})', html_str))


def _classify_image_type(html_str: str) -> str:
    """Mirror the project's three difficulty buckets."""
    if not html_str:
        return "simple"
    max_cs = max((int(m.group(1)) for m in re.finditer(r'colspan\s*=\s*["\']?(\d+)', html_str)), default=1)
    max_rs = max((int(m.group(1)) for m in re.finditer(r'rowspan\s*=\s*["\']?(\d+)', html_str)), default=1)
    n_header_rows = len(re.findall(r'<th[\s>]', html_str.lower()))
    has_span = max_cs > 1 or max_rs > 1
    if max_cs > 3 or max_rs > 3 or n_header_rows >= 3:
        return "extreme"
    if has_span or n_header_rows == 2:
        return "medium"
    return "simple"


# ──────────────────────────────────────────────────────────────────────────
def _from_hf_dataset() -> Iterator[dict]:
    """Yield {'img_id', 'image_bytes', 'html'} from the HF cache."""
    from datasets import load_dataset
    print("Loading apoidea/fintabnet-html from cache …")
    ds = load_dataset("apoidea/fintabnet-html", "en", split="train")
    for i, row in enumerate(ds):
        img = row.get("image")
        html = row.get("html") or row.get("html_table") or row.get("text")
        if img is None or not html:
            continue
        # The dataset returns a PIL Image directly
        if isinstance(img, Image.Image):
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="PNG")
            img_bytes = buf.getvalue()
        else:
            img_bytes = img if isinstance(img, (bytes, bytearray)) else None
        if not img_bytes:
            continue
        yield {"img_id": f"fintab_{i:06d}", "image_bytes": img_bytes, "html": html}


def _from_jsonl(jsonl_path: Path) -> Iterator[dict]:
    """Yield rows from FinTabNet_train.jsonl (html_table + imgid only — no image)."""
    print(f"Reading {jsonl_path} …")
    with open(jsonl_path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            html = d.get("html_table") or d.get("html") or ""
            img_id = str(d.get("imgid") or d.get("img_id") or "")
            if not (html and img_id):
                continue
            yield {"img_id": img_id, "image_bytes": None, "html": html}


# ──────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data/processed_local",
                   help="Where to write the CSV and images/ subdirectory")
    p.add_argument("--n-samples", type=int, default=1300,
                   help="Total samples to keep (matches the team's 1300-sample pool)")
    p.add_argument("--hard-examples-only", action="store_true",
                   help="Keep only tables with colspan>1 or rowspan>1")
    p.add_argument("--source", choices=["hf", "jsonl", "auto"], default="auto")
    p.add_argument("--jsonl", default="data/raw/FinTabNet_train.jsonl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-frac", type=float, default=0.75)
    p.add_argument("--val-frac",   type=float, default=0.12)
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"; img_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "training_fintabnet_pool_splits_local.csv"

    # Choose source
    sources = []
    if args.source in ("auto", "hf"):
        try:
            sources.append(_from_hf_dataset())
        except Exception as exc:
            print(f"  (HF dataset unavailable: {exc})")
    if args.source in ("auto", "jsonl"):
        if Path(args.jsonl).exists():
            sources.append(_from_jsonl(Path(args.jsonl)))

    if not sources:
        sys.exit("No data source available. Either install `datasets` or "
                 "provide a valid --jsonl path.")

    # Sample to target count, applying the hard-examples filter on the fly
    kept = []
    seen = 0
    import itertools
    for row in itertools.chain(*sources):
        seen += 1
        if args.hard_examples_only and not _has_span(row["html"]):
            continue
        kept.append(row)
        if len(kept) >= args.n_samples:
            break
        if seen % 5000 == 0:
            print(f"  scanned {seen}, kept {len(kept)}")
    print(f"Kept {len(kept)} of {seen} scanned rows.")

    # Materialise images + classify + split
    rows = []
    for r in tqdm(kept, desc="Writing images"):
        img_id = r["img_id"]
        img_path = img_dir / f"{img_id}.png"
        if not img_path.exists() and r.get("image_bytes"):
            img_path.write_bytes(r["image_bytes"])
        if not img_path.exists():
            continue  # skip rows we have no image for
        rows.append({
            "img_id":     img_id,
            "img_path":   str(img_path.resolve()),
            "html":       r["html"],
            "image_type": _classify_image_type(r["html"]),
        })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n = len(df)
    n_tr = int(n * args.train_frac)
    n_va = int(n * args.val_frac)
    df["split"] = "test"
    df.loc[:n_tr, "split"] = "train"
    df.loc[n_tr : n_tr + n_va, "split"] = "val"
    df["phase"] = "phase1"
    df.to_csv(csv_path, index=False)

    print(f"\nWrote {csv_path}")
    print(f"  train: {(df.split == 'train').sum()}")
    print(f"  val:   {(df.split == 'val').sum()}")
    print(f"  test:  {(df.split == 'test').sum()}  (LOCKED)")
    print(f"\nimage_type distribution:")
    print(df["image_type"].value_counts().to_string())
    print(f"\nReady. Run the fine-tuner with:")
    print(f"  python TableSight/models/spartan/train_spartan.py \\")
    print(f"      --csv    {csv_path} \\")
    print(f"      --images {img_dir} \\")
    print(f"      --out-dir TableSight/models/spartan")


if __name__ == "__main__":
    main()
