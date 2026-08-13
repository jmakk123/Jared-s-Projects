"""
Industrial-Scale Synthetic Data Pipeline
==========================================
- Generates 2,000 samples: 1,000 row_heavy + 1,000 col_heavy
- asyncio.Semaphore (concurrency=12) for Apple Silicon stability
- Niche distribution: 20% simple / 30% medium / 50% extreme
- Saves metadata to data/synthetic/labels.jsonl
  Fields: imgid, mode, difficulty, noise_level, html_table, img_path

Usage:
  # Smoke test (10 samples → verification/)
  python pipeline.py --smoke

  # Full run (2000 samples)
  python pipeline.py

  # Custom count
  python pipeline.py --total 500
"""

import argparse
import asyncio
import json
import os
import sys
import uuid
import random
from pathlib import Path

# Allow running from repo root or from src/synthetic/
_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))

from generator import generate_table, get_difficulty
from renderer import TableRenderer, _random_title, _random_footnote, sample_noise_level

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = _THIS_DIR.parent.parent
LABELS_PATH = REPO_ROOT / "data" / "synthetic" / "labels.jsonl"
IMAGES_DIR  = str(REPO_ROOT / "data" / "synthetic" / "images")
VERIFY_DIR  = str(REPO_ROOT / "verification")

# ---------------------------------------------------------------------------
# Single sample generator
# ---------------------------------------------------------------------------

async def generate_one_sample(
    renderer: TableRenderer,
    sem: asyncio.Semaphore,
    mode: str,
    difficulty: str,
    output_dir: str,
) -> dict:
    """Generate, render, augment one sample. Returns metadata dict."""
    async with sem:
        table_data = generate_table(mode=mode, difficulty=difficulty)
        noise_level = sample_noise_level()
        title    = _random_title()
        footnote = _random_footnote()

        imgid    = f"{mode}_{uuid.uuid4().hex[:10]}"
        img_path = await renderer.render(
            table_html  = table_data["html"],
            css         = table_data["css"],
            filename    = imgid,
            title       = title,
            footnote    = footnote,
            noise_level = noise_level,
        )

        return {
            "imgid":       imgid,
            "mode":        mode,
            "difficulty":  difficulty,
            "noise_level": noise_level,
            "n_cols":      table_data["n_cols"],
            "html_table":  table_data["html"],
            "img_path":    img_path,
        }

# ---------------------------------------------------------------------------
# Validation (colspan integrity check)
# ---------------------------------------------------------------------------

def validate_sample(record: dict) -> bool:
    """
    Light structural check: verifies a record dict has required keys and
    that img_path file exists.
    Full colspan/rowspan integrity is handled by validate_labels.py.
    """
    required = {"imgid", "mode", "difficulty", "noise_level", "html_table", "img_path"}
    if not required.issubset(record.keys()):
        return False
    if not os.path.exists(record["img_path"]):
        return False
    return True

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

async def run_smoke_test(concurrency: int = 4) -> None:
    """Generate 20 diverse samples into verification/ directory."""
    os.makedirs(VERIFY_DIR, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    modes = ["row_heavy"] * 10 + ["col_heavy"] * 10
    random.shuffle(modes)
    diffs = random.choices(["simple", "medium", "extreme"], weights=[0.2, 0.3, 0.5], k=20)

    print(f"[SMOKE] Generating 20 verification samples → {VERIFY_DIR}")
    async with TableRenderer(output_dir=VERIFY_DIR) as renderer:
        tasks = [
            generate_one_sample(renderer, sem, mode=modes[i], difficulty=diffs[i], output_dir=VERIFY_DIR)
            for i in range(20)
        ]
        results = await asyncio.gather(*tasks)

    smoke_labels = Path(VERIFY_DIR) / "smoke_labels.jsonl"
    with open(smoke_labels, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"[SMOKE] Done. Labels saved to {smoke_labels}")
    for r in results:
        status = "✓" if validate_sample(r) else "✗"
        print(f"  {status} {r['imgid']} | {r['mode']:10s} | {r['difficulty']:6s} | noise={r['noise_level']}")

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run_pipeline(total_samples: int = 2000, concurrency: int = 12) -> None:
    """
    Generates total_samples split evenly between row_heavy and col_heavy.
    Difficulty distribution: 20% simple / 30% medium / 50% extreme.
    """
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(LABELS_PATH.parent, exist_ok=True)

    half = total_samples // 2
    modes = ["row_heavy"] * half + ["col_heavy"] * (total_samples - half)
    random.shuffle(modes)
    difficulties = random.choices(
        ["simple", "medium", "extreme"],
        weights=[0.20, 0.30, 0.50],
        k=total_samples,
    )

    sem = asyncio.Semaphore(concurrency)
    metadata = []
    batch_size = concurrency * 2   # process in batches for progress logging

    print(f"[PIPELINE] Starting: {total_samples} samples | concurrency={concurrency}")

    async with TableRenderer(output_dir=IMAGES_DIR) as renderer:
        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            tasks = [
                generate_one_sample(
                    renderer     = renderer,
                    sem          = sem,
                    mode         = modes[i],
                    difficulty   = difficulties[i],
                    output_dir   = IMAGES_DIR,
                )
                for i in range(batch_start, batch_end)
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for r in batch_results:
                if isinstance(r, Exception):
                    print(f"  [WARN] Sample failed: {r}")
                else:
                    metadata.append(r)

            print(f"  [{batch_end}/{total_samples}] "
                  f"{batch_end/total_samples*100:.1f}% complete  "
                  f"(good={len(metadata)})")

    # Write labels
    with open(LABELS_PATH, "w") as f:
        for m in metadata:
            f.write(json.dumps(m) + "\n")

    # Summary stats
    row_cnt = sum(1 for m in metadata if m["mode"] == "row_heavy")
    col_cnt = sum(1 for m in metadata if m["mode"] == "col_heavy")
    diff_counts = {d: sum(1 for m in metadata if m["difficulty"] == d) for d in ["simple", "medium", "extreme"]}
    noise_counts = {n: sum(1 for m in metadata if m["noise_level"] == n) for n in ["clean", "medium", "heavy"]}

    print(f"\n[PIPELINE] Complete! {len(metadata)} samples written to {LABELS_PATH}")
    print(f"  Modes  → row_heavy={row_cnt}, col_heavy={col_cnt}")
    print(f"  Diff   → {diff_counts}")
    print(f"  Noise  → {noise_counts}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic Table Data Factory")
    parser.add_argument("--smoke", action="store_true",
                        help="Run 10-sample smoke test into verification/ directory")
    parser.add_argument("--total", type=int, default=2000,
                        help="Total samples to generate (default: 2000)")
    parser.add_argument("--concurrency", type=int, default=12,
                        help="Async semaphore concurrency limit (default: 12)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke:
        asyncio.run(run_smoke_test(concurrency=4))
    else:
        asyncio.run(run_pipeline(total_samples=args.total, concurrency=args.concurrency))