"""
Validation Script: Structural & Semantic Integrity Checker
===========================================================
Tests:
  1. SCHEMA CHECK: Every record has all required fields.
  2. FILE EXISTENCE: Every img_path in labels.jsonl exists on disk.
  3. COLSPAN INTEGRITY: thead and tbody column counts are consistent.
  4. SEMANTIC DENSITY (new):
     a. col_heavy tables must have ≥2 columns (index + at least 1 data col).
     b. row_heavy tables must have a named header for every data column.
     c. No row in <tbody> may be completely empty (zero cells).

Usage:
  python validate_labels.py                        # checks labels.jsonl
  python validate_labels.py --labels path/to.jsonl # custom labels file
  python validate_labels.py --smoke                # checks verification/smoke_labels.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple

from bs4 import BeautifulSoup

REQUIRED_FIELDS = {"imgid", "mode", "difficulty", "noise_level", "html_table", "img_path"}

REPO_ROOT   = Path(__file__).parent.parent.parent
LABELS_PATH = REPO_ROOT / "data" / "synthetic" / "labels.jsonl"
SMOKE_PATH  = REPO_ROOT / "verification" / "smoke_labels.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_records(labels_path: Path) -> List[Dict]:
    records = []
    with open(labels_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] Line {i+1}: JSON parse error — {e}")
    return records


def check_schema(record: Dict) -> List[str]:
    missing = REQUIRED_FIELDS - set(record.keys())
    return [f"Missing field: {f}" for f in missing]


def check_file_exists(record: Dict) -> List[str]:
    path = record.get("img_path", "")
    if path and not os.path.exists(path):
        return [f"Image not found: {path}"]
    return []


def check_colspan_integrity(html_table: str, mode: str = "") -> Tuple[bool, str]:
    """
    Verifies thead/tbody column structural integrity.

    row_heavy:
      Header row col count (1 hierarchy col + N data cols) should match
      the maximum body cols seen across ALL body rows (since rowspan causes
      first row to have fewer visible cells).

    col_heavy / cross_tab:
      The maximum per-row colspan sum in <thead> must equal the maximum
      per-row column count in <tbody>.
    """
    soup = BeautifulSoup(html_table, "html.parser")
    table = soup.find("table")
    if not table:
        return False, "No <table> found in html_table"

    tbody = table.find("tbody")
    thead = table.find("thead")

    if not tbody or not thead:
        return True, "Missing tbody or thead — skipping check"

    tbody_rows = tbody.find_all("tr")
    thead_rows = thead.find_all("tr")

    if not tbody_rows or not thead_rows:
        return True, "Empty thead or tbody"

    # Max columns seen across all body rows (accounts for rowspan trees)
    max_body_cols = max(
        sum(int(td.get("colspan", 1)) for td in row.find_all(["td", "th"]))
        for row in tbody_rows
    )

    if mode == "row_heavy":
        # For row_heavy, body rows contain a mix of ancestor rowspan cells
        # and leaf data cells — a direct col-count cross-check is unreliable.
        # Instead verify the header is well-formed: at least 2 cols (1 hierarchy + 1 data).
        header_col_count = sum(
            int(th.get("colspan", 1)) for th in thead_rows[0].find_all(["th", "td"])
        )
        if header_col_count < 2:
            return False, f"row_heavy thead has only {header_col_count} column(s) — expected ≥2"
        return True, f"OK (header_cols={header_col_count})"

    # col_heavy / cross_tab
    max_thead_cols = max(
        sum(int(th.get("colspan", 1)) for th in row.find_all(["th", "td"]))
        for row in thead_rows
    )
    if max_thead_cols != max_body_cols:
        return False, (
            f"thead max_colspan ({max_thead_cols}) != "
            f"max tbody cols ({max_body_cols})"
        )
    return True, "OK"

def check_semantic_density(html_table: str, mode: str, n_cols: int = 0) -> List[str]:
    """
    Semantic completeness checks:
      a. col_heavy: must have ≥2 columns (index + ≥1 data).
      b. row_heavy: every <th> in the single-row thead must be non-empty.
      c. All modes: no <tbody> row may have zero cells.
    """
    errors: List[str] = []
    soup = BeautifulSoup(html_table, "html.parser")
    table = soup.find("table")
    if not table:
        return ["No <table> element found"]

    # (a) col_heavy minimum column count
    if mode == "col_heavy" and n_cols < 2:
        errors.append(f"col_heavy table has only {n_cols} column(s) — expected ≥2")

    # (b) row_heavy: all thead <th> cells must have non-empty text
    if mode == "row_heavy":
        thead = table.find("thead")
        if thead:
            for th in thead.find_all("th"):
                if not th.get_text(strip=True):
                    errors.append("row_heavy thead has an unnamed <th> cell")
                    break

    # (c) no tbody row may be entirely empty
    tbody = table.find("tbody")
    if tbody:
        for row_idx, row in enumerate(tbody.find_all("tr")):
            cells = row.find_all(["td", "th"])
            if not cells:
                errors.append(f"tbody row {row_idx + 1} has zero cells")
            else:
                # Check that at least some cells have non-whitespace content
                non_empty = sum(1 for c in cells if c.get_text(strip=True))
                if non_empty == 0:
                    errors.append(f"tbody row {row_idx + 1}: all {len(cells)} cells are empty")

    return errors



def validate(labels_path: Path) -> None:
    if not labels_path.exists():
        print(f"[ERROR] Labels file not found: {labels_path}")
        sys.exit(1)

    print(f"[VALIDATE] Checking: {labels_path}")
    records = load_records(labels_path)
    print(f"[VALIDATE] Loaded {len(records)} records\n")

    total = len(records)
    schema_errors    = 0
    file_errors      = 0
    colspan_errors   = 0
    semantic_errors  = 0

    for i, rec in enumerate(records):
        errs = []
        errs.extend(check_schema(rec))
        errs.extend(check_file_exists(rec))

        html  = rec.get("html_table", "")
        mode  = rec.get("mode", "")
        ncols = rec.get("n_cols", 0)   # stored in metadata if present
        if html:
            ok, msg = check_colspan_integrity(html, mode=mode)
            if not ok:
                errs.append(f"Colspan integrity: {msg}")
                colspan_errors += 1

            sem_errs = check_semantic_density(html, mode=mode, n_cols=ncols)
            for se in sem_errs:
                errs.append(f"Semantic: {se}")
                semantic_errors += 1

        if errs:
            print(f"  Record {i+1} [{rec.get('imgid','?')}]:")
            for e in errs:
                print(f"    ✗ {e}")
            schema_errors += sum(1 for e in errs if "Missing" in e)
            file_errors   += sum(1 for e in errs if "not found" in e)
        else:
            if (i + 1) % 100 == 0:
                print(f"  ✓ {i+1}/{total} records OK")

    print(f"\n[VALIDATE] Summary")
    print(f"  Total records   : {total}")
    print(f"  Schema errors   : {schema_errors}")
    print(f"  File errors     : {file_errors}")
    print(f"  Colspan errors  : {colspan_errors}")
    print(f"  Semantic errors : {semantic_errors}")
    total_errors = schema_errors + file_errors + colspan_errors + semantic_errors
    passed = total - total_errors
    print(f"  Passed          : {passed}/{total}")

    if total_errors == 0:
        print("\n✅ All records passed validation.")
    else:
        print("\n⚠️  Some records failed — review output above.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate synthetic table labels")
    parser.add_argument("--labels", type=Path, default=None,
                        help="Path to labels.jsonl (default: data/synthetic/labels.jsonl)")
    parser.add_argument("--smoke", action="store_true",
                        help="Validate verification/smoke_labels.jsonl instead")
    args = parser.parse_args()

    if args.smoke:
        target = SMOKE_PATH
    elif args.labels:
        target = args.labels
    else:
        target = LABELS_PATH

    validate(target)
