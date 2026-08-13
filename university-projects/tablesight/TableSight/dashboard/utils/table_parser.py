"""HTML to pandas.DataFrame parser.

Converts a model-predicted HTML <table> into a rectangular DataFrame, handling
the structural cases that show up in FinTabNet: colspan expansion, rowspan
carry-down, and row-length normalization.

Returns a result dict with the parse status, alignment diagnostics, and the
DataFrame itself, so the dashboard can decide whether to surface a "needed
fixing" badge.

Lifted from the Sprint-3 parsing engine and adapted into a module so it can be
imported by the Streamlit app and by batch scripts without notebook overhead.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from bs4 import BeautifulSoup


def parse_table_with_span(html: str) -> List[List[str]]:
    """Walk an HTML <table>, expand colspan, carry rowspan, return a 2D grid."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        raise ValueError("no_table_found")

    grid: List[List[str]] = []
    rowspan_map: Dict[tuple, tuple] = {}

    for row_idx, tr in enumerate(table.find_all("tr")):
        row: List[str] = []
        col_idx = 0

        for cell in tr.find_all(["td", "th"]):
            # Fill positions already claimed by a rowspan from above
            while (row_idx, col_idx) in rowspan_map:
                value, remaining = rowspan_map.pop((row_idx, col_idx))
                row.append(value)
                if remaining > 1:
                    rowspan_map[(row_idx + 1, col_idx)] = (value, remaining - 1)
                col_idx += 1

            text = cell.get_text(" ", strip=True)
            try:
                colspan = int(cell.get("colspan", 1))
            except (TypeError, ValueError):
                colspan = 1
            try:
                rowspan = int(cell.get("rowspan", 1))
            except (TypeError, ValueError):
                rowspan = 1

            for offset in range(colspan):
                row.append(text)
                if rowspan > 1:
                    rowspan_map[(row_idx + 1, col_idx + offset)] = (text, rowspan - 1)
            col_idx += colspan

        # Flush any rowspan-carried cells that fall after this row's last <td>
        while (row_idx, col_idx) in rowspan_map:
            value, remaining = rowspan_map.pop((row_idx, col_idx))
            row.append(value)
            if remaining > 1:
                rowspan_map[(row_idx + 1, col_idx)] = (value, remaining - 1)
            col_idx += 1

        grid.append(row)

    if not grid:
        raise ValueError("empty_table")
    return grid


def normalize_rows(rows: List[List[str]], fill_value: str = "") -> List[List[str]]:
    max_len = max(len(row) for row in rows)
    return [row + [fill_value] * (max_len - len(row)) for row in rows]


def check_row_alignment(rows: List[List[str]]) -> Dict[str, Any]:
    row_lengths = [len(row) for row in rows]
    return {
        "min_cols":       min(row_lengths),
        "max_cols":       max(row_lengths),
        "unique_lengths": sorted(set(row_lengths)),
        "is_aligned":     len(set(row_lengths)) == 1,
    }


def robust_html_to_dataframe(html: str) -> Dict[str, Any]:
    """Parse HTML to a DataFrame with full diagnostics.

    Status values:
        success_clean              all rows had matching widths
        success_fixed_misalignment rows were padded to the max width
        failed                     no table found / parse error (df=None)
    """
    try:
        rows = parse_table_with_span(html)
        info = check_row_alignment(rows)

        if info["is_aligned"]:
            status, normalized = "success_clean", False
        else:
            rows = normalize_rows(rows)
            status, normalized = "success_fixed_misalignment", True

        df = pd.DataFrame(rows)
        return {
            "status":                status,
            "error_type":            "",
            "error_message":         "",
            "normalized":            normalized,
            "min_cols_before":       info["min_cols"],
            "max_cols_before":       info["max_cols"],
            "unique_lengths_before": info["unique_lengths"],
            "num_rows":              df.shape[0],
            "num_cols":              df.shape[1],
            "df":                    df,
        }
    except Exception as exc:
        return {
            "status":                "failed",
            "error_type":            type(exc).__name__,
            "error_message":         str(exc),
            "normalized":            False,
            "min_cols_before":       None,
            "max_cols_before":       None,
            "unique_lengths_before": None,
            "num_rows":              None,
            "num_cols":              None,
            "df":                    None,
        }


__all__ = [
    "parse_table_with_span",
    "normalize_rows",
    "check_row_alignment",
    "robust_html_to_dataframe",
]
