"""SPARTAN runner — OCR-first hierarchical-table extractor.

This is the production version of the pipeline that scored mean TEDS 0.539
on the FinTabNet test split (with tuned `ocr_min_conf=0.30` and
`row_tol_frac=0.80`).

Pipeline:
    1. Preprocess  (normalise to target_width, optional CLAHE + deskew)
    2. EasyOCR over the full image → word boxes with text + confidence
    3. Cluster word y-centers into rows (tol = row_tol_frac × median word height)
    4. 1D KMeans on x-centers → column centers (n_cols inferred from modal row width)
    5. Place each word into its nearest (row, col) cell
    6. Detect header rows (top contiguous no-digit rows)
    7. Detect colspan (word bbox straddling >= 2 column centers)
    8. Emit <table><thead><tr><th>…<tbody><tr><td>… HTML

The runner exposes the legacy class name `SpartanRunner` for backwards
compatibility with the dashboard's inference layer (the UI labels it "SPARTAN").

Tuned parameters are loaded at __init__ time from:
    <this_dir>/preprocessing_params.json
    <this_dir>/grid_params.json
both produced by `train_spartan.py`. If either file is
missing, sensible defaults are used (matching the notebook's "baseline" row).
"""
from __future__ import annotations

import io
import json
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


# ── default params (match the notebook's baseline) ──────────────────────────

DEFAULT_PRE = {
    "target_width": 1600,
    "clahe": False,
    "deskew": False,
}
DEFAULT_GRID = {
    "ocr_min_conf":     0.20,
    "row_tol_frac":     0.55,
    "header_max_rows":  3,
    "do_colspan_merge": True,
}


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _deskew(gray: np.ndarray, max_angle: float = 8.0) -> np.ndarray:
    import cv2
    edges = cv2.Canny(gray, 50, 150)
    coords = np.column_stack(np.where(edges > 0))
    if coords.shape[0] < 30:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 1.0 or abs(angle) > max_angle:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _preprocess(pil: Image.Image, params: Dict[str, Any]) -> np.ndarray:
    import cv2
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    rgb = np.array(pil)
    h, w = rgb.shape[:2]
    target_w = int(params.get("target_width", 1600))
    if w != target_w:
        scale = target_w / w
        rgb = cv2.resize(rgb, (target_w, int(round(h * scale))),
                          interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if params.get("clahe"):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    if params.get("deskew"):
        gray = _deskew(gray)
    # Return the COLOR image for OCR (EasyOCR likes RGB) at the same resolution
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def _words_from_ocr(image_arr: np.ndarray, ocr_reader, min_conf: float
                     ) -> List[Dict[str, Any]]:
    try:
        raw = ocr_reader.readtext(image_arr, detail=1, paragraph=False)
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for box, txt, conf in raw:
        if conf < min_conf or not txt or not txt.strip():
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        out.append({
            "x":  float(min(xs)),
            "y":  float(min(ys)),
            "w":  float(max(xs) - min(xs)),
            "h":  float(max(ys) - min(ys)),
            "cx": float((min(xs) + max(xs)) / 2),
            "cy": float((min(ys) + max(ys)) / 2),
            "text": txt.strip(),
            "conf": float(conf),
        })
    return out


def _cluster_rows(words: List[Dict[str, Any]],
                   tol_frac: float = 0.55) -> List[List[Dict[str, Any]]]:
    if not words:
        return []
    median_h = float(np.median([w["h"] for w in words]))
    tol = max(6.0, median_h * tol_frac)
    sorted_w = sorted(words, key=lambda w: w["cy"])
    rows: List[List[Dict[str, Any]]] = [[sorted_w[0]]]
    cur_cy = sorted_w[0]["cy"]
    for w in sorted_w[1:]:
        if w["cy"] - cur_cy <= tol:
            rows[-1].append(w)
            cur_cy = (cur_cy * (len(rows[-1]) - 1) + w["cy"]) / len(rows[-1])
        else:
            rows.append([w])
            cur_cy = w["cy"]
    for r in rows:
        r.sort(key=lambda w: w["cx"])
    return rows


def _estimate_n_cols(rows: List[List[Dict[str, Any]]]) -> int:
    if not rows:
        return 1
    counts = [len(r) for r in rows]
    if not counts:
        return 1
    p60 = float(np.percentile(counts, 60))
    big = [c for c in counts if c >= p60]
    return Counter(big).most_common(1)[0][0] if big else max(counts)


def _column_centers(rows: List[List[Dict[str, Any]]], n_cols: int) -> List[float]:
    if n_cols <= 1:
        return [0.0]
    seed_row = next((r for r in rows if len(r) == n_cols), None)
    if seed_row is None:
        seed_row = max(rows, key=lambda r: len(r))
    if len(seed_row) >= n_cols:
        seeds = sorted([w["cx"] for w in seed_row[:n_cols]])
    else:
        all_cx = sorted([w["cx"] for r in rows for w in r])
        seeds = list(np.linspace(min(all_cx), max(all_cx), n_cols))
    all_cx = sorted([w["cx"] for r in rows for w in r])
    for _ in range(5):
        new_sum = [0.0] * n_cols
        new_n = [0] * n_cols
        for x in all_cx:
            j = min(range(n_cols), key=lambda i: abs(seeds[i] - x))
            new_sum[j] += x
            new_n[j] += 1
        seeds = [new_sum[i] / new_n[i] if new_n[i] else seeds[i] for i in range(n_cols)]
        seeds.sort()
    return seeds


def _assign_to_grid(rows: List[List[Dict[str, Any]]],
                     col_centers: List[float]) -> List[List[str]]:
    n_rows = len(rows)
    n_cols = len(col_centers)
    grid = [[""] * n_cols for _ in range(n_rows)]
    for r_i, row_words in enumerate(rows):
        for w in row_words:
            c_i = min(range(n_cols), key=lambda i: abs(col_centers[i] - w["cx"]))
            grid[r_i][c_i] = (grid[r_i][c_i] + " " + w["text"]).strip()
    return grid


def _detect_header_rows(grid: List[List[str]], max_rows: int = 3) -> int:
    if not grid:
        return 1
    n = 0
    for row in grid:
        non_empty = [c for c in row if c.strip()]
        if non_empty and not any(any(ch.isdigit() for ch in c) for c in non_empty):
            n += 1
        else:
            break
    return max(1, min(n, max_rows))


def _detect_spans(rows: List[List[Dict[str, Any]]], col_centers: List[float]
                   ) -> Dict[Tuple[int, int], int]:
    spans: Dict[Tuple[int, int], int] = {}
    n_cols = len(col_centers)
    for r_i, row_words in enumerate(rows):
        for w in row_words:
            covered = [i for i in range(n_cols)
                       if w["x"] - 4 <= col_centers[i] <= w["x"] + w["w"] + 4]
            if len(covered) >= 2:
                spans[(r_i, covered[0])] = len(covered)
    return spans


def _grid_to_html(grid: List[List[str]], header_rows: int,
                   spans: Optional[Dict[Tuple[int, int], int]] = None) -> str:
    if not grid or not grid[0]:
        return "<table></table>"
    spans = spans or {}
    absorbed = set()
    for (r, c), cs in spans.items():
        for k in range(1, cs):
            absorbed.add((r, c + k))
    n_rows, n_cols = len(grid), len(grid[0])

    thead_parts: List[str] = []
    for r in range(header_rows):
        cells: List[str] = []
        for c in range(n_cols):
            if (r, c) in absorbed:
                continue
            cs = spans.get((r, c), 1)
            attr = f' colspan="{cs}"' if cs > 1 else ""
            cells.append(f"<th{attr}>{grid[r][c]}</th>")
        thead_parts.append("<tr>" + "".join(cells) + "</tr>")
    thead = "<thead>" + "".join(thead_parts) + "</thead>" if thead_parts else ""

    tbody_parts: List[str] = []
    for r in range(header_rows, n_rows):
        cells = []
        for c in range(n_cols):
            if (r, c) in absorbed:
                continue
            cs = spans.get((r, c), 1)
            attr = f' colspan="{cs}"' if cs > 1 else ""
            cells.append(f"<td{attr}>{grid[r][c]}</td>")
        tbody_parts.append("<tr>" + "".join(cells) + "</tr>")
    tbody = "<tbody>" + "".join(tbody_parts) + "</tbody>" if tbody_parts else ""

    return f"<table>{thead}{tbody}</table>"


# ──────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────


class SpartanRunner:
    """SPARTAN pipeline (legacy class name kept for the dashboard inference layer).

    Loads tuned params from `<this_dir>/preprocessing_params.json` and
    `<this_dir>/grid_params.json` at init. Falls back to baseline defaults if
    missing.
    """

    def __init__(self, checkpoint_path: Optional[str] = None,
                  device: str = "cpu", **_: Any):
        self.device = device
        here = Path(__file__).parent

        pre_path  = here / "preprocessing_params.json"
        grid_path = here / "grid_params.json"

        self.pre_params = dict(DEFAULT_PRE)
        if pre_path.exists():
            try:
                loaded = json.loads(pre_path.read_text())
                self.pre_params.update(loaded.get("global", {}))
            except Exception as exc:
                warnings.warn(f"SPARTAN: failed to load {pre_path.name}: {exc}")

        self.grid_params = dict(DEFAULT_GRID)
        if grid_path.exists():
            try:
                loaded = json.loads(grid_path.read_text())
                self.grid_params.update(loaded.get("global", {}))
            except Exception as exc:
                warnings.warn(f"SPARTAN: failed to load {grid_path.name}: {exc}")

        # Lazy OCR reader — initialised on first predict()
        self._ocr_reader = None

    # ── inference ─────────────────────────────────────────────────────────

    def _ensure_reader(self):
        if self._ocr_reader is not None:
            return
        try:
            import easyocr
            try:
                import torch
                gpu = torch.cuda.is_available()
            except Exception:
                gpu = False
            self._ocr_reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
        except Exception as exc:
            raise RuntimeError(f"SPARTAN: EasyOCR initialisation failed: {exc}") from exc

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        self._ensure_reader()
        timings: Dict[str, float] = {}

        t0 = time.perf_counter()
        color = _preprocess(image, self.pre_params)
        timings["preprocess_s"] = round(time.perf_counter() - t0, 3)

        t0 = time.perf_counter()
        words = _words_from_ocr(color, self._ocr_reader,
                                  min_conf=self.grid_params.get("ocr_min_conf", 0.20))
        timings["ocr_s"] = round(time.perf_counter() - t0, 3)
        if not words:
            return {"html": "<table></table>",
                    "metadata": {"reason": "no_ocr_output", "n_cells": 0,
                                 "timings_s": timings, "params": self._meta_params()}}

        t0 = time.perf_counter()
        rows = _cluster_rows(words,
                              tol_frac=self.grid_params.get("row_tol_frac", 0.55))
        n_cols = _estimate_n_cols(rows)
        col_centers = _column_centers(rows, n_cols)
        grid = _assign_to_grid(rows, col_centers)
        header_rows = _detect_header_rows(grid,
                                            max_rows=int(self.grid_params.get("header_max_rows", 3)))
        spans = _detect_spans(rows, col_centers) if self.grid_params.get("do_colspan_merge", True) else {}
        timings["assembly_s"] = round(time.perf_counter() - t0, 3)

        html = _grid_to_html(grid, header_rows=header_rows, spans=spans)

        n_cells = sum(1 for r in grid for c in r if c.strip())
        return {
            "html": html,
            "metadata": {
                "n_words":     len(words),
                "n_rows":      len(grid),
                "n_cols":      n_cols,
                "n_cells":     n_cells,
                "header_rows": header_rows,
                "n_spans":     len(spans),
                "table_type":  "nested_column" if header_rows >= 2 else "normal",
                "timings_s":   timings,
                "params":      self._meta_params(),
            },
        }

    # ── introspection ────────────────────────────────────────────────────

    def _meta_params(self) -> Dict[str, Any]:
        return {"pre": dict(self.pre_params), "grid": dict(self.grid_params)}


__all__ = ["SpartanRunner"]
