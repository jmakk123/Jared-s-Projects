"""SPARTAN — Hierarchical Table Extractor (CLI script).

Converted from the original Colab notebook.  Each `# %%` block is a
notebook cell; open this file in VS Code or Jupyter for a notebook-
style view, or run end-to-end with `python train_spartan.py`.

CLI flags (when run as a script):
    --benchmark         run inference + TEDS on the test split
    --train             coordinate-descent tune SPARTAN params
    --image PATH        single-image inference
"""

# %% [markdown]
# # SPARTAN — Hierarchical Table Extractor
#
# **UChicago MS-ADS · Spring 2026**
#
# This notebook implements the [SPARTAN](https://arxiv.org/abs/2306.06942)-inspired
# heuristic pipeline as a complementary extractor to the Florence-2 + LoRA
# fine-tuned model. It is the **OCR-based baseline** that replaces the pytesseract
# pipeline (which scored poorly on borderless and multi-header FinTabNet tables).
#
# | | |
# |---|---|
# | **Target** | TEDS ≥ 0.50 on the FinTabNet test split |
# | **Florence-2 zero-shot reference** | ~0.15 mean TEDS (sprint 1) |
# | **OCR** | EasyOCR (CRAFT + CRNN) — drop-in replacement for pytesseract |
# | **Output schema** | Matches sprint-2 ablation format for direct comparison |
#
# ### Pipeline
# ```
# PNG → preprocess → column detect → region segment → table parse → OCR + cells → HTML
#                                                                             ↓
#                                                                    compute TEDS vs GT
# ```

# %% [markdown]
# ## 1. Install dependencies
#
# EasyOCR (CRAFT detector + CRNN recognizer) is the OCR engine. The cell is idempotent — installed packages are skipped, missing ones are filled in. On Colab the Tesseract binary is also installed as a backup fallback.

# %%
import importlib, sys, subprocess

REQUIRED = {
    'cv2':           'opencv-python-headless',
    'PIL':           'Pillow',
    'pandas':        'pandas',
    'numpy':         'numpy',
    'lxml':          'lxml',
    'apted':         'apted',
    'tqdm':          'tqdm',
    'matplotlib':    'matplotlib',
    'easyocr':       'easyocr',
    'datasets':      'datasets',
    'torch':         'torch',
}

def _need(mod, pkg):
    try:
        importlib.import_module(mod)
        return False
    except Exception:
        return True

missing = [(mod, pkg) for mod, pkg in REQUIRED.items() if _need(mod, pkg)]
if missing:
    print(f'Installing {len(missing)} missing packages…')
    pkgs = sorted({pkg for _, pkg in missing})
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *pkgs])
else:
    print('All required packages already installed.')

# Optional: Tesseract binary (only used by some fallbacks). Try apt-get on Colab; skip elsewhere.
try:
    if 'google.colab' in sys.modules:
        subprocess.run(['apt-get', 'install', '-q', '-y', 'tesseract-ocr', 'tesseract-ocr-eng'],
                       capture_output=True, check=False)
except Exception:
    pass

import easyocr, torch
print(f'easyocr={easyocr.__version__}  torch={torch.__version__}  '
      f'cuda={torch.cuda.is_available()}')

# %% [markdown]
# ## 2. Setup paths & bootstrap data
#
# **No Google Drive required.** The cell auto-detects whether you're on Colab or running locally, picks sane paths, and downloads FinTabNet from HuggingFace (`apoidea/fintabnet-html`) if no local splits CSV exists. PNGs and the splits CSV land at:
#
# * **Local**: `<repo>/data/processed_local/`
# * **Colab**: `/content/spartan_workspace/`
#
# Trained-parameter outputs write to `<repo>/TableSight/models/jared/` when running locally (so the dashboard picks them up immediately), or to a flat Colab folder that's zipped + downloaded at the end.

# %%
"""Environment-aware setup: works on Colab AND locally without Drive.

Decision tree:
    1. Detect ENV (colab vs local).
    2. Build a CACHE_DIR that lives on local disk (Colab: /content/data, local: ./outputs/spartan).
    3. Build a splits CSV from `apoidea/fintabnet-html` (HF Hub) if one isn't already on disk.
       Filter to hierarchical tables (colspan>1 or rowspan>1) and write per-image PNGs.
    4. Set the output paths so trained artifacts (preprocessing_params.json,
       grid_params.json, benchmark CSV) land somewhere the dashboard can read.
"""
from pathlib import Path
import os, sys, json, re, hashlib

IS_COLAB = 'google.colab' in sys.modules
print(f'Environment: {"COLAB" if IS_COLAB else "LOCAL"}')

# ── 1. Choose paths ──────────────────────────────────────────────────
if IS_COLAB:
    PROJECT_ROOT = Path('/content/spartan_workspace')
else:
    # local: assume notebook lives in TableSight/notebooks/ inside the repo
    here = Path('.').resolve()
    candidates = [here, here.parent, here.parent.parent]
    PROJECT_ROOT = next((p for p in candidates if (p / 'TableSight').exists()), here)
DATA_ROOT      = PROJECT_ROOT / ('data' if not IS_COLAB else '')
IMAGES_DIR     = (PROJECT_ROOT / 'data' / 'processed_local' / 'images') if not IS_COLAB else (PROJECT_ROOT / 'images')
SPLITS_CSV     = (PROJECT_ROOT / 'data' / 'processed_local' / 'training_fintabnet_pool_splits_local.csv') if not IS_COLAB else (PROJECT_ROOT / 'splits.csv')

# Output paths: when running locally, write straight into TableSight/models/jared so the
# dashboard picks them up automatically. In Colab we write to a flat folder + zip at the end.
if not IS_COLAB and (PROJECT_ROOT / 'TableSight' / 'models' / 'jared').exists():
    SPARTAN_OUT = PROJECT_ROOT / 'TableSight' / 'models' / 'jared'
else:
    SPARTAN_OUT = PROJECT_ROOT / 'outputs' / 'spartan'

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
SPARTAN_OUT.mkdir(parents=True, exist_ok=True)
SPLITS_CSV.parent.mkdir(parents=True, exist_ok=True)

print(f'PROJECT_ROOT : {PROJECT_ROOT}')
print(f'IMAGES_DIR   : {IMAGES_DIR}')
print(f'SPLITS_CSV   : {SPLITS_CSV}')
print(f'SPARTAN_OUT  : {SPARTAN_OUT}')

# ── 2. Bootstrap data from HuggingFace if the local splits CSV is missing ─
N_SAMPLES = 1300                  # match the 1300-sample pool
HARD_ONLY = True                  # keep only tables with colspan>1 OR rowspan>1


def _has_span(html_str):
    if not html_str: return False
    return bool(re.search(r'(?:col|row)span\s*=\s*["\']?\s*(?:[2-9]|\d{2,})', html_str))


def _image_type(html_str):
    if not html_str: return 'simple'
    max_cs = max((int(m.group(1)) for m in re.finditer(r'colspan\s*=\s*["\']?(\d+)', html_str)), default=1)
    max_rs = max((int(m.group(1)) for m in re.finditer(r'rowspan\s*=\s*["\']?(\d+)', html_str)), default=1)
    n_th  = len(re.findall(r'<th[\s>]', html_str.lower()))
    if max_cs > 3 or max_rs > 3 or n_th >= 12: return 'extreme'
    if max_cs > 1 or max_rs > 1 or n_th >= 6:  return 'medium'
    return 'simple'


def _bootstrap_from_hf(target_csv: Path, images_dir: Path,
                       n_samples: int = N_SAMPLES, hard_only: bool = HARD_ONLY):
    import io, pandas as pd
    from PIL import Image as PILImage
    from datasets import load_dataset

    print(f'\nBootstrapping FinTabNet from HuggingFace (apoidea/fintabnet-html)…')
    ds = load_dataset('apoidea/fintabnet-html', 'en', split='train')
    print(f'  loaded {len(ds):,} HF rows  →  filtering to hierarchical, n={n_samples}')

    kept, seen = [], 0
    for row in ds:
        seen += 1
        html = row.get('html') or row.get('html_table') or ''
        if hard_only and not _has_span(html):
            continue
        kept.append((seen, row, html))
        if len(kept) >= n_samples: break

    print(f'  kept {len(kept):,} of {seen:,} scanned rows')

    # Materialise images + build CSV
    rows = []
    for i, (seq, row, html) in enumerate(kept):
        img_id = f'fintab_{i:06d}'
        img_path = images_dir / f'{img_id}.png'
        if not img_path.exists():
            img = row.get('image')
            if img is None: continue
            if not isinstance(img, PILImage.Image):
                try: img = PILImage.open(io.BytesIO(img))
                except Exception: continue
            img.convert('RGB').save(img_path, format='PNG')
        rows.append({
            'image_id':   img_id,
            'img_path':   str(img_path),
            'html':       html,
            'image_type': _image_type(html),
        })
    df = pd.DataFrame(rows)
    # deterministic train/val/test split
    rng = pd.Series(df.image_id).apply(lambda s: int(hashlib.md5(s.encode()).hexdigest(), 16) % 1000)
    df = df.assign(_hash=rng).sort_values('_hash').reset_index(drop=True).drop(columns='_hash')
    n = len(df); n_tr = int(n * 0.75); n_va = int(n * 0.12)
    df['split'] = 'test'
    df.loc[:n_tr - 1, 'split'] = 'train'
    df.loc[n_tr:n_tr + n_va - 1, 'split'] = 'val'
    df['phase'] = 'phase1'
    df.to_csv(target_csv, index=False)
    return df


import pandas as pd

if SPLITS_CSV.exists():
    splits = pd.read_csv(SPLITS_CSV)
    # confirm images exist (handle stale paths)
    needs_image = ~splits['img_path'].apply(lambda p: Path(p).exists())
    if needs_image.any():
        print(f'⚠ {needs_image.sum()} rows in CSV have missing PNGs — rebuilding from HF…')
        splits = _bootstrap_from_hf(SPLITS_CSV, IMAGES_DIR)
else:
    splits = _bootstrap_from_hf(SPLITS_CSV, IMAGES_DIR)

print(f'\n── inventory ────────────────────────────────────────────')
print(f'  splits.csv         : {SPLITS_CSV}  rows={len(splits):,}')
print(f'  split breakdown    : {splits.split.value_counts().to_dict()}')
print(f'  image_type distrib : {splits.image_type.value_counts().to_dict()}')
print(f'  png count on disk  : {sum(1 for _ in IMAGES_DIR.glob("*.png")):,}')
print(f'  output dir         : {SPARTAN_OUT}')

# %% [markdown]
# ## 3. Load the test split
#
# Selects the 13% slice held out for evaluation (deterministic split via MD5 hash of `image_id` — same images across re-runs).

# %%
"""Pick the test split — what the benchmark loop runs against.

Normalises column names so the notebook works whether the splits CSV came
from the HF bootstrap (uses `image_id`) or from prepare_local_data.py
(uses `img_id`).
"""
# Harmonise schemas — accept either image_id or img_id
if 'image_id' not in splits.columns and 'img_id' in splits.columns:
    splits = splits.rename(columns={'img_id': 'image_id'})
elif 'img_id' not in splits.columns and 'image_id' in splits.columns:
    splits['img_id'] = splits['image_id']

test_df = splits[splits['split'] == 'test'].copy().reset_index(drop=True)
test_df['exists'] = test_df['img_path'].apply(lambda p: Path(p).exists())
print(f'Test rows                : {len(test_df):,}')
print(f'Test rows with image     : {test_df.exists.sum():,}')
print(f'\nimage_type distribution:')
print(test_df.image_type.value_counts().to_string())
# alias so the existing code below (which references img_path_resolved) keeps working
test_df['img_path_resolved'] = test_df['img_path']

# %% [markdown]
# ## 4. TEDS scorer
#
# Identical to the sprint-2 implementation so results are directly comparable. Supports `structure_only=True` for skeleton-vs-skeleton comparisons (matches Florence-2 phase-1 eval).

# %%
from apted import APTED, Config
from lxml import html as lhtml
from collections import deque

class TableTree:
    __slots__ = ('tag', 'children', 'colspan', 'rowspan', 'content')
    def __init__(self, tag='', content='', colspan=1, rowspan=1, children=None):
        self.tag = tag
        self.content = content
        self.colspan = colspan
        self.rowspan = rowspan
        self.children = children or []

class _TEDSConfig(Config):
    def __init__(self, structure_only: bool = False):
        self.structure_only = structure_only
    def rename(self, n1: TableTree, n2: TableTree) -> float:
        if n1.tag != n2.tag:
            return 1.0
        if n1.colspan != n2.colspan or n1.rowspan != n2.rowspan:
            return 1.0
        if not self.structure_only and n1.tag in ('td','th'):
            if n1.content.strip() != n2.content.strip():
                return 1.0
        return 0.0
    def children(self, node):
        return node.children

def _parse_el(el) -> TableTree:
    node = TableTree(
        tag=el.tag,
        content=(el.text or '').strip(),
        colspan=int(el.get('colspan', 1) or 1),
        rowspan=int(el.get('rowspan', 1) or 1),
    )
    for ch in el:
        node.children.append(_parse_el(ch))
    return node

def html_to_tree(html_str: str) -> TableTree:
    if not html_str or '<table' not in html_str:
        return TableTree(tag='table')
    try:
        root = lhtml.fragment_fromstring(html_str, create_parent='div')
        table = root.find('.//table')
        if table is None:
            return TableTree(tag='table')
        return _parse_el(table)
    except Exception:
        return TableTree(tag='table')

def _tree_size(t: TableTree) -> int:
    return 1 + sum(_tree_size(c) for c in t.children)

def teds_score(pred_html: str, true_html: str, structure_only: bool = False) -> float:
    pred = html_to_tree(pred_html or '')
    true = html_to_tree(true_html or '')
    cfg  = _TEDSConfig(structure_only=structure_only)
    distance = APTED(pred, true, cfg).compute_edit_distance()
    denom = max(_tree_size(pred), _tree_size(true))
    if denom == 0: return 0.0
    return max(0.0, 1.0 - distance / denom)

# Smoke test
pred = '<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>'
true = '<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>'
print(f'identical → TEDS = {teds_score(pred, true):.3f}')
mismatch = '<table><tr><th>A</th><th>X</th></tr><tr><td>1</td><td>2</td></tr></table>'
print(f'one cell diff → TEDS = {teds_score(pred, mismatch):.3f}')
print(f'structure-only ignores cell diff → TEDS-S = {teds_score(pred, mismatch, structure_only=True):.3f}')

# %% [markdown]
# ## 5. Pre-processing module
#
# Follows SPARTAN's pre-processing recipe step-by-step. Returns both a binary image (for grid detection) and a normalised color image (for OCR).

# %%
import cv2
import numpy as np
from PIL import Image as PILImage

def preprocess_image(img_path, target_width: int = 1600,
                     adaptive_block: int = 15, adaptive_C: int = 10):
    '''Load image and apply SPARTAN's preprocessing pipeline.

    Returns
    -------
    (binary_img, color_img) — both np.ndarray at the normalised target width.
    '''
    pil = PILImage.open(str(img_path))
    if pil.mode != 'RGB':
        pil = pil.convert('RGB')
    rgb = np.array(pil)                                       # (H, W, 3)

    # Normalise width to target_width preserving aspect
    h, w = rgb.shape[:2]
    if w != target_width:
        scale = target_width / w
        new_h = int(round(h * scale))
        rgb = cv2.resize(rgb, (target_width, new_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        adaptive_block, adaptive_C,
    )
    return binary, rgb

# %% [markdown]
# ## 6. Column detection module
#
# Whitespace-based vertical column detector — only fires for landscape pages (aspect ratio > 1.5) where multi-column layout is plausible. Returns a list of `(x_start, x_end)` segments.

# %%
def detect_columns(binary_img: np.ndarray, min_gap_height: int = 200,
                   text_margin: int = 15, merge_x_tolerance: int = 15
                   ) -> 'list[tuple[int,int]]':
    h, w = binary_img.shape
    if w / max(1, h) <= 1.5:
        return [(0, w)]   # portrait → assume single column

    # binary_img is 0/255; columns of all-255 ARE whitespace
    col_is_white = (binary_img == 255).all(axis=0)
    # Find runs of whitespace columns
    runs = []
    in_run = False; rs = 0
    for i, b in enumerate(col_is_white):
        if b and not in_run: in_run, rs = True, i
        elif not b and in_run: in_run = False; runs.append((rs, i))
    if in_run: runs.append((rs, w))

    # Keep gaps with sufficient vertical "tallness" and text on both sides
    text_col = ~col_is_white
    keep = []
    for (a, b) in runs:
        if (b - a) < 4: continue
        # vertical extent of darkness: look at the central window
        # Approximate gap_height by counting columns that ARE whitespace in this run.
        # (Geometric tallness — already implied by all-255 column membership.)
        gap_height = h * (b - a)            # proportional proxy
        if gap_height < min_gap_height * (b - a):  # weak filter; we'll let length carry
            pass
        # Require text within text_margin px on both sides
        left  = max(0, a - text_margin); right = min(w, b + text_margin)
        if text_col[:, left:a].any() and text_col[:, b:right].any():
            keep.append((a, b))
    if not keep:
        return [(0, w)]
    # Merge same-x neighbours
    keep.sort()
    merged = [keep[0]]
    for (a, b) in keep[1:]:
        pa, pb = merged[-1]
        if a - pb <= merge_x_tolerance:
            merged[-1] = (pa, b)
        else:
            merged.append((a, b))
    # Convert white-gap centres → column-band boundaries
    bands = []
    cur = 0
    for (a, b) in merged:
        bands.append((cur, a))
        cur = b
    bands.append((cur, w))
    bands = [(a, b) for (a, b) in bands if (b - a) >= 60]
    return bands or [(0, w)]

# %% [markdown]
# ## 7. Region segmentation module
#
# Three-strategy table detection (SPARTAN's core):
#
# * **A** — boundary-based (ruling-line tables, morphological open with long kernels)
# * **B** — text-based (borderless tables — the dominant FinTabNet case)
# * **C** — unify and de-duplicate via IoU

# %%
def _iou(a, b) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1: return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / max(1, union)


def _crop_outer_boundary(binary: np.ndarray, color: np.ndarray):
    page_area = binary.shape[0] * binary.shape[1]
    cnts, _ = cv2.findContours(255 - binary, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return binary, color, (0, 0, binary.shape[1], binary.shape[0])
    best = None
    for c in cnts:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.contourArea(c) >= 0.75 * page_area:
            best = cv2.boundingRect(approx); break
    if best is None:
        return binary, color, (0, 0, binary.shape[1], binary.shape[0])
    x, y, w, h = best
    pad = 25
    x1 = max(0, x - pad); y1 = max(0, y - pad)
    x2 = min(binary.shape[1], x + w + pad); y2 = min(binary.shape[0], y + h + pad)
    return binary[y1:y2, x1:x2], color[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)


def _boundary_based_cells(binary: np.ndarray):
    '''Return a list of cell bboxes detected via morphological grid extraction.'''
    inv = 255 - binary
    best_cells = []
    for dilate in (False, True):
        bin_use = inv.copy()
        if dilate:
            bin_use = cv2.dilate(bin_use, np.ones((3, 3), np.uint8), iterations=1)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        h_lines = cv2.morphologyEx(bin_use, cv2.MORPH_OPEN, h_kernel, iterations=2)
        v_lines = cv2.morphologyEx(bin_use, cv2.MORPH_OPEN, v_kernel, iterations=2)
        mask = cv2.addWeighted(h_lines, 0.5, v_lines, 0.5, 0.0)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cells = []
        for c in cnts:
            approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(approx) != 4: continue
            x, y, w, h = cv2.boundingRect(approx)
            if w > 100 and h > 20:
                cells.append((x, y, w, h))
        if len(cells) > len(best_cells):
            best_cells = cells
    return best_cells


def _text_based_table(color_img: np.ndarray, ocr_reader):
    '''Detect borderless tables via aligned-column word stacks.

    Returns the bbox of the largest "tabular" arrangement OR None.
    Also returns a pseudo-bordered image (synthetic grid drawn over the layout).
    '''
    res = ocr_reader.readtext(color_img, detail=1, paragraph=False)
    if not res: return None, None
    # Each entry: ([(x1,y1),(x2,y1),(x2,y2),(x1,y2)], text, conf)
    boxes = []
    for box, txt, conf in res:
        if conf < 0.20 or not txt.strip(): continue
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        boxes.append((min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys), txt))
    if len(boxes) < 6: return None, None

    boxes.sort(key=lambda b: (b[1], b[0]))
    # Cluster column candidates by left-edge alignment within 5px tolerance
    lefts = sorted({b[0] for b in boxes})
    column_groups = []
    cur = [lefts[0]]
    for x in lefts[1:]:
        if x - cur[-1] <= 5: cur.append(x)
        else:
            column_groups.append(cur); cur = [x]
    column_groups.append(cur)
    # Need at least 2 columns with multiple stacked words
    column_groups = [g for g in column_groups
                     if sum(1 for b in boxes
                            if any(abs(b[0] - x) <= 5 for x in g)) >= 2]
    if len(column_groups) < 2:
        return None, None

    xs_all = [b[0] for b in boxes]
    ys_all = [b[1] for b in boxes]
    ws_all = [b[0] + b[2] for b in boxes]
    hs_all = [b[1] + b[3] for b in boxes]
    bbox = (min(xs_all), min(ys_all),
            max(ws_all) - min(xs_all),
            max(hs_all) - min(ys_all))

    # Build pseudo-bordered image — draw rectangles around each word
    pseudo = color_img.copy()
    # Column lines
    col_xs = sorted({min(g) for g in column_groups})
    for x in col_xs:
        cv2.line(pseudo, (x - 2, bbox[1]), (x - 2, bbox[1] + bbox[3]),
                 (0, 0, 0), 1)
    # Row lines: cluster y-tops by proximity
    y_lines = sorted(set(b[1] for b in boxes))
    cur_y = [y_lines[0]]
    rows_y = []
    for y in y_lines[1:]:
        if y - cur_y[-1] <= 5: cur_y.append(y)
        else: rows_y.append(int(np.mean(cur_y))); cur_y = [y]
    rows_y.append(int(np.mean(cur_y)))
    for y in rows_y:
        cv2.line(pseudo, (bbox[0], y - 2),
                 (bbox[0] + bbox[2], y - 2), (0, 0, 0), 1)
    return bbox, pseudo


def segment_table_regions(binary_img: np.ndarray, color_img: np.ndarray,
                          ocr_reader) -> list:
    '''Run boundary + text detectors, unify by IoU, return non-overlapping regions.'''
    bin_crop, col_crop, _ = _crop_outer_boundary(binary_img, color_img)
    regions = []

    # Strategy A — boundary
    cells_b = _boundary_based_cells(bin_crop)
    if cells_b:
        xs = [c[0] for c in cells_b]; ys = [c[1] for c in cells_b]
        rs = [c[0] + c[2] for c in cells_b]; bs = [c[1] + c[3] for c in cells_b]
        bb = (min(xs), min(ys), max(rs) - min(xs), max(bs) - min(ys))
        if bb[2] >= 500 and bb[3] >= 100:
            regions.append({'bbox': bb, 'type': 'bordered',
                            'pseudo_bordered_img': col_crop[bb[1]:bb[1]+bb[3],
                                                            bb[0]:bb[0]+bb[2]]})

    # Strategy B — text-based
    text_bb, pseudo = _text_based_table(col_crop, ocr_reader)
    if text_bb is not None and pseudo is not None:
        # IoU vs bordered region
        if regions and _iou(text_bb, regions[0]['bbox']) > 0.5:
            pass   # subsumed
        else:
            x, y, w, h = text_bb
            regions.append({'bbox': text_bb, 'type': 'borderless',
                            'pseudo_bordered_img': pseudo[y:y+h, x:x+w]})

    if not regions:
        H, W = col_crop.shape[:2]
        regions = [{'bbox': (0, 0, W, H), 'type': 'borderless',
                    'pseudo_bordered_img': col_crop}]
    return regions

# %% [markdown]
# ## 8. Table parser module
#
# CV LSD-based grid reconstruction. Returns cells + spans + header-row count + table-type heuristic (`normal` / `key_value` / `nested_column`).

# %%
def _merge_close(values, tol=5):
    if not values: return []
    s = sorted(values)
    out = [s[0]]
    for v in s[1:]:
        if v - out[-1] <= tol: out[-1] = (out[-1] + v) // 2
        else: out.append(v)
    return out


def parse_table_structure(table_img: np.ndarray) -> dict:
    if table_img.ndim == 3:
        gray = cv2.cvtColor(table_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = table_img
    H, W = gray.shape

    # OpenCV LSD only exists in cv2.ximgproc on some builds — fall back to HoughLinesP
    try:
        lsd = cv2.createLineSegmentDetector(0)
        segs = lsd.detect(gray)[0]
        if segs is None: segs = []
    except Exception:
        edges = cv2.Canny(gray, 50, 150)
        seg_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80,
                                     minLineLength=40, maxLineGap=10)
        segs = [] if seg_lines is None else seg_lines

    h_y, v_x = [], []
    for s in segs:
        x1, y1, x2, y2 = (s[0] if hasattr(s, '__len__') and len(s) == 1 else s)[0:4] \
                          if hasattr(s[0], '__len__') else s
        if abs(y2 - y1) < 5 and abs(x2 - x1) > 20:
            h_y.append(int((y1 + y2) / 2))
        elif abs(x2 - x1) < 5 and abs(y2 - y1) > 20:
            v_x.append(int((x1 + x2) / 2))

    h_y = _merge_close(h_y, tol=5)
    v_x = _merge_close(v_x, tol=5)
    # ensure outer boundaries are present
    if not h_y or h_y[0] > 8:  h_y = [0] + h_y
    if not h_y or h_y[-1] < H - 8: h_y.append(H)
    if not v_x or v_x[0] > 8:  v_x = [0] + v_x
    if not v_x or v_x[-1] < W - 8: v_x.append(W)

    n_rows, n_cols = max(0, len(h_y) - 1), max(0, len(v_x) - 1)

    cells = []
    for r in range(n_rows):
        for c in range(n_cols):
            cells.append({
                'bbox': (v_x[c], h_y[r],
                         v_x[c + 1] - v_x[c], h_y[r + 1] - h_y[r]),
                'row': r, 'col': c, 'rowspan': 1, 'colspan': 1,
            })

    # Heuristic header detection: look at brightness of first vs second row band
    header_rows = 1
    if n_rows >= 2 and table_img.ndim == 3:
        band0 = table_img[h_y[0]:h_y[1]]
        band1 = table_img[h_y[1]:h_y[2]]
        if band0.size and band1.size:
            if band0.mean() < band1.mean() - 8:   # darker top band → likely header bg
                header_rows = 1
            # Nested-column heuristic: fewer cells in row 0 than row 1
            row0_segs = sum(1 for c in cells if c['row'] == 0)
            row1_segs = sum(1 for c in cells if c['row'] == 1)
            if row0_segs and row1_segs and row0_segs < row1_segs:
                header_rows = 2

    table_type = 'normal'
    if n_cols == 2 and n_rows <= 5: table_type = 'key_value'
    if header_rows == 2:            table_type = 'nested_column'

    return {
        'cells': cells, 'n_rows': n_rows, 'n_cols': n_cols,
        'header_rows': header_rows, 'table_type': table_type,
    }

# %% [markdown]
# ## 9. OCR + data curation module
#
# SPARTAN's dual-mode OCR. Single-cell mode for small tables; multi-cell chunked mode for large tables (≥120 cells) — concatenates crops vertically with `RSRSRSRS@` row separators and splits OCR output, falling back to single-cell on mismatch.

# %%
# Initialise the EasyOCR reader ONCE — it loads ~80 MB of weights
import easyocr
OCR_READER = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
print(f'EasyOCR ready on GPU={torch.cuda.is_available()}')


def _ocr_crop(crop, ocr_reader) -> str:
    try:
        out = ocr_reader.readtext(crop, detail=0, paragraph=True)
    except Exception:
        return ''
    return ' '.join(s.strip() for s in out if s and s.strip())


def extract_cell_text(table_img: np.ndarray, cells: 'list[dict]',
                      ocr_reader, chunk_size: int = 15,
                      single_threshold: int = 120) -> 'list[list[str]]':
    if not cells: return [[]]
    n_rows = max(c['row'] for c in cells) + 1
    n_cols = max(c['col'] for c in cells) + 1
    grid = [[''] * n_cols for _ in range(n_rows)]
    use_single = len(cells) < single_threshold

    if use_single:
        for c in cells:
            x, y, w, h = c['bbox']
            pad = 3
            x1, y1 = max(0, x + pad), max(0, y + pad)
            x2, y2 = min(table_img.shape[1], x + w - pad), min(table_img.shape[0], y + h - pad)
            if x2 <= x1 or y2 <= y1: continue
            crop = table_img[y1:y2, x1:x2]
            grid[c['row']][c['col']] = _ocr_crop(crop, ocr_reader)
        return grid

    # multi-cell chunked OCR
    by_col = {}
    for c in cells: by_col.setdefault(c['col'], []).append(c)
    SEP_TEXT = 'RSRSRSRS@'
    SEP_H = 24
    for col, cell_list in by_col.items():
        cell_list.sort(key=lambda c: c['row'])
        # Chunk into groups of `chunk_size`
        for i in range(0, len(cell_list), chunk_size):
            chunk = cell_list[i:i + chunk_size]
            crops = []
            for c in chunk:
                x, y, w, h = c['bbox']
                pad = 3
                x1 = max(0, x + pad); y1 = max(0, y + pad)
                x2 = min(table_img.shape[1], x + w - pad); y2 = min(table_img.shape[0], y + h - pad)
                if x2 <= x1 or y2 <= y1:
                    crops.append(None); continue
                crops.append(table_img[y1:y2, x1:x2])
            valid = [c for c in crops if c is not None]
            if not valid: continue
            max_w = max(c.shape[1] for c in valid)
            stripes = []
            sep_h = SEP_H
            sep = np.full((sep_h, max_w, 3), 255, dtype=np.uint8)
            cv2.putText(sep, SEP_TEXT, (5, 18),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            for cr in valid:
                pad_r = max_w - cr.shape[1]
                if pad_r > 0:
                    cr = cv2.copyMakeBorder(cr, 0, 0, 0, pad_r,
                                             cv2.BORDER_CONSTANT, value=(255, 255, 255))
                stripes.append(cr); stripes.append(sep)
            stack = np.vstack(stripes)
            text_blob = _ocr_crop(stack, ocr_reader)
            pieces = text_blob.split('RSRSRSRS')
            pieces = [p.strip().lstrip('@').strip() for p in pieces if p]
            if len(pieces) != len(valid):
                # fallback: single-cell for this chunk
                for c, cr in zip(chunk, crops):
                    if cr is None: continue
                    grid[c['row']][c['col']] = _ocr_crop(cr, ocr_reader)
            else:
                for c, txt in zip(chunk, pieces):
                    grid[c['row']][c['col']] = txt
    return grid

# %% [markdown]
# ## 10. HTML reconstruction
#
# Flatten the parsed structure back into valid HTML with `<thead>` / `<tbody>` and spans preserved. Nested column headers are joined with `→` so the downstream DataFrame view can split on it for a MultiIndex.

# %%
def cells_to_html(cell_grid: 'list[list[str]]', table_structure: dict) -> str:
    if not cell_grid or not cell_grid[0]:
        return '<table></table>'
    header_rows = int(table_structure.get('header_rows', 1))
    table_type = table_structure.get('table_type', 'normal')

    n_rows = len(cell_grid); n_cols = len(cell_grid[0])

    # For nested_column: merge parent+child header text per column
    if table_type == 'nested_column' and header_rows >= 2 and n_rows >= 2:
        merged_header = []
        for c in range(n_cols):
            parent = cell_grid[0][c].strip()
            child  = cell_grid[1][c].strip()
            if parent and child and parent != child:
                merged_header.append(f'{parent} → {child}')
            else:
                merged_header.append(parent or child)
        body_rows = cell_grid[2:]
        thead = '<thead><tr>' + ''.join(f'<th>{h}</th>' for h in merged_header) + '</tr></thead>'
    else:
        thead_rows = []
        for r in range(header_rows):
            row = ''.join(f'<th>{cell_grid[r][c]}</th>' for c in range(n_cols))
            thead_rows.append(f'<tr>{row}</tr>')
        thead = '<thead>' + ''.join(thead_rows) + '</thead>'
        body_rows = cell_grid[header_rows:]

    tbody_rows = []
    for row in body_rows:
        tbody_rows.append('<tr>' + ''.join(f'<td>{c}</td>' for c in row) + '</tr>')
    tbody = '<tbody>' + ''.join(tbody_rows) + '</tbody>'

    return f'<table>{thead}{tbody}</table>'

# %% [markdown]
# ## 11. Full SPARTAN pipeline

# %%
"""Robust SPARTAN pipeline — OCR-first grid construction.

The old LSD-based path required visible ruling lines.  ~80% of FinTabNet
is borderless and produces n_rows == 0 from LSD, so the pipeline returned
empty HTML.  This rewrite builds the cell grid DIRECTLY from EasyOCR word
boxes via 1D clustering, then promotes header rows by content heuristic.

Output is fed into `cells_to_html` which the rest of the notebook already
knows how to consume.
"""
import time, traceback
import numpy as np


def _words_from_ocr(image_arr, ocr_reader, min_conf=0.20):
    """Run EasyOCR over an RGB image array, return cleaned word list."""
    try:
        raw = ocr_reader.readtext(image_arr, detail=1, paragraph=False)
    except Exception:
        return []
    words = []
    for box, txt, conf in raw:
        if conf < min_conf or not txt or not txt.strip():
            continue
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        words.append({
            'x':  float(min(xs)),  'y':  float(min(ys)),
            'w':  float(max(xs) - min(xs)),
            'h':  float(max(ys) - min(ys)),
            'cx': float((min(xs) + max(xs)) / 2),
            'cy': float((min(ys) + max(ys)) / 2),
            'text': txt.strip(),
            'conf': float(conf),
        })
    return words


def _cluster_rows(words, tol_frac=0.55):
    """Cluster words into rows by y-center.  tol = tol_frac * median height."""
    if not words: return []
    median_h = float(np.median([w['h'] for w in words]))
    tol = max(6.0, median_h * tol_frac)
    sorted_w = sorted(words, key=lambda w: w['cy'])
    rows = [[sorted_w[0]]]
    cur_cy = sorted_w[0]['cy']
    for w in sorted_w[1:]:
        # row member if its center is within tol of running row mean
        if w['cy'] - cur_cy <= tol:
            rows[-1].append(w)
            cur_cy = (cur_cy * (len(rows[-1]) - 1) + w['cy']) / len(rows[-1])
        else:
            rows.append([w])
            cur_cy = w['cy']
    for r in rows:
        r.sort(key=lambda w: w['cx'])
    return rows


def _estimate_n_cols(rows):
    """Pick the most common row-width above the 60th percentile.
    Filters out tiny title rows / single-cell footers."""
    if not rows: return 1
    counts = [len(r) for r in rows]
    if not counts: return 1
    # Use the modal width of the FULLER half of rows
    p60 = float(np.percentile(counts, 60))
    big_rows = [c for c in counts if c >= p60]
    from collections import Counter
    return Counter(big_rows).most_common(1)[0][0] if big_rows else max(counts)


def _column_centers(rows, n_cols):
    """1D KMeans on the x-centers of words from the row(s) with n_cols words."""
    if n_cols <= 1: return [0.0]
    # Seed from any row that has exactly n_cols words; fall back to linspace
    seed_row = next((r for r in rows if len(r) == n_cols), None)
    if seed_row is None:
        seed_row = max(rows, key=lambda r: len(r))
    if len(seed_row) >= n_cols:
        seeds = sorted([w['cx'] for w in seed_row[:n_cols]])
    else:
        all_cx = sorted([w['cx'] for r in rows for w in r])
        seeds = list(np.linspace(min(all_cx), max(all_cx), n_cols))
    # 5 rounds of refinement on ALL word centers
    all_cx = sorted([w['cx'] for r in rows for w in r])
    for _ in range(5):
        new_sum = [0.0] * n_cols; new_n = [0] * n_cols
        for x in all_cx:
            j = min(range(n_cols), key=lambda i: abs(seeds[i] - x))
            new_sum[j] += x; new_n[j] += 1
        seeds = [new_sum[i]/new_n[i] if new_n[i] else seeds[i] for i in range(n_cols)]
        seeds.sort()
    return seeds


def _assign_to_grid(rows, column_centers):
    """Place each word into its nearest (row, col) cell — produce 2D text grid."""
    n_rows = len(rows); n_cols = len(column_centers)
    grid = [[''] * n_cols for _ in range(n_rows)]
    for r_i, row_words in enumerate(rows):
        for w in row_words:
            c_i = min(range(n_cols), key=lambda i: abs(column_centers[i] - w['cx']))
            grid[r_i][c_i] = (grid[r_i][c_i] + ' ' + w['text']).strip()
    return grid


def _detect_header_rows(grid):
    """Top contiguous rows whose non-empty cells contain NO digits are headers."""
    if not grid: return 1
    def _has_digit(s): return any(ch.isdigit() for ch in s)
    n_header = 0
    for row in grid:
        non_empty = [c for c in row if c.strip()]
        if non_empty and not any(_has_digit(c) for c in non_empty):
            n_header += 1
        else:
            break
    return max(1, min(n_header, 3))   # cap at 3 (we don't expect deeper)


def _detect_spans(rows, column_centers):
    """Mark cells whose source-word bbox straddles multiple column centers
    as having a colspan.  Returns dict (row_idx, col_idx) → colspan."""
    spans = {}
    n_cols = len(column_centers)
    for r_i, row_words in enumerate(rows):
        for w in row_words:
            # Find columns whose center lies inside [w.x, w.x+w.w]
            covered = [i for i in range(n_cols)
                       if w['x'] - 4 <= column_centers[i] <= w['x'] + w['w'] + 4]
            if len(covered) >= 2:
                spans[(r_i, covered[0])] = len(covered)
    return spans


def _grid_to_html(grid, header_rows: int, spans=None) -> str:
    """Build HTML from a 2D text grid + header_rows + optional span dict."""
    spans = spans or {}
    if not grid or not grid[0]:
        return '<table></table>'
    n_rows, n_cols = len(grid), len(grid[0])
    # mark cells absorbed by an earlier span so we skip them
    absorbed = set()
    for (r, c), cs in spans.items():
        for k in range(1, cs):
            absorbed.add((r, c + k))

    thead_parts = []
    for r in range(header_rows):
        cells = []
        for c in range(n_cols):
            if (r, c) in absorbed: continue
            cs = spans.get((r, c), 1)
            attr = f' colspan="{cs}"' if cs > 1 else ''
            cells.append(f'<th{attr}>{grid[r][c]}</th>')
        thead_parts.append('<tr>' + ''.join(cells) + '</tr>')
    thead = '<thead>' + ''.join(thead_parts) + '</thead>' if thead_parts else ''

    tbody_parts = []
    for r in range(header_rows, n_rows):
        cells = []
        for c in range(n_cols):
            if (r, c) in absorbed: continue
            cs = spans.get((r, c), 1)
            attr = f' colspan="{cs}"' if cs > 1 else ''
            cells.append(f'<td{attr}>{grid[r][c]}</td>')
        tbody_parts.append('<tr>' + ''.join(cells) + '</tr>')
    tbody = '<tbody>' + ''.join(tbody_parts) + '</tbody>' if tbody_parts else ''

    return f'<table>{thead}{tbody}</table>'


def run_spartan_pipeline(img_path, ocr_reader, debug: bool = False,
                          min_conf: float = 0.20,
                          row_tol_frac: float = 0.55) -> 'tuple[str, dict]':
    """End-to-end pipeline — uses OCR-first grid construction."""
    timings = {}
    try:
        t0 = time.perf_counter()
        binary, color = preprocess_image(img_path)
        timings['preprocess_s'] = round(time.perf_counter() - t0, 3)

        t0 = time.perf_counter()
        words = _words_from_ocr(color, ocr_reader, min_conf=min_conf)
        timings['ocr_s'] = round(time.perf_counter() - t0, 3)
        if not words:
            return '<table></table>', {'n_cells': 0, 'reason': 'no_ocr_output', 'timings_s': timings}

        t0 = time.perf_counter()
        rows = _cluster_rows(words, tol_frac=row_tol_frac)
        n_cols = _estimate_n_cols(rows)
        col_centers = _column_centers(rows, n_cols)
        grid = _assign_to_grid(rows, col_centers)
        header_rows = _detect_header_rows(grid)
        spans = _detect_spans(rows, col_centers)
        timings['assembly_s'] = round(time.perf_counter() - t0, 3)

        html = _grid_to_html(grid, header_rows=header_rows, spans=spans)
        meta = {
            'n_cells':     sum(1 for r in grid for c in r if c.strip()),
            'n_rows':      len(grid),
            'n_cols':      n_cols,
            'header_rows': header_rows,
            'n_spans':     len(spans),
            'n_words':     len(words),
            'table_type':  'nested_column' if header_rows >= 2 else 'normal',
            'timings_s':   timings,
        }
        return html, meta
    except Exception as exc:
        if debug: traceback.print_exc()
        return '<table></table>', {'error': str(exc), 'n_cells': 0,
                                     'timings_s': timings}


# Smoke test on the first test image
if len(test_df) > 0:
    sample = test_df.iloc[0]
    print(f'Sample : {sample.image_id}  ({sample.image_type})')
    html, meta = run_spartan_pipeline(sample.img_path_resolved, OCR_READER)
    print(f'Cells   : {meta.get("n_cells")}')
    print(f'Grid    : {meta.get("n_rows")} rows × {meta.get("n_cols")} cols')
    print(f'Headers : {meta.get("header_rows")}    Spans: {meta.get("n_spans")}')
    print(f'Words   : {meta.get("n_words")}')
    print(f'Timings : {meta.get("timings_s")}')
    print(f'TEDS    : {teds_score(html, sample.html):.3f}')
    print(f'TEDS-S  : {teds_score(html, sample.html, structure_only=True):.3f}')

# %% [markdown]
# ## 12. Benchmark on the full test split
#
# Checkpoint after every 10 images so a Colab disconnect doesn't lose progress. Print running mean every 25 images. Matches the sprint-2 checkpoint pattern.

# %%
# %%time
from tqdm.auto import tqdm

CHECKPOINT_PATH = SPARTAN_OUT / 'spartan_benchmark_checkpoint.csv'

# Resume from checkpoint if it exists
if CHECKPOINT_PATH.exists():
    done = pd.read_csv(CHECKPOINT_PATH)
    print(f'Resuming from {len(done)} completed rows.')
else:
    done = pd.DataFrame(columns=['image_id', 'image_type', 'teds', 'teds_s',
                                  'inference_time_s', 'pred_html', 'n_cells',
                                  'table_type', 'error'])

done_ids = set(done['image_id'].tolist())
todo = test_df[~test_df['image_id'].isin(done_ids)]
print(f'Remaining test rows: {len(todo)} / {len(test_df)}')

rows = done.to_dict('records')
ckpt_every = 10
print_every = 25
for i, (_, sample) in enumerate(tqdm(todo.iterrows(), total=len(todo))):
    t_start = time.perf_counter()
    try:
        pred_html, meta = run_spartan_pipeline(sample.img_path_resolved, OCR_READER)
        teds   = teds_score(pred_html, sample.html)
        teds_s = teds_score(pred_html, sample.html, structure_only=True)
        err = meta.get('error', '')
    except Exception as exc:
        pred_html, meta = '<table></table>', {'n_cells': 0, 'table_type': ''}
        teds = teds_s = 0.0
        err = str(exc)
    rows.append({
        'image_id': sample.image_id,
        'image_type': sample.image_type,
        'teds': round(teds, 4),
        'teds_s': round(teds_s, 4),
        'inference_time_s': round(time.perf_counter() - t_start, 3),
        'pred_html': pred_html,
        'n_cells': meta.get('n_cells', 0),
        'table_type': meta.get('table_type', ''),
        'error': err,
    })
    if (i + 1) % ckpt_every == 0:
        pd.DataFrame(rows).to_csv(CHECKPOINT_PATH, index=False)
    if (i + 1) % print_every == 0:
        partial = pd.DataFrame(rows[-print_every:])
        print(f'  rolling mean (last {print_every}): '
              f'TEDS={partial.teds.mean():.3f}  TEDS-S={partial.teds_s.mean():.3f}')

# Final save
results_df = pd.DataFrame(rows)
results_df.to_csv(CHECKPOINT_PATH, index=False)
print(f'\nSaved checkpoint to {CHECKPOINT_PATH}')

# %% [markdown]
# ## 13. Results analysis
#
# Produce the same summary format as sprint-2 so the rows integrate into the ablation table. Compute cell-level F1 on extracted text vs ground-truth cell text. Two plots: TEDS by image_type and time vs TEDS scatter.

# %%
import matplotlib.pyplot as plt
from lxml import html as lhtml

# ── overall stats ───────────────────────────────────────────────────────
print(f'N evaluated: {len(results_df)}')
print(f'Mean TEDS       : {results_df.teds.mean():.4f}')
print(f'Median TEDS     : {results_df.teds.median():.4f}')
print(f'Mean TEDS-S     : {results_df.teds_s.mean():.4f}')
print(f'>= 0.50 rate    : {(results_df.teds >= 0.50).mean()*100:.1f}%')

# ── failure buckets ─────────────────────────────────────────────────────
def _bucket(t):
    if t == 0:     return 'A: zero'
    if t < 0.3:    return 'B: <0.3'
    if t < 0.5:    return 'C: 0.3-0.5'
    if t < 0.8:    return 'D: 0.5-0.8'
    return 'E: >=0.8'
results_df['bucket'] = results_df.teds.apply(_bucket)
print('\nFailure-mode buckets:')
print(results_df.bucket.value_counts().sort_index().to_string())

# ── per image_type breakdown ────────────────────────────────────────────
print('\nMean TEDS by image_type:')
print(results_df.groupby('image_type')['teds'].agg(['count','mean','median']).round(3).to_string())

# ── cell-level F1 (whitespace-normalised string match) ──────────────────
def _cell_texts(html_str: str) -> 'list[str]':
    try:
        root = lhtml.fragment_fromstring(html_str or '', create_parent='div')
        cells = root.cssselect('td, th')
        return [c.text_content().strip().lower() for c in cells if c.text_content().strip()]
    except Exception:
        return []

tp = fp = fn = 0
for _, r in results_df.iterrows():
    pred = set(_cell_texts(r['pred_html']))
    gt_html = test_df[test_df.image_id == r['image_id']]
    if gt_html.empty: continue
    gt = set(_cell_texts(gt_html.iloc[0]['html']))
    tp += len(pred & gt)
    fp += len(pred - gt)
    fn += len(gt - pred)
prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
f1 = 2 * prec * rec / max(1e-9, prec + rec)
print(f'\nCell F1: precision={prec:.3f} recall={rec:.3f} F1={f1:.3f}')

# ── timing ─────────────────────────────────────────────────────────────
print(f'\nInference time: mean {results_df.inference_time_s.mean():.2f}s   '
      f'p95 {results_df.inference_time_s.quantile(0.95):.2f}s   '
      f'max {results_df.inference_time_s.max():.2f}s')

# ── plot 1: TEDS by image_type vs Florence-2 ZeroShot baseline ─────────
flor_zs = 0.15   # sprint-1 reference; replace with real values if available
fig, ax = plt.subplots(figsize=(9, 4))
gb = results_df.groupby('image_type')['teds'].mean()
xs = np.arange(len(gb))
ax.bar(xs - 0.18, [flor_zs] * len(gb), width=0.36, color='#888', label='Florence-2 ZeroShot')
ax.bar(xs + 0.18, gb.values, width=0.36, color='#800000', label='SPARTAN')
ax.set_xticks(xs); ax.set_xticklabels(gb.index, rotation=15)
ax.set_ylabel('Mean TEDS'); ax.set_title('TEDS by image_type')
ax.axhline(0.50, color='#15803d', ls='--', label='Target 0.50')
ax.legend(frameon=False); plt.tight_layout(); plt.show()

# ── plot 2: time vs TEDS scatter ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
for it, sub in results_df.groupby('image_type'):
    ax.scatter(sub.inference_time_s, sub.teds, s=18, alpha=0.7, label=it)
ax.set_xlabel('Inference time (s)'); ax.set_ylabel('TEDS')
ax.set_title('Time vs TEDS by image_type')
ax.axhline(0.50, color='#15803d', ls='--', lw=0.8)
ax.legend(frameon=False, fontsize=8); plt.tight_layout(); plt.show()

results_df.to_csv(SPARTAN_OUT / 'spartan_results.csv', index=False)
print(f'\nSaved {SPARTAN_OUT / "spartan_results.csv"}')

# %% [markdown]
# ## 14. DataFrame export — HTML → `pd.DataFrame` with hierarchical headers
#
# This is the product-layer artefact for Member D's HITL UI: extracted tables surfaced as DataFrames with `MultiIndex` columns where the source HTML had nested headers.

# %%
from IPython.display import display, Image as IPYImage
from lxml import html as lhtml

def html_to_dataframe(html_str: str) -> 'pd.DataFrame':
    try:
        root = lhtml.fragment_fromstring(html_str or '', create_parent='div')
        table = root.find('.//table')
        if table is None: return pd.DataFrame()
        thead = table.find('thead'); tbody = table.find('tbody')

        # Build column header(s)
        header_rows = thead.findall('tr') if thead is not None else []
        if not header_rows:
            header_rows = [table.find('tr')] if table.find('tr') is not None else []
        if header_rows:
            top = header_rows[0]
            header_cells = [c.text_content().strip() for c in top.findall('th') + top.findall('td')]
            # split on '→' to recover nested hierarchy
            tuples = []
            for h in header_cells:
                if '→' in h:
                    parts = [p.strip() for p in h.split('→')]
                    tuples.append(tuple(parts))
                else:
                    tuples.append((h,))
            depth = max(len(t) for t in tuples) if tuples else 1
            tuples = [t + ('',) * (depth - len(t)) for t in tuples]
            cols = pd.MultiIndex.from_tuples(tuples) if depth > 1 else [t[0] for t in tuples]
        else:
            cols = None

        body_rows = (tbody.findall('tr') if tbody is not None else []) or table.findall('tr')[len(header_rows):]
        data = []
        for tr in body_rows:
            data.append([c.text_content().strip() for c in tr.findall('td') + tr.findall('th')])
        if cols is not None and data:
            # pad / trim row widths to match cols
            ncol = len(cols) if not isinstance(cols, pd.MultiIndex) else cols.size
            data = [row + [''] * (ncol - len(row)) for row in data]
            data = [row[:ncol] for row in data]
            return pd.DataFrame(data, columns=cols)
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()


# Demonstrate on 3 sample test images
sample_ids = results_df.sort_values('teds', ascending=False).head(3)['image_id'].tolist()
for sid in sample_ids:
    row = results_df[results_df.image_id == sid].iloc[0]
    gt  = test_df[test_df.image_id == sid].iloc[0]
    print(f'\n=== {sid}  ({row.image_type})  TEDS={row.teds:.3f} ===')
    display(IPYImage(filename=gt.img_path_resolved, width=520))
    df = html_to_dataframe(row.pred_html)
    print('  → extracted DataFrame:')
    display(df.head(12))

# %% [markdown]
# ## 15. Tune SPARTAN parameters by grid search on the TRAIN split
#
# Sweeps the two parameters that actually move TEDS in the OCR-first pipeline (`ocr_min_conf` and `row_tol_frac`) via coordinate descent. Saves `preprocessing_params.json` + `grid_params.json` ready for the dashboard's SPARTAN runner.

# %%
"""Tune SPARTAN by grid search on the TRAIN split.

These are the parameters that actually move TEDS in the OCR-first pipeline.

    preprocessing_params.json
        target_width        — image normalisation width (affects OCR resolution)
        clahe               — CLAHE contrast boost before threshold
        deskew              — auto-correct skew > 1°
    grid_params.json
        ocr_min_conf        — minimum EasyOCR confidence to accept a word
        row_tol_frac        — y-tolerance for row clustering, as fraction of median word height
        header_max_rows     — cap on header-row count
        do_colspan_merge    — emit colspan attrs for wide words

The pipeline always returns a usable grid (OCR-first), so the optimiser never
has to deal with all-zero scores — but we keep defensive defaults if a config
errors out.
"""
import json, itertools, random, time, numpy as np
from tqdm.auto import tqdm

train_df = splits[splits['split'] == 'train']

# Stratified sample of train images for fast scoring
TUNE_PER_TYPE = 18
random.seed(42)
tune_idx = []
for it, sub in train_df.groupby('image_type'):
    pick = sub.sample(min(TUNE_PER_TYPE, len(sub)), random_state=42)
    tune_idx.extend(pick.index.tolist())
tune_df = train_df.loc[tune_idx].reset_index(drop=True)
print(f'Tuning on {len(tune_df)} stratified train images')

# ── Defaults (the "do-nothing" config — used if tuning yields no improvement)
DEFAULTS = {
    'target_width':     1600,
    'clahe':            False,
    'deskew':           False,
    'ocr_min_conf':     0.20,
    'row_tol_frac':     0.55,
    'header_max_rows':  3,
    'do_colspan_merge': True,
}

# Compact grids — coordinate descent over 2 axes at a time
GRID = {
    'ocr_min_conf':  [0.15, 0.20, 0.25, 0.30],
    'row_tol_frac':  [0.40, 0.50, 0.55, 0.65, 0.80],
}

def _eval_config(cfg, sub_df):
    teds_vals = []
    for _, s in sub_df.iterrows():
        try:
            html, _ = run_spartan_pipeline(
                s.img_path, OCR_READER,
                min_conf=cfg['ocr_min_conf'],
                row_tol_frac=cfg['row_tol_frac'],
            )
            teds_vals.append(teds_score(html, s.html))
        except Exception:
            teds_vals.append(0.0)
    return float(np.mean(teds_vals)) if teds_vals else 0.0


# Coordinate descent
print(f'\nBaseline (defaults) on tune set …')
base_cfg = dict(DEFAULTS)
base_t = _eval_config(base_cfg, tune_df)
print(f'  baseline TEDS = {base_t:.3f}')

best_cfg = dict(base_cfg)
best_t   = base_t

print(f'\n[1/2] ocr_min_conf sweep …')
for v in tqdm(GRID['ocr_min_conf'], leave=False):
    c = dict(best_cfg, ocr_min_conf=v)
    t = _eval_config(c, tune_df)
    print(f'  ocr_min_conf={v}:  TEDS={t:.3f}')
    if t > best_t:
        best_t, best_cfg = t, c
print(f'  best so far: ocr_min_conf={best_cfg["ocr_min_conf"]}  TEDS={best_t:.3f}')

print(f'\n[2/2] row_tol_frac sweep …')
for v in tqdm(GRID['row_tol_frac'], leave=False):
    c = dict(best_cfg, row_tol_frac=v)
    t = _eval_config(c, tune_df)
    print(f'  row_tol_frac={v}:  TEDS={t:.3f}')
    if t > best_t:
        best_t, best_cfg = t, c
print(f'  best so far: row_tol_frac={best_cfg["row_tol_frac"]}  TEDS={best_t:.3f}')


# Save tuned params split into the two files the dashboard reads
pre_params = {
    'global': {
        'target_width': best_cfg['target_width'],
        'clahe':        best_cfg['clahe'],
        'deskew':       best_cfg['deskew'],
    }
}
grid_params = {
    'global': {
        'ocr_min_conf':     best_cfg['ocr_min_conf'],
        'row_tol_frac':     best_cfg['row_tol_frac'],
        'header_max_rows':  best_cfg['header_max_rows'],
        'do_colspan_merge': best_cfg['do_colspan_merge'],
    }
}

(SPARTAN_OUT / 'preprocessing_params.json').write_text(json.dumps(pre_params, indent=2))
(SPARTAN_OUT / 'grid_params.json').write_text(json.dumps(grid_params, indent=2))
print(f'\n✓ wrote {SPARTAN_OUT/"preprocessing_params.json"}')
print(f'✓ wrote {SPARTAN_OUT/"grid_params.json"}')
print(f'\n── summary ──────────────────────────────────────')
print(f'  baseline TEDS    : {base_t:.4f}')
print(f'  tuned TEDS       : {best_t:.4f}   Δ {best_t-base_t:+.4f}')
print(f'  best config      : {best_cfg}')

# %% [markdown]
# ## 16. Hand off to the dashboard
#
# Local run → JSON params already in `TableSight/models/jared/`. Click **🔁 Reload models** in the Streamlit sidebar to apply.
#
# Colab run → artefacts are zipped and the browser triggers a download. Extract the zip into your local `TableSight/models/jared/` directory and reload.

# %%
"""Hand off the trained SPARTAN artefacts to the dashboard.

Local run  → params already wrote to TableSight/models/jared.  Just summarise.
Colab run  → zip the artefact bundle and trigger a browser download so the
             user can drop it into their local TableSight folder.
"""
import shutil, json, sys
from pathlib import Path

artefacts = {
    'preprocessing_params.json': SPARTAN_OUT / 'preprocessing_params.json',
    'grid_params.json':           SPARTAN_OUT / 'grid_params.json',
    'benchmark.csv':      SPARTAN_OUT / 'benchmark.csv',
    'spartan_results.csv':        SPARTAN_OUT / 'spartan_results.csv',
}

# Also save the final benchmark in the schema the dashboard expects
if 'results_df' in dir():
    bench_path = SPARTAN_OUT / 'benchmark.csv'
    results_df.to_csv(bench_path, index=False)
    artefacts['benchmark.csv'] = bench_path

print('Generated artefacts:')
for name, p in artefacts.items():
    status = '✓' if Path(p).exists() else '✗ (missing)'
    print(f'  {status}  {name}  →  {p}')

if IS_COLAB:
    # Bundle into a zip and trigger download
    bundle_path = Path('/content/spartan_handoff.zip')
    if bundle_path.exists(): bundle_path.unlink()
    import zipfile
    with zipfile.ZipFile(bundle_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for name, p in artefacts.items():
            if Path(p).exists():
                zf.write(p, arcname=name)
    print(f'\n📦 packaged: {bundle_path}  ({bundle_path.stat().st_size//1024} KB)')
    try:
        from google.colab import files
        print('   Triggering browser download…')
        files.download(str(bundle_path))
        print('\nDrop the contents into your local repo at:')
        print('   TableSight/models/jared/')
        print('Then click "🔁 Reload models" in the Streamlit dashboard.')
    except Exception as exc:
        print(f'   (auto-download unavailable: {exc})')
        print('   Right-click the file in the Colab file browser → Download.')
else:
    # Local run — params are already in the right place if PROJECT_ROOT/TableSight exists
    target = PROJECT_ROOT / 'TableSight' / 'models' / 'jared'
    if SPARTAN_OUT == target:
        print('\n✓ Artefacts are already at the dashboard\'s pickup location.')
        print('  Click "🔁 Reload models" in the Streamlit dashboard to apply.')
    else:
        # Copy them over
        target.mkdir(parents=True, exist_ok=True)
        for name, p in artefacts.items():
            if Path(p).exists():
                shutil.copy2(p, target / name)
        print(f'\n✓ Copied artefacts to {target}.')
        print('  Click "🔁 Reload models" in the Streamlit dashboard to apply.')

print('\n── final summary ────────────────────────────────────────')
if 'results_df' in dir():
    overall = results_df.teds.mean()
    target_t = 0.50
    print(f'Mean TEDS on test split   : {overall:.4f}')
    print(f'Target                    : {target_t:.2f}')
    print(f'Hit target?               : {"YES" if overall >= target_t else "NO"}')
else:
    print('(run the benchmark cell before this one for a TEDS summary)')

# %%

