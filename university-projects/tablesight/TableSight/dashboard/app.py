"""TableSight  ―  Model Comparison Dashboard.

Switch between three table-extraction models and score them against ground
truth in real time, on either an uploaded image or a sample from the local
FinTabNet gallery.

Models
------
    1. Florence-2 — LoRA on Florence-2-base; falls back to OCR-with-region
                    clustering when no adapter is loaded.
    2. UniTable   — 3-pass encoder-decoder (structure / bbox / per-cell text)
                    using the poloclub/unitable source.
    3. SPARTAN    — EasyOCR + 1D-KMeans grid clustering, tuned params loaded
                    from models/spartan/*.json.

Tabs
----
    Compare              upload OR pick from gallery, multiselect models,
                         run + score against ground truth
    Benchmark            summary stats from the saved evaluation run
    Batch Eval           run all loaded models over a CSV slice
    Failure Analysis     TEDS<threshold cases with diffs and category pies

Run
---
    streamlit run TableSight/dashboard/app.py
"""
from __future__ import annotations

import concurrent.futures
import difflib
import io
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from PIL import Image

# Ensure repo root is importable
_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from TableSight.dashboard.utils.model_loader import load_config, load_models     # noqa: E402
from TableSight.dashboard.utils.metrics import (                                      # noqa: E402
    compute_teds, compute_teds_s, classify_failure,
)
from TableSight.dashboard.utils.charts import (                             # noqa: E402
    metric_bars, teds_distribution, failure_pies, summary_table,
)
from TableSight.dashboard.utils.table_parser import (                          # noqa: E402
    robust_html_to_dataframe,
)


# ══════════════════════════════════════════════════════════════════════════
# Page chrome — UChicago maroon palette
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="TableSight · Model Comparison", layout="wide")

# Display labels — keep internal model names ("spartan") stable so the
# config / runner code doesn't churn, but show "SPARTAN" in the UI.
MODEL_DISPLAY = {
    "florence":    "Florence-2",
    "unitable":    "UniTable",
    "spartan": "SPARTAN",
}
def _disp(name: str) -> str:
    return MODEL_DISPLAY.get(name, name)

st.markdown("""
<style>
:root {
    --maroon:      #800000;
    --maroon-dk:   #5a0000;
    --maroon-lt:   #fdf4f4;
    --maroon-md:   #f0e0e0;
    --border:      #dedad2;
    --border-hi:   #c8c3b8;
    --bg:          #f5f4f0;
    --surface:     #ffffff;
    --surface-2:   #f9f8f5;
    --text:        #1c1917;
    --text-2:      #44403c;
    --muted:       #78716c;
    --green:       #15803d;
    --amber:       #b45309;
    --red:         #b91c1c;
    --blue:        #1d4ed8;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── Top banner — UChicago-style ─────────────────────────────── */
.ts-banner {
    background: linear-gradient(180deg, var(--maroon) 0%, var(--maroon-dk) 100%);
    color: #fff;
    padding: 18px 34px;
    margin: -16px -2rem 28px -2rem;
    border-bottom: 4px double rgba(255,255,255,0.25);
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 2px 12px rgba(90, 0, 0, 0.18);
}
.ts-banner-left  { display: flex; align-items: baseline; gap: 18px; }
.ts-banner-right { display: flex; align-items: center; gap: 16px; }

.ts-title {
    font-family: 'Source Serif 4', 'Times New Roman', serif;
    font-size: 24px;
    font-weight: 700;
    color: #fff;
    letter-spacing: -0.3px;
}
.ts-sub {
    font-family: 'Source Serif 4', 'Times New Roman', serif;
    font-size: 14px;
    font-weight: 400;
    font-style: italic;
    color: rgba(255,255,255,0.78);
    letter-spacing: 0.2px;
    border-left: 1px solid rgba(255,255,255,0.3);
    padding-left: 18px;
}
.ts-tag {
    font-family: 'Inter', sans-serif;
    font-size: 10px;
    font-weight: 500;
    color: rgba(255,255,255,0.9);
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 3px;
    padding: 4px 11px;
    letter-spacing: 0.6px;
    text-transform: uppercase;
}
.ts-uchicago {
    font-family: 'Source Serif 4', 'Times New Roman', serif;
    font-size: 12px;
    font-weight: 600;
    color: #fff;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    border-left: 1px solid rgba(255,255,255,0.3);
    padding-left: 16px;
}

/* ── Cards / chrome ───────────────────────────────────────────── */
.small-caption { font-size: 11px; color: var(--muted); }
.model-card {
    border: 1px solid var(--border); border-radius: 8px;
    padding: 12px 14px; margin-bottom: 8px;
    background: var(--surface);
    border-left: 3px solid var(--maroon);
}
.model-card.no-output { border-left-color: var(--amber); background: var(--maroon-lt); }
.kpi-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 14px 18px;
}
.kpi-label { font-size: 10px; text-transform: uppercase; color: var(--muted); letter-spacing: 1px; }
.kpi-value { font-size: 24px; font-weight: 700; color: var(--text); font-family: 'JetBrains Mono', monospace; }
.kpi-value.green  { color: var(--green); }
.kpi-value.amber  { color: var(--amber); }
.kpi-value.red    { color: var(--red); }
.kpi-value.maroon { color: var(--maroon); }
.kpi-value.blue   { color: var(--blue); }

/* ── Buttons ───────────────────────────────────────────────────── */
div[data-testid="stButton"] button[kind="primary"],
div[data-testid="stDownloadButton"] button {
    background: var(--maroon) !important;
    border: 1px solid var(--maroon-dk) !important;
    color: #fff !important;
    font-weight: 600 !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover,
div[data-testid="stDownloadButton"] button:hover {
    background: var(--maroon-dk) !important;
    border-color: var(--maroon-dk) !important;
}
div[data-testid="stButton"] button:not([kind="primary"]) {
    border: 1px solid var(--border-hi) !important;
    background: var(--surface) !important;
    color: var(--text-2) !important;
}

/* ── Sidebar ────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--surface-2) !important;
    border-right: 1px solid var(--border) !important;
}

/* ── Tabs ─────────────────────────────────────────────────────── */
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--maroon) !important;
    border-bottom-color: var(--maroon) !important;
}
button[data-baseweb="tab"][aria-selected="true"] p {
    color: var(--maroon) !important;
    font-weight: 600 !important;
}

/* ── Tables / dataframes ──────────────────────────────────────── */
table { font-size: 12px; }
thead th { background: var(--surface-2) !important; color: var(--text) !important;
           font-size: 10px !important; text-transform: uppercase;
           letter-spacing: 0.6px !important; }

/* ── File uploader ────────────────────────────────────────────── */
div[data-testid="stFileUploaderDropzone"] {
    background: var(--surface-2) !important;
    border: 2px dashed var(--border-hi) !important;
}
div[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--maroon) !important;
    background: var(--maroon-lt) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div class='ts-banner'>"
    "<div class='ts-banner-left'>"
    "<span class='ts-title'>TableSight</span>"
    "<span class='ts-sub'>Hierarchical Table Extraction</span>"
    "</div>"
    "<div class='ts-banner-right'>"
    "<span class='ts-tag'>Florence-2 · UniTable · SPARTAN</span>"
    "<span class='ts-uchicago'>The University of Chicago</span>"
    "</div>"
    "</div>",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════
# Model loading (cached in session_state)
# ══════════════════════════════════════════════════════════════════════════

def _ensure_models() -> Dict[str, Optional[Any]]:
    if "extractors" in st.session_state:
        return st.session_state.extractors
    with st.spinner("Loading models … (this may take 30-60 s on first run)"):
        cfg = load_config()
        st.session_state.cfg = cfg
        extractors, status = load_models(cfg)
        st.session_state.extractors = extractors
        st.session_state.status = status
    return st.session_state.extractors


# ══════════════════════════════════════════════════════════════════════════
# Gallery (locally cached FinTabNet PNGs)
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _load_gallery_index(splits_csv: str, image_root: str) -> pd.DataFrame:
    """Return a DataFrame of the local FinTabNet samples with usable img_path
    columns. Caches across reruns."""
    if not splits_csv or not Path(splits_csv).exists():
        return pd.DataFrame()
    df = pd.read_csv(splits_csv)
    # Normalise img_path
    if "img_path" not in df.columns:
        df["img_path"] = ""
    needs_fix = df["img_path"].isna() | (df["img_path"].astype(str) == "")
    if image_root and needs_fix.any():
        base = Path(image_root)
        df.loc[needs_fix, "img_path"] = df.loc[needs_fix, "img_id"].apply(
            lambda i: str(base / f"{i}.png")
        )
    # only keep rows with a real file
    df = df[df["img_path"].apply(lambda p: Path(p).exists())]
    if "image_type" not in df.columns:
        df["image_type"] = "unknown"
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _load_benchmark(path: str) -> pd.DataFrame:
    if not path or not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# ══════════════════════════════════════════════════════════════════════════
# Inference helpers
# ══════════════════════════════════════════════════════════════════════════

def run_models(image: Image.Image, names: List[str], parallel: bool = True
               ) -> Dict[str, Dict[str, Any]]:
    """Run the selected models on one image."""
    extractors = st.session_state.extractors
    active = {n: e for n, e in extractors.items() if n in names and e is not None}
    if not active:
        return {}
    results: Dict[str, Dict[str, Any]] = {}
    if parallel and len(active) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(active)) as pool:
            futs = {pool.submit(e.predict, image): n for n, e in active.items()}
            for fut in concurrent.futures.as_completed(futs):
                n = futs[fut]
                try:
                    results[n] = fut.result()
                except Exception as exc:
                    results[n] = {"html": "", "time_s": 0.0,
                                  "metadata": {"error": str(exc)}}
    else:
        for n, e in active.items():
            try:
                results[n] = e.predict(image)
            except Exception as exc:
                results[n] = {"html": "", "time_s": 0.0,
                              "metadata": {"error": str(exc)}}
    return results


def compute_metrics(pred_html: str, gt_html: str) -> Dict[str, float]:
    return {
        "TEDS":   compute_teds(pred_html, gt_html),
        "TEDS-S": compute_teds_s(pred_html, gt_html),
    }


def html_stats(pred_html: str) -> Dict[str, Any]:
    """Structural counts derived straight from the predicted HTML.

    Used to populate the side-by-side table for EVERY model, regardless of
    which metadata keys each runner happens to emit (e.g. UniTable returns
    `n_bboxes`/`n_cells_filled` but not `n_rows`/`n_cols`).
    """
    empty = {"n_rows": "—", "n_cols": "—", "n_cells": "—",
             "header_rows": "—", "n_spans": "—"}
    if not pred_html or "<table" not in pred_html.lower():
        return empty
    try:
        from lxml import html as lhtml
        root = lhtml.fragment_fromstring(pred_html, create_parent="div")
        tbl = root.find(".//table")
        if tbl is None:
            return empty
    except Exception:
        return empty

    trs = tbl.xpath(".//tr")
    n_cells = len(tbl.xpath(".//td | .//th"))

    # widest row (accounting for colspan) = column count
    widths, n_spans = [], 0
    for tr in trs:
        w = 0
        for cell in tr.xpath("./td | ./th"):
            cs = cell.get("colspan", "1")
            rs = cell.get("rowspan", "1")
            try: w += int(cs)
            except ValueError: w += 1
            if (cs.isdigit() and int(cs) > 1) or (rs.isdigit() and int(rs) > 1):
                n_spans += 1
        widths.append(w)

    # contiguous all-<th> rows from the top = header rows
    header_rows = 0
    for tr in trs:
        cells = tr.xpath("./td | ./th")
        if cells and all(c.tag == "th" for c in cells):
            header_rows += 1
        else:
            break

    return {
        "n_rows":      len(trs),
        "n_cols":      max(widths) if widths else 0,
        "n_cells":     n_cells,
        "header_rows": header_rows,
        "n_spans":     n_spans,
    }


def pdf_first_page_to_image(file_bytes: bytes) -> Optional[Image.Image]:
    try:
        import fitz
    except ImportError:
        return None
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        if doc.page_count == 0: return None
        pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2, 2))
        return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════
# Sidebar — model status + gallery filter + selection
# ══════════════════════════════════════════════════════════════════════════

st.sidebar.title("TableSight")
st.sidebar.caption("Multi-model table extraction")

if st.sidebar.button("Reload models"):
    for k in ("extractors", "status", "cfg"): st.session_state.pop(k, None)
    st.cache_data.clear()

extractors = _ensure_models()
status = st.session_state.get("status", {})
cfg = st.session_state.get("cfg", {})

st.sidebar.subheader("Model status")
for name in ("florence", "unitable", "spartan"):
    ok = extractors.get(name) is not None
    icon = "[ok]" if ok else "[warn]"
    msg = "loaded" if ok else (status.get(name) or "not configured")
    short = msg if len(msg) < 60 else msg[:57] + "…"
    st.sidebar.markdown(
        f"{icon} **{_disp(name)}** — <span class='small-caption'>{short}</span>",
        unsafe_allow_html=True,
    )

device_used = next((e.device for e in extractors.values() if e), "—")
st.sidebar.caption(f"Device: `{device_used}`")

st.sidebar.markdown("---")
st.sidebar.subheader("Active models")
available = [n for n, e in extractors.items() if e is not None]
if not available:
    st.sidebar.warning("No models loaded.  Check `dashboard/config.yaml`.")
selected_models = st.sidebar.multiselect(
    "Choose which models to run",
    available,
    default=available,
    format_func=_disp,
    key="selected_models",
)


# ══════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════

TAB_COMPARE, TAB_BENCH, TAB_BATCH, TAB_FAIL = st.tabs([
    "Compare",
    "Benchmark",
    "Batch Eval",
    "Failure Analysis",
])

# ─────────────────────────────────────── Tab 1: Compare ─────────────────
with TAB_COMPARE:
    st.header("Single-image comparison")
    st.caption("Pick a table image, run the selected models, and see them side-by-side.")

    gallery_df = _load_gallery_index(
        cfg.get("paths", {}).get("splits_csv", ""),
        cfg.get("paths", {}).get("image_root", ""),
    )

    # Source picker
    source = st.radio(
        "Image source", ["Upload", "Random from gallery", "Browse gallery"],
        horizontal=True, index=1 if not gallery_df.empty else 0,
        label_visibility="collapsed",
    )

    image: Optional[Image.Image] = None
    image_label: str = ""
    gt_html: str = ""
    chosen_id: Optional[str] = None

    # ── Source: upload ────────────────────────────────────────────────
    if source == "Upload":
        uploaded = st.file_uploader("Table image", type=["png", "jpg", "jpeg", "pdf"])
        gt_html = st.text_area("Ground-truth HTML (optional, paste here for metrics)",
                                height=120, placeholder="<table>...</table>")
        if uploaded is not None:
            raw = uploaded.read()
            if uploaded.type == "application/pdf":
                image = pdf_first_page_to_image(raw)
                if image is None:
                    st.error("Install pymupdf for PDF support.")
                    st.stop()
            else:
                image = Image.open(io.BytesIO(raw)).convert("RGB")
            image_label = uploaded.name

    # ── Source: random gallery ───────────────────────────────────────
    elif source == "Random from gallery":
        if gallery_df.empty:
            st.warning("No local gallery found. Check `paths.splits_csv` / `paths.image_root` in config.")
        else:
            cA, cB = st.columns([1, 1])
            with cA:
                if st.button("Pick random", type="primary"):
                    st.session_state["random_pick"] = gallery_df.sample(
                        1, random_state=random.randint(0, 1_000_000)
                    ).iloc[0].to_dict()
            with cB:
                st.metric("Gallery size", len(gallery_df))
            pick = st.session_state.get("random_pick")
            if pick:
                chosen_id = pick["img_id"]
                image = Image.open(pick["img_path"]).convert("RGB")
                image_label = f"{pick['img_id']}  ·  {pick.get('image_type','')}"
                gt_html = pick.get("html", "") or ""

    # ── Source: browse gallery ───────────────────────────────────────
    elif source == "Browse gallery":
        if gallery_df.empty:
            st.warning("No local gallery found.")
        else:
            sub = gallery_df
            search = st.text_input("Search by img_id (substring)", key="browse_search").strip().lower()
            if search:
                sub = sub[sub["img_id"].str.lower().str.contains(search)]
            ids = sub["img_id"].tolist()
            n_show = min(60, len(ids))
            st.caption(f"{len(sub)} matches, showing first {n_show}.")
            if ids:
                chosen_id = st.selectbox("Pick image", ids[:200], key="browse_pick")
                row = sub[sub["img_id"] == chosen_id].iloc[0].to_dict()
                image = Image.open(row["img_path"]).convert("RGB")
                image_label = f"{row['img_id']}  ·  {row.get('image_type','')}"
                gt_html = row.get("html", "") or ""

    # ── Display + run ──────────────────────────────────────────────────
    if image is not None:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(image, caption=image_label, use_container_width=True)
        with c2:
            st.markdown(f"**{image_label}**" if image_label else "")
            if gt_html:
                with st.expander("Ground-truth HTML"):
                    st.code(gt_html, language="html", line_numbers=False)
                st.markdown("**Rendered GT:**")
                st.markdown(gt_html, unsafe_allow_html=True)

        run_disabled = not selected_models
        run_label = ", ".join(_disp(m) for m in selected_models) if selected_models else "—"
        if st.button(f"Run {len(selected_models)} model(s):  {run_label}",
                      type="primary", disabled=run_disabled):
            with st.spinner("Running inference …"):
                t0 = time.perf_counter()
                results = run_models(image, selected_models, parallel=True)
                total = time.perf_counter() - t0
            st.session_state["last_results"] = results
            st.success(f"Done in {total:.2f}s")

        results = st.session_state.get("last_results", {})
        if results:
            st.markdown("---")
            cols = st.columns(len(results))
            for c, (name, r) in zip(cols, results.items()):
                with c:
                    has_html = bool(r.get("html"))
                    card_cls = "model-card" if has_html else "model-card no-output"
                    st.markdown(
                        f"<div class='{card_cls}'><b>{_disp(name)}</b> "
                        f"<span class='small-caption'>· {r.get('time_s', 0):.2f}s</span></div>",
                        unsafe_allow_html=True,
                    )
                    if has_html:
                        st.markdown(r["html"], unsafe_allow_html=True)
                    else:
                        err = r.get("metadata", {}).get("error", "")
                        st.warning(f"(empty output) {err}")

                    # ── HTML → DataFrame + CSV download ─────────────────
                    if has_html:
                        parsed = robust_html_to_dataframe(r["html"])
                        df = parsed.get("df")
                        if df is not None and not df.empty:
                            status_msg = {
                                "success_clean":              "parsed cleanly",
                                "success_fixed_misalignment": f"parsed (padded "
                                    f"{parsed['max_cols_before']-parsed['min_cols_before']} cols)",
                            }.get(parsed["status"], parsed["status"])
                            st.caption(
                                f"{status_msg} · {parsed['num_rows']} rows × "
                                f"{parsed['num_cols']} cols"
                            )
                            with st.expander("DataFrame preview"):
                                st.dataframe(df, use_container_width=True,
                                              height=min(320, 36 + 30 * len(df)))
                            csv_bytes = df.to_csv(index=False).encode("utf-8")
                            fname = (
                                f"{image_label.replace(' ', '_').replace('·','_') or 'table'}"
                                f"__{name}.csv"
                            )
                            st.download_button(
                                label=f"Download {_disp(name)} as CSV",
                                data=csv_bytes,
                                file_name=fname,
                                mime="text/csv",
                                key=f"dl_csv_{name}",
                                use_container_width=True,
                            )
                        else:
                            st.caption(
                                f"parse failed: "
                                f"{parsed.get('error_message') or parsed['status']}"
                            )
                    with st.expander("Raw HTML"):
                        st.code(r.get("html", ""), language="html")
                    with st.expander("Metadata"):
                        st.json(r.get("metadata", {}))

            # ── Side-by-side comparison summary ────────────────────────
            st.markdown("---")
            st.subheader("Side-by-side")
            summary_rows = []
            for name, r in results.items():
                meta = r.get("metadata", {}) or {}
                # Derive structural counts from the predicted HTML so the row
                # populates for every model, not just the ones whose metadata
                # happens to carry these keys (UniTable does not).
                stats = html_stats(r.get("html", ""))
                row = {
                    "Model":   _disp(name),
                    "Time (s)": round(float(r.get("time_s", 0)), 2),
                    "Rows":    stats["n_rows"],
                    "Cols":    stats["n_cols"],
                    "Cells":   stats["n_cells"],
                    "Headers": stats["header_rows"],
                    "Spans":   stats["n_spans"],
                    "Mode":    meta.get("mode", meta.get("model", "")),
                }
                if gt_html.strip():
                    m = compute_metrics(r["html"], gt_html)
                    row["TEDS"]   = round(float(m.get("TEDS", 0)), 3)
                    row["TEDS-S"] = round(float(m.get("TEDS-S", 0)), 3)
                summary_rows.append(row)
            st.dataframe(pd.DataFrame(summary_rows).set_index("Model"),
                          use_container_width=True)

            if gt_html.strip():
                st.markdown("---")
                st.subheader("Metrics vs ground truth")
                metrics_by_model = {
                    _disp(name): compute_metrics(r["html"], gt_html)
                    for name, r in results.items()
                }
                st.dataframe(pd.DataFrame(metrics_by_model).T.round(4),
                              use_container_width=True)
                st.plotly_chart(metric_bars(metrics_by_model), use_container_width=True)


# ─────────────────────────────────────── Tab 2: Benchmark ───────────────
with TAB_BENCH:
    st.header("Benchmark dashboard")
    st.caption("Pre-computed scores on the 1,300-sample FinTabNet evaluation set.")

    bench = _load_benchmark(cfg.get("paths", {}).get("benchmark_csv", ""))
    if bench.empty:
        st.info("No benchmark file at `paths.benchmark_csv`. Skipping.")
    else:
        # Detect available model columns
        model_cols = [c for c in bench.columns if c.endswith("_teds")]
        model_names = [c[:-5] for c in model_cols]

        # Filter
        types = ["all"] + sorted(bench["image_type"].dropna().unique().tolist())
        cT, cM = st.columns([1, 2])
        with cT:
            sel_type = st.selectbox("Image type", types, key="bench_type")
        with cM:
            sel_models = st.multiselect("Models", model_names, default=model_names,
                                         key="bench_models")
        if sel_type != "all":
            view = bench[bench["image_type"] == sel_type]
        else:
            view = bench

        # KPI strip
        cols = st.columns(max(1, len(sel_models) + 1))
        with cols[0]:
            st.markdown(
                f"<div class='kpi-card'>"
                f"<div class='kpi-label'>Samples</div>"
                f"<div class='kpi-value maroon'>{len(view)}</div>"
                f"</div>", unsafe_allow_html=True,
            )
        for i, m in enumerate(sel_models, start=1):
            col = f"{m}_teds"
            if col not in view: continue
            mean_t = view[col].mean()
            med_t = view[col].median()
            color = "green" if mean_t >= 0.5 else ("amber" if mean_t >= 0.3 else "red")
            with cols[i]:
                st.markdown(
                    f"<div class='kpi-card'>"
                    f"<div class='kpi-label'>{m} mean TEDS</div>"
                    f"<div class='kpi-value {color}'>{mean_t:.3f}</div>"
                    f"<div class='small-caption'>median {med_t:.3f}</div>"
                    f"</div>", unsafe_allow_html=True,
                )

        # By image-type breakdown
        st.markdown("##### Mean TEDS by image_type")
        rows = []
        for it in sorted(bench["image_type"].dropna().unique().tolist()):
            sub = bench[bench["image_type"] == it]
            row = {"image_type": it, "N": len(sub)}
            for m in sel_models:
                col = f"{m}_teds"
                if col in sub: row[m] = round(sub[col].mean(), 4)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Per-sample distribution
        st.markdown("##### TEDS distribution per model")
        import plotly.graph_objs as go
        fig = go.Figure()
        for m in sel_models:
            col = f"{m}_teds"
            if col in view:
                fig.add_trace(go.Histogram(
                    x=view[col], name=m, opacity=0.7, nbinsx=30,
                ))
        fig.update_layout(barmode="overlay", height=350,
                          margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="TEDS", yaxis_title="samples")
        st.plotly_chart(fig, use_container_width=True)

        # Failure-mode summary
        st.markdown("##### Failure mode flags")
        flag_cols = [c for c in bench.columns
                     if c.startswith(("florence2_", "unitable_", "spartan_", "pytesseract_"))
                     and c.endswith(("header_collapse", "colspan_ignored", "rowspan_ignored",
                                    "table_not_found", "cell_count_wrong", "content_mismatch"))]
        if flag_cols:
            fdf = view[flag_cols].mean().reset_index()
            fdf.columns = ["flag", "rate"]
            fdf["rate"] = (fdf["rate"] * 100).round(1)
            st.dataframe(fdf, use_container_width=True)

        # Worst-failure browser
        st.markdown("##### Worst failures")
        teds_for_worst = sel_models[0] if sel_models else (model_names[0] if model_names else None)
        if teds_for_worst:
            worst = view.nsmallest(20, f"{teds_for_worst}_teds")[
                ["image_id", "image_type"] + [f"{m}_teds" for m in sel_models if f"{m}_teds" in view]
            ]
            st.dataframe(worst, use_container_width=True)
            chosen = st.selectbox("Open worst-failure image",
                                   ["—"] + worst["image_id"].tolist(),
                                   key="bench_open")
            if chosen and chosen != "—":
                gallery_df = _load_gallery_index(
                    cfg.get("paths", {}).get("splits_csv", ""),
                    cfg.get("paths", {}).get("image_root", ""),
                )
                # try both id formats (benchmark uses fintabnet_*, local uses fintab_*)
                row = gallery_df[gallery_df["img_id"] == chosen]
                if row.empty:
                    short = chosen.replace("fintabnet_", "fintab_")
                    row = gallery_df[gallery_df["img_id"] == short]
                if not row.empty:
                    r = row.iloc[0]
                    cI, cG = st.columns([1, 1])
                    with cI: st.image(Image.open(r["img_path"]), caption=r["img_id"])
                    with cG:
                        st.markdown("**Ground truth (rendered):**")
                        st.markdown(r.get("html", ""), unsafe_allow_html=True)
                else:
                    st.info(f"No local image for {chosen}.")


# ─────────────────────────────────────── Tab 3: Batch Eval ─────────────
with TAB_BATCH:
    st.header("Batch evaluation")
    st.markdown("Run all loaded models over a slice of a splits CSV (or the "
                "auto-detected local CSV).")

    default_csv = cfg.get("paths", {}).get("splits_csv", "")
    default_root = cfg.get("paths", {}).get("image_root", "")

    bcsv = st.file_uploader("Splits CSV (img_id, img_path, html, image_type)",
                              type=["csv"], key="batch_csv")
    csv_path = None
    if bcsv is None and default_csv and Path(default_csv).exists():
        st.caption(f"Using default CSV: `{default_csv}`")
        csv_path = default_csv
    image_root = st.text_input("Image root", value=default_root)
    limit = st.number_input("Sample limit (0 = all rows)", min_value=0, value=20)

    if (bcsv is not None or csv_path) and st.button("Run batch", type="primary",
                                                       key="run_batch"):
        df = pd.read_csv(bcsv) if bcsv is not None else pd.read_csv(csv_path)
        if limit > 0:
            df = df.sample(min(limit, len(df)), random_state=0).reset_index(drop=True)
        if image_root and "img_path" in df:
            base = Path(image_root)
            df["img_path"] = df.apply(
                lambda r: str(base / f"{r['img_id']}.png") if not r.get("img_path") else r["img_path"],
                axis=1,
            )
        prog = st.progress(0.0, text="0 / 0")
        rows = []
        active_models = {n: e for n, e in extractors.items()
                          if n in selected_models and e is not None}
        total = len(df) * max(1, len(active_models))
        seen = 0
        for _, r in df.iterrows():
            try:
                img = Image.open(r["img_path"]).convert("RGB")
            except Exception:
                continue
            for name, ext in active_models.items():
                try:
                    out = ext.predict(img)
                    rows.append({
                        "img_id":     r["img_id"],
                        "image_type": r.get("image_type", "unknown"),
                        "model":      name,
                        "teds":       compute_teds(out["html"], r["html"]),
                        "teds_s":     compute_teds_s(out["html"], r["html"]),
                        "time_s":     out["time_s"],
                        "pred_html":  out["html"],
                        "gt_html":    r["html"],
                        "img_path":   r["img_path"],
                    })
                except Exception as exc:
                    rows.append({"img_id": r["img_id"], "model": name,
                                 "teds": 0.0, "teds_s": 0.0, "time_s": 0.0,
                                 "pred_html": "", "gt_html": r["html"],
                                 "image_type": r.get("image_type", "unknown"),
                                 "img_path": r.get("img_path", ""),
                                 "error": str(exc)})
                seen += 1
                prog.progress(seen / max(1, total), text=f"{seen} / {total}")
        df_res = pd.DataFrame(rows)
        st.session_state["batch_results"] = df_res

    if "batch_results" in st.session_state:
        df_res = st.session_state["batch_results"]
        st.subheader("Per-sample scores")
        st.dataframe(df_res.drop(columns=["pred_html", "gt_html"], errors="ignore"),
                      use_container_width=True, height=300)
        st.subheader("Summary")
        st.dataframe(summary_table(df_res), use_container_width=True)
        st.plotly_chart(teds_distribution(df_res), use_container_width=True)
        st.download_button(
            "Download results CSV",
            data=df_res.drop(columns=["pred_html", "gt_html"], errors="ignore")
                       .to_csv(index=False).encode(),
            file_name="tablesight_batch_results.csv", mime="text/csv",
        )


# ─────────────────────────────────────── Tab 4: Failure Analysis ───────
with TAB_FAIL:
    st.header("Failure analysis")
    df_res = st.session_state.get("batch_results")
    if df_res is None or df_res.empty:
        st.info("Run a batch first (Tab: Batch Eval).")
    else:
        threshold = st.slider("TEDS failure threshold", 0.0, 1.0, 0.5, 0.05)
        fails = df_res[df_res["teds"] < threshold].copy()
        if fails.empty:
            st.success("No failures below threshold.")
        else:
            fails["category"] = fails.apply(
                lambda r: classify_failure(r["pred_html"], r["gt_html"], r["teds"]),
                axis=1,
            )
            counts_by_model = {
                m: dict(sub["category"].value_counts())
                for m, sub in fails.groupby("model")
            }
            st.plotly_chart(failure_pies(counts_by_model), use_container_width=True)

            st.markdown("### Side-by-side diffs")
            n_show = st.number_input("Cases to display", min_value=1, max_value=50, value=5)
            sample = fails.sort_values("teds").head(n_show)
            for _, r in sample.iterrows():
                with st.expander(f"{r['img_id']}  ·  {_disp(r['model'])}  ·  "
                                  f"TEDS={r['teds']:.3f}  ·  [{r['category']}]"):
                    try:
                        st.image(Image.open(r.get("img_path", "")), width=380)
                    except Exception: pass
                    differ = difflib.HtmlDiff(wrapcolumn=80)
                    diff_html = differ.make_table(
                        (r["pred_html"] or "").splitlines() or [""],
                        (r["gt_html"] or "").splitlines() or [""],
                        fromdesc="Prediction", todesc="Ground Truth",
                    )
                    st.components.v1.html(diff_html, height=300, scrolling=True)


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center; padding:18px 0 8px; margin-top:32px;"
    " border-top:1px solid var(--border); color:var(--muted);"
    " font-family: \"Source Serif 4\", \"Times New Roman\", serif; font-size:12px;'>"
    "TableSight  ·  Hierarchical Table Extraction  ·  "
    "<span style='color:var(--maroon); font-weight:600;'>The University of Chicago</span>"
    "</div>",
    unsafe_allow_html=True,
)
