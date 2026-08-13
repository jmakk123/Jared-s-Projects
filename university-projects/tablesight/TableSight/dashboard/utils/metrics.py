"""
TEDS (Tree-Edit Distance Similarity) scoring for HTML tables.

Same APTED-based TEDS scorer used in sprint-2 so dashboard scores are
(Cell 12 — `compute_teds`, `html_to_tree`, `TableTree`, `_TEDSConfig`).

Includes:
  - compute_teds(pred_html, gt_html, structure_only=False) -> float in [0,1]
  - compute_teds_s(pred_html, gt_html) -> structure-only TEDS
  - clean_html_structure(html) -> skeleton (content stripped, attributes preserved)
  - classify_failure(pred_html, gt_html, teds) -> failure category string
  - count_cells, count_spans helpers

Backend: APTED with custom node-rename cost (text + tag + colspan/rowspan).
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional

from apted import APTED, Config
from lxml import html as lhtml


# ── tree representation ──────────────────────────────────────────────────────


class TableTree:
    """Node in the table tree used for APTED edit-distance computation.

    Each node carries a tag, optional text, colspan / rowspan, and children.
    Equality / rename-cost are defined on (tag, content, colspan, rowspan)
    so structural and content differences both contribute to the distance.
    """

    __slots__ = ("tag", "content", "colspan", "rowspan", "children")

    def __init__(
        self,
        tag: str,
        content: str = "",
        colspan: int = 1,
        rowspan: int = 1,
        children: Optional[List["TableTree"]] = None,
    ):
        self.tag = tag
        self.content = content or ""
        self.colspan = colspan
        self.rowspan = rowspan
        self.children = children or []

    def __repr__(self) -> str:
        return (f"TableTree(tag={self.tag!r}, content={self.content!r}, "
                f"cs={self.colspan}, rs={self.rowspan}, "
                f"children={len(self.children)})")


class _TEDSConfig(Config):
    """APTED cost configuration for table-tree edit distance.

    insert / delete costs are 1 per node.
    rename cost is composed of:
        - 1 if tags differ (structural mismatch)
        - 1 if spans differ
        - normalised character-level edit distance between texts (∈ [0,1])
    """

    def __init__(self, structure_only: bool = False):
        self.structure_only = structure_only

    def rename(self, n1: TableTree, n2: TableTree) -> float:
        cost = 0.0
        if n1.tag != n2.tag:
            cost += 1.0
        if (n1.colspan != n2.colspan) or (n1.rowspan != n2.rowspan):
            cost += 1.0
        if not self.structure_only:
            cost += _char_edit_distance(n1.content, n2.content)
        return cost

    def children(self, node: TableTree) -> Iterable[TableTree]:
        return node.children


# ── helpers ──────────────────────────────────────────────────────────────────


def _char_edit_distance(a: str, b: str) -> float:
    """Normalised character-level Levenshtein distance ∈ [0,1].

    0 when strings are identical, 1 when fully different.
    Empty strings on both sides → 0.
    """
    a, b = (a or "").strip(), (b or "").strip()
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    # classic two-row DP
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        ca = a[i - 1]
        for j in range(1, n + 1):
            cb = b[j - 1]
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[n] / max(m, n)


def _node_text(el) -> str:
    """Direct text of an element, recursively flattened, whitespace collapsed."""
    parts = []
    for t in el.itertext():
        if t:
            parts.append(t)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _intspan(el, attr: str) -> int:
    v = el.get(attr)
    try:
        return max(1, int(v))
    except (TypeError, ValueError):
        return 1


def _build_node(el) -> TableTree:
    """Recursively convert an lxml element into a TableTree node."""
    tag = el.tag.lower() if isinstance(el.tag, str) else "?"
    if tag in ("td", "th"):
        node = TableTree(
            tag=tag,
            content=_node_text(el),
            colspan=_intspan(el, "colspan"),
            rowspan=_intspan(el, "rowspan"),
        )
        return node
    children = [_build_node(c) for c in el.iterchildren() if isinstance(c.tag, str)]
    return TableTree(tag=tag, children=children)


def html_to_tree(html_str: str) -> TableTree:
    """Parse an HTML table string into a TableTree.

    Tolerates fragments without <html><body>. If no <table> element is found
    the returned tree is a degenerate root node with no children, which
    yields TEDS = 0 against any non-empty ground truth.
    """
    if not html_str or not html_str.strip():
        return TableTree(tag="empty")
    try:
        doc = lhtml.fromstring(html_str)
    except Exception:
        return TableTree(tag="empty")
    tables = doc.xpath("//table")
    if not tables:
        # the input might itself be a <table>
        if doc.tag and doc.tag.lower() == "table":
            tables = [doc]
        else:
            return TableTree(tag="empty")
    return _build_node(tables[0])


# ── public scoring API ──────────────────────────────────────────────────────


def _tree_size(t: TableTree) -> int:
    return 1 + sum(_tree_size(c) for c in t.children)


def compute_teds(
    pred_html: str,
    gt_html: str,
    structure_only: bool = False,
) -> float:
    """Tree-Edit Distance Similarity ∈ [0, 1]. 1 = perfect match."""
    t_pred = html_to_tree(pred_html)
    t_gt = html_to_tree(gt_html)
    if t_pred.tag == "empty" and t_gt.tag == "empty":
        return 1.0
    if t_pred.tag == "empty" or t_gt.tag == "empty":
        return 0.0
    ed = APTED(t_pred, t_gt, _TEDSConfig(structure_only=structure_only)).compute_edit_distance()
    max_size = max(_tree_size(t_pred), _tree_size(t_gt))
    if max_size == 0:
        return 1.0
    return max(0.0, 1.0 - ed / max_size)


def compute_teds_s(pred_html: str, gt_html: str) -> float:
    """Structure-only TEDS — text content is ignored, only tags + spans matter."""
    return compute_teds(pred_html, gt_html, structure_only=True)


# ── structural utilities ────────────────────────────────────────────────────


def clean_html_structure(html_str: str) -> str:
    """Strip cell text but keep structural attributes (rowspan / colspan).

    Used by the Florence pipeline so that the structure-only TEDS doesn't
    waste tokens on content. Returns a minimal `<table>...</table>` string.
    """
    if not html_str:
        return ""
    try:
        doc = lhtml.fromstring(html_str)
    except Exception:
        return ""
    tables = doc.xpath("//table")
    if not tables:
        if doc.tag and doc.tag.lower() == "table":
            tables = [doc]
        else:
            return ""
    tbl = tables[0]

    for cell in tbl.xpath(".//td|.//th"):
        for child in list(cell):
            cell.remove(child)
        cell.text = ""
        # drop any non-structural attribute
        for attr in list(cell.attrib.keys()):
            if attr not in ("colspan", "rowspan"):
                del cell.attrib[attr]
    return lhtml.tostring(tbl, encoding="unicode")


def count_cells(html_str: str) -> int:
    try:
        doc = lhtml.fromstring(html_str or "")
        return len(doc.xpath("//td")) + len(doc.xpath("//th"))
    except Exception:
        return 0


def count_spans(html_str: str) -> dict:
    """Return {n_colspan: int, n_rowspan: int, max_cs: int, max_rs: int}."""
    out = {"n_colspan": 0, "n_rowspan": 0, "max_cs": 1, "max_rs": 1}
    try:
        doc = lhtml.fromstring(html_str or "")
    except Exception:
        return out
    for cell in doc.xpath("//td|//th"):
        cs = _intspan(cell, "colspan")
        rs = _intspan(cell, "rowspan")
        if cs > 1:
            out["n_colspan"] += 1
            out["max_cs"] = max(out["max_cs"], cs)
        if rs > 1:
            out["n_rowspan"] += 1
            out["max_rs"] = max(out["max_rs"], rs)
    return out


def count_header_rows(html_str: str) -> int:
    """Approximate header rows: any <tr> entirely composed of <th>, OR <tr> inside <thead>."""
    try:
        doc = lhtml.fromstring(html_str or "")
    except Exception:
        return 0
    n = len(doc.xpath("//thead/tr"))
    for tr in doc.xpath("//tr"):
        if any(c.tag == "thead" for c in tr.iterancestors()):
            continue   # already counted
        cells = [c for c in tr.iterchildren() if c.tag in ("td", "th")]
        if cells and all(c.tag == "th" for c in cells):
            n += 1
    return n


# ── failure classification ──────────────────────────────────────────────────


def classify_failure(pred_html: str, gt_html: str, teds: Optional[float] = None) -> str:
    """Categorise a failure mode for samples with low TEDS.

    Failure-mode categories:
      - table_not_found     : prediction has no <table>
      - header_collapse     : prediction has 0-1 header rows where GT has 2+
      - colspan_ignored     : GT has spanning columns, prediction has none
      - rowspan_ignored     : GT has spanning rows, prediction has none
      - cell_count_wrong    : |Δ cells| / gt_cells > 0.25
      - content_mismatch    : structure matches reasonably but text differs
      - ok                  : TEDS ≥ 0.5  (caller usually filters before invocation)
    """
    if teds is not None and teds >= 0.5:
        return "ok"

    if "<table" not in (pred_html or "").lower():
        return "table_not_found"

    gt_spans = count_spans(gt_html)
    pr_spans = count_spans(pred_html)
    if gt_spans["n_colspan"] >= 1 and pr_spans["n_colspan"] == 0:
        return "colspan_ignored"
    if gt_spans["n_rowspan"] >= 1 and pr_spans["n_rowspan"] == 0:
        return "rowspan_ignored"

    gt_hdrs = count_header_rows(gt_html)
    pr_hdrs = count_header_rows(pred_html)
    if gt_hdrs >= 2 and pr_hdrs <= 1:
        return "header_collapse"

    gt_n = count_cells(gt_html)
    pr_n = count_cells(pred_html)
    if gt_n and abs(pr_n - gt_n) / gt_n > 0.25:
        return "cell_count_wrong"

    return "content_mismatch"


__all__ = [
    "TableTree",
    "compute_teds",
    "compute_teds_s",
    "clean_html_structure",
    "classify_failure",
    "count_cells",
    "count_spans",
    "count_header_rows",
    "html_to_tree",
]
