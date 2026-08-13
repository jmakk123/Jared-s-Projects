"""
SOTA-Defeating Synthetic Table Generator
=========================================
Modes:
  - row_heavy  : BOM-style deep rowspan trees (Level 3-6)
  - col_heavy  : Multi-level hierarchical colspan headers
  - cross_tab  : Combined complex headers + deep body rowspans

Style Engine:
  - Professional color palettes, zebra stripes
  - Borders: Full Grid / Horizontal Only / Borderless (10%)
  - Typography: Serif / Sans-Serif / Monospace
  - Per-column text-align randomization
"""

import random
import dataclasses
from typing import List, Optional, Dict, Any


# ---------------------------------------------------------------------------
# Content Library
# ---------------------------------------------------------------------------

class ContentLibrary:
    """Provides varied text lengths; Long strings trigger CSS word-wrap."""

    SHORT: List[str] = [
        "Part ID", "Qty", "Rev", "USD", "kg", "Status", "N/A", "OK",
        "SKU", "Ref", "Unit", "Tax", "% Chg", "YTD", "Q1", "Q2", "Q3", "Q4",
    ]
    MEDIUM: List[str] = [
        "Brushless DC Motor", "Electronic Speed Controller",
        "Carbon Fiber Propeller", "Main Flight Controller",
        "Thermal Management Unit", "Precision Ball Bearing",
        "Operating Cash Flow", "Net Revenue Growth",
        "Customer Acquisition Cost", "Gross Profit Margin",
        "Inventory Turnover Ratio", "Return on Equity",
        "Capital Expenditure", "Depreciation & Amortization",
    ]
    LONG: List[str] = [
        "High-performance lithium polymer battery with integrated thermal management and safety sensors.",
        "Ultra-lightweight aerospace grade aluminum alloy chassis for maximum structural integrity.",
        "Precision-engineered landing gear assembly with shock absorption and quick-release mechanism.",
        "Advanced telemetry module supporting long-range encrypted communication and real-time GPS tracking.",
        "Multi-axis inertial measurement unit with temperature-compensated MEMS accelerometer and gyroscope.",
        "Consolidated net revenues attributable to recurring subscription services excluding one-time items.",
        "Year-over-year percentage change in adjusted EBITDA before non-cash stock compensation charges.",
        "Total addressable market estimate based on bottom-up analysis of customer segment penetration rates.",
        "Annualized run-rate contract value including committed expansion bookings and renewal agreements.",
        "Weighted average cost of capital used as discount rate for discounted cash flow valuation model.",
    ]

    # Domain-themed header label pools
    FINANCIAL_HEADERS: List[str] = [
        "Revenue", "EBITDA", "Net Income", "Free Cash Flow", "EPS",
        "Gross Margin", "Op. Expense", "D&A", "CapEx", "Working Capital",
    ]
    BOM_HEADERS: List[str] = [
        "Assembly", "Sub-Assembly", "Component", "Part Number",
        "Description", "Material", "Supplier", "UOM", "Lead Time",
    ]
    STAT_HEADERS: List[str] = [
        "Mean", "Std Dev", "Min", "Max", "Median", "95th %ile", "p-value", "N",
    ]

    # Data column headers for row_heavy tables (right-side metric columns)
    DATA_COL_HEADERS: List[str] = [
        "Unit Cost", "Lead Time", "Supplier Code", "Safety Stock", "Reorder Qty",
        "MOQ", "Weight (kg)", "List Price", "Discount %", "Net Price",
        "On Hand", "On Order", "Cycle Time", "Scrap Rate", "Revision",
        "Approved By", "Last Updated", "Compliance", "Tolerance", "Yield %",
        "Revenue", "Variance", "Budget", "Actual", "Forecast",
    ]

    # Row index labels for col_heavy tables (far-left identifying column)
    INDEX_LABELS_SHORT: List[str] = [
        "Department A", "Department B", "Region: North", "Region: South",
        "Q1 Result", "Q2 Result", "Q3 Result", "Q4 Result",
        "SKU-1001", "SKU-2047", "SKU-9920", "SKU-3311",
        "Product Line A", "Product Line B", "Segment: SMB", "Segment: Enterprise",
        "Plant 01", "Plant 02", "Warehouse A", "Warehouse B",
        "FY2022", "FY2023", "FY2024", "H1 2024", "H2 2024",
        "Group Alpha", "Group Beta", "Group Gamma", "Tier 1", "Tier 2",
    ]
    INDEX_LABELS_LONG: List[str] = [
        "North American Commercial Aviation — Recurring Maintenance Services",
        "Enterprise Software Licensing — Multi-Year Subscription Agreements",
        "Asia-Pacific Retail Distribution — Direct-to-Consumer Channel",
        "Research & Development — Advanced Materials Innovation Program",
        "Government Contracts — Federal Infrastructure Modernization Initiative",
        "Emerging Markets — Sub-Saharan Africa Expansion Portfolio",
        "Strategic Partnerships — Joint Venture with Tier-1 OEM Suppliers",
        "Digital Transformation — Cloud Migration and Platform Consolidation",
    ]

    @staticmethod
    def get_text(weights: Optional[List[float]] = None) -> str:
        w = weights or [0.45, 0.35, 0.20]
        choice = random.choices(["S", "M", "L"], weights=w)[0]
        if choice == "S":
            return random.choice(ContentLibrary.SHORT)
        if choice == "M":
            return random.choice(ContentLibrary.MEDIUM)
        return random.choice(ContentLibrary.LONG)

    @staticmethod
    def get_header_label(domain: str = "generic") -> str:
        if domain == "financial":
            return random.choice(ContentLibrary.FINANCIAL_HEADERS)
        if domain == "bom":
            return random.choice(ContentLibrary.BOM_HEADERS)
        if domain == "stat":
            return random.choice(ContentLibrary.STAT_HEADERS)
        return ContentLibrary.get_text(weights=[0.6, 0.35, 0.05])

    @staticmethod
    def get_data_col_header() -> str:
        """Named header for a right-side data column in row_heavy tables."""
        return random.choice(ContentLibrary.DATA_COL_HEADERS)

    @staticmethod
    def get_index_label() -> str:
        """Row label for the far-left index column in col_heavy tables."""
        # 25% chance of a long label to trigger text-wrapping
        if random.random() < 0.25:
            return random.choice(ContentLibrary.INDEX_LABELS_LONG)
        return random.choice(ContentLibrary.INDEX_LABELS_SHORT)

    @staticmethod
    def get_data_value() -> str:
        """Numeric / status data cell values — never empty."""
        r = random.random()
        if r < 0.30:
            return f"{random.uniform(0, 9999):.2f}"
        if r < 0.55:
            return str(random.randint(0, 10000))
        if r < 0.70:
            return random.choice(["Active", "Pending", "Closed", "N/A", "TBD", "—"])
        if r < 0.85:
            return f"{random.uniform(-50, 150):.1f}%"
        return ContentLibrary.get_text(weights=[0.7, 0.25, 0.05])


def _safe_name(name: str, fallback_domain: str = "bom") -> str:
    """Return name if non-empty, otherwise draw a fallback label."""
    s = (name or "").strip()
    return s if s else ContentLibrary.get_header_label(fallback_domain)


# ---------------------------------------------------------------------------
# Style Engine
# ---------------------------------------------------------------------------

# Color palettes: (th_bg, th_text, accent_stripe)
# Standard mode — dark header
COLOR_PALETTES_STANDARD = [
    ("#1a3a5c", "#ffffff", "#eaf1fb"),   # Navy corporate
    ("#2d6a4f", "#ffffff", "#eaf4ee"),   # Forest green
    ("#6b3fa0", "#ffffff", "#f3eeff"),   # Purple analytics
    ("#b5451b", "#ffffff", "#fff1ec"),   # Burnt orange
    ("#34495e", "#ffffff", "#f0f3f4"),   # Slate
    ("#0d3b66", "#faf0ca", "#fff8e7"),   # Yale blue/cream
    ("#1a1a2e", "#e0e0e0", "#f5f5fa"),   # Deep navy
    ("#3a3a3a", "#f5f5f5", "#f9f9f9"),   # Charcoal
]

# Inverted mode — light/white header, colored body stripe
COLOR_PALETTES_INVERTED = [
    ("#f5f5f5", "#1a3a5c", "#ddeeff"),   # White header, blue body
    ("#f0fff4", "#2d6a4f", "#d4f1e0"),   # Mint header, green body
    ("#fff9e6", "#7a4f00", "#fde68a"),   # Cream header, amber body
    ("#fdf4ff", "#6b3fa0", "#e9d5ff"),   # Lavender header, purple body
    ("#ffffff", "#111111", "#eff6ff"),   # White header, blue-tint body
]

# B&W grayscale palettes: (th_bg, th_text, accent_stripe)
BW_PALETTES = [
    ("#1a1a1a", "#ffffff", "#f0f0f0"),   # Black header, white body
    ("#444444", "#ffffff", "#f5f5f5"),   # Dark gray header
    ("#888888", "#ffffff", "#f9f9f9"),   # Medium gray header
    ("#cccccc", "#111111", "#f2f2f2"),   # Light gray header, dark text
    ("#ffffff", "#000000", "#eeeeee"),   # White header, black text
    ("#000000", "#eeeeee", "#f8f8f8"),   # Pure black header
    ("#333333", "#f0f0f0", "#fafafa"),   # Soft black header
]

# Style mode weights: bw=40%, standard=25%, inverted=15%, flat=10%, minimal=10%
STYLE_MODES = ["bw", "standard", "inverted", "flat", "minimal"]
STYLE_WEIGHTS = [0.40, 0.25, 0.15, 0.10, 0.10]

FONT_STACKS = [
    "'Helvetica Neue', Helvetica, Arial, sans-serif",
    "'Georgia', 'Times New Roman', Times, serif",
    "'Courier New', Courier, monospace",
    "Verdana, Geneva, Tahoma, sans-serif",
    "'Trebuchet MS', 'Lucida Sans Unicode', sans-serif",
    "'Palatino Linotype', 'Book Antiqua', Palatino, serif",
]


@dataclasses.dataclass
class StyleConfig:
    palette: tuple
    font: str
    border_style: str       # "full" | "horizontal" | "none"
    padding: int
    col_alignments: List[str]
    table_width_pct: int
    zebra: bool
    font_size: int
    style_mode: str         # "bw" | "standard" | "inverted" | "flat" | "minimal"


class StyleEngine:
    """Generates randomized CSS across 5 style modes to defeat pattern-matching."""

    @staticmethod
    def generate(num_cols: int = 4) -> StyleConfig:
        style_mode = random.choices(STYLE_MODES, weights=STYLE_WEIGHTS)[0]
        border_style = random.choices(
            ["full", "horizontal", "none"], weights=[0.60, 0.30, 0.10]
        )[0]
        alignments = [
            random.choice(["left", "center", "right"]) for _ in range(num_cols)
        ]

        if style_mode == "bw":
            palette = random.choice(BW_PALETTES)
        elif style_mode == "inverted":
            palette = random.choice(COLOR_PALETTES_INVERTED)
        elif style_mode in ("flat", "minimal"):
            # Flat/minimal: header same bg as body (white or very light)
            palette = ("#ffffff", "#111111", "#f5f5f5")
        else:  # standard
            palette = random.choice(COLOR_PALETTES_STANDARD)

        return StyleConfig(
            palette=palette,
            font=random.choice(FONT_STACKS),
            border_style=border_style,
            padding=random.randint(5, 14),
            col_alignments=alignments,
            table_width_pct=random.randint(80, 100),
            zebra=random.random() > 0.40,
            font_size=random.randint(11, 14),
            style_mode=style_mode,
        )

    @staticmethod
    def to_css(cfg: StyleConfig) -> str:
        th_bg, th_text, stripe = cfg.palette
        p = cfg.padding
        fs = cfg.font_size
        mode = cfg.style_mode

        # Border definitions
        if cfg.border_style == "full":
            b_color = "#999999" if mode == "bw" else "#cccccc"
            td_border = f"border: 1px solid {b_color};"
            th_border = f"border: 1px solid {b_color};"
            tbl_border = "border-collapse: collapse;"
        elif cfg.border_style == "horizontal":
            b_color = "#888888" if mode == "bw" else "#dddddd"
            sep_color = "#555555" if mode == "bw" else "#aaaaaa"
            td_border = f"border-top: 1px solid {b_color}; border-bottom: 1px solid {b_color};"
            th_border = f"border-bottom: 2px solid {sep_color};"
            tbl_border = "border-collapse: collapse;"
        else:  # none
            sep_color = "#555555" if mode == "bw" else "#888888"
            td_border = "border: none;"
            th_border = f"border-bottom: 2px solid {sep_color};"
            tbl_border = "border-collapse: collapse;"

        zebra_css = (
            f"tbody tr:nth-child(even) {{ background-color: {stripe}; }}"
            if cfg.zebra else ""
        )

        # Mode-specific th overrides
        if mode == "flat":
            # Header blends with body — same bg, just bold text + separator line
            th_extra = f"""
        th {{
            background-color: {th_bg};
            color: {th_text};
            font-weight: 700;
            border-bottom: 2px solid #888888;
            white-space: nowrap;
        }}"""
        elif mode == "minimal":
            # No background on headers at all, just italic bold
            th_extra = """
        th {
            background-color: transparent;
            color: #111111;
            font-weight: 700;
            font-style: italic;
            border-bottom: 1px solid #bbbbbb;
            white-space: nowrap;
        }"""
        elif mode == "inverted":
            # Header is light, use top-border accent instead of bg dominance
            th_extra = f"""
        th {{
            background-color: {th_bg};
            color: {th_text};
            font-weight: 600;
            border-top: 3px solid {stripe};
            white-space: nowrap;
        }}"""
        else:  # bw or standard
            th_extra = f"""
        th {{
            background-color: {th_bg};
            color: {th_text};
            font-weight: 600;
            white-space: nowrap;
        }}"""

        # Body bg: bw always white
        body_bg = "#ffffff"

        return f"""
        body {{ background-color: {body_bg}; margin: 24px; }}
        table {{
            font-family: {cfg.font};
            font-size: {fs}px;
            {tbl_border}
            width: {cfg.table_width_pct}%;
        }}
        th, td {{
            {td_border}
            padding: {p}px {p + 2}px;
            word-wrap: break-word;
            max-width: 220px;
            vertical-align: top;
        }}
        {th_extra}
        {zebra_css}
        h2 {{
            font-family: {cfg.font};
            font-size: {fs + 4}px;
            color: #222222;
            margin-bottom: 8px;
        }}
        p.footnote {{
            font-family: {cfg.font};
            font-size: {fs - 2}px;
            color: #777777;
            margin-top: 6px;
        }}
        """


# ---------------------------------------------------------------------------
# Row-Heavy (BOM) Table Generator
# ---------------------------------------------------------------------------

class TableNode:
    def __init__(self, name: str, level: int = 0):
        self.name = name
        self.level = level
        self.children: List["TableNode"] = []
        self.span: int = 0


def _grow_tree(parent: TableNode, curr_depth: int, max_depth: int) -> None:
    if curr_depth >= max_depth:
        return
    # Deeper nodes have fewer children to keep table size manageable
    num_children = random.randint(1, max(1, 4 - curr_depth))
    for _ in range(num_children):
        child = TableNode(
            name=ContentLibrary.get_header_label("bom"),
            level=curr_depth + 1,
        )
        parent.children.append(child)
        _grow_tree(child, curr_depth + 1, max_depth)


def calculate_spans(node: TableNode) -> int:
    if not node.children:
        node.span = 1
        return 1
    node.span = sum(calculate_spans(c) for c in node.children)
    return node.span


def flatten_to_rows(node: TableNode, rows=None, prefix=None):
    """Convert tree to a list-of-lists (each inner list = one <tr>'s hierarchy cells)."""
    if rows is None:
        rows = []
    if prefix is None:
        prefix = []
    cell = {"name": node.name, "span": node.span}
    new_prefix = prefix + [cell]
    if not node.children:
        rows.append(new_prefix)
    else:
        for i, child in enumerate(node.children):
            flatten_to_rows(child, rows, new_prefix if i == 0 else [])
    return rows
def generate_row_heavy_table(depth: int = 4) -> Dict[str, Any]:
    """
    BOM-style deep rowspan table.
    - Hierarchy columns on the left (colspan-merged header) with deep rowspan nesting.
    - 2–5 named data columns on the right, each with a descriptive header.
    - No cell is left empty (fallback via _safe_name).
    """
    root = TableNode(name=_safe_name(ContentLibrary.get_header_label("bom")), level=0)
    _grow_tree(root, 0, depth)
    calculate_spans(root)
    rows_data = flatten_to_rows(root)

    # Compute the max number of hierarchy columns across all rows.
    # The BOM tree produces variable-width rows; the widest row defines
    # how many columns the hierarchy header must span.
    max_hier_cols = max(len(row) for row in rows_data) if rows_data else 1

    # 2–5 named data columns — drawn without replacement to avoid duplicates
    n_data_cols = random.randint(2, 5)
    header_pool = ContentLibrary.DATA_COL_HEADERS.copy()
    random.shuffle(header_pool)
    data_col_headers = header_pool[:n_data_cols]
    align_list = ["left"] + [random.choice(["left", "center", "right"]) for _ in range(n_data_cols)]

    # Hierarchy header spans ALL hierarchy columns — same color, single wide cell
    hierarchy_header = random.choice([
        "Component Hierarchy", "BOM Structure", "Part Breakdown",
        "Assembly Tree", "Item Hierarchy", "Product Structure",
    ])
    hier_cs = f' colspan="{max_hier_cols}"' if max_hier_cols > 1 else ""

    # Build header row: one wide hierarchy th + one th per data col
    header_cells = "".join(
        f'<th style="text-align:{align_list[i+1]}">{h}</th>'
        for i, h in enumerate(data_col_headers)
    )
    header_row = (
        f'<tr><th{hier_cs} style="text-align:left">{hierarchy_header}</th>'
        f'{header_cells}</tr>'
    )

    # Build body rows — ensure every hierarchy cell name is non-empty
    html_rows = []
    for row in rows_data:
        tr = "<tr>"
        for cell in row:
            span_val = cell["span"]
            rs = f' rowspan="{span_val}"' if span_val > 1 else ""
            label = _safe_name(cell["name"], "bom")
            tr += f'<td{rs} style="text-align:left"><strong>{label}</strong></td>'
        for i in range(n_data_cols):
            tr += f'<td style="text-align:{align_list[i+1]}">{ContentLibrary.get_data_value()}</td>'
        tr += "</tr>"
        html_rows.append(tr)

    table_html = (
        f'<table><thead>{header_row}</thead>'
        f'<tbody>{"".join(html_rows)}</tbody></table>'
    )
    return {
        "html": table_html,
        "n_cols": max_hier_cols + n_data_cols,
        "depth": depth,
    }


# ---------------------------------------------------------------------------
# Col-Heavy (Hierarchical Header) Table Generator
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class HeaderNode:
    label: str
    children: List["HeaderNode"] = dataclasses.field(default_factory=list)
    colspan: int = 1   # computed
    rowspan: int = 1   # computed (for leaf nodes spanning multiple header rows)


def _build_header_tree(levels: int, domain: str = "financial") -> List[HeaderNode]:
    """Build a list of top-level header nodes with `levels` of nesting."""
    # Number of top-level categories
    n_cats = random.randint(2, 4)
    roots = []
    for _ in range(n_cats):
        node = HeaderNode(label=ContentLibrary.get_header_label(domain))
        _fill_header_children(node, 1, levels, domain)
        roots.append(node)
    return roots


def _fill_header_children(
    node: HeaderNode, curr_level: int, max_levels: int, domain: str
) -> None:
    if curr_level >= max_levels:
        return
    n_children = random.randint(2, 3)
    for _ in range(n_children):
        child = HeaderNode(label=ContentLibrary.get_header_label(domain))
        _fill_header_children(child, curr_level + 1, max_levels, domain)
        node.children.append(child)


def _compute_header_spans(node: HeaderNode, max_depth: int, curr_depth: int = 0) -> None:
    if not node.children:
        node.colspan = 1
        node.rowspan = max_depth - curr_depth  # leaf spans down to fill all rows
    else:
        for c in node.children:
            _compute_header_spans(c, max_depth, curr_depth + 1)
        node.colspan = sum(c.colspan for c in node.children)
        node.rowspan = 1


def _get_header_depth(nodes: List[HeaderNode]) -> int:
    def depth(n: HeaderNode) -> int:
        if not n.children:
            return 1
        return 1 + max(depth(c) for c in n.children)
    return max(depth(n) for n in nodes)


def _render_thead_rows(roots: List[HeaderNode], total_depth: int) -> str:
    """Render multi-row <thead> with proper colspan/rowspan attributes."""
    # BFS level-by-level
    rows: List[List[HeaderNode]] = []
    current_level = roots
    for _ in range(total_depth):
        rows.append(current_level)
        next_level = []
        for node in current_level:
            next_level.extend(node.children)
        current_level = next_level

    thead_rows = []
    for row_nodes in rows:
        cells = ""
        for node in row_nodes:
            cs = f' colspan="{node.colspan}"' if node.colspan > 1 else ""
            rs = f' rowspan="{node.rowspan}"' if node.rowspan > 1 else ""
            cells += f"<th{cs}{rs}>{node.label}</th>"
        thead_rows.append(f"<tr>{cells}</tr>")
    return "<thead>" + "".join(thead_rows) + "</thead>"


def _count_leaf_cols(nodes: List[HeaderNode]) -> int:
    total = 0
    for n in nodes:
        if not n.children:
            total += 1
        else:
            total += _count_leaf_cols(n.children)
    return total


def generate_col_heavy_table(header_levels: int = 3) -> Dict[str, Any]:
    """
    Multi-level colspan hierarchical header table with a mandatory far-left
    index column providing a named label for every row.

    Structure:
      [Index Col Header (spans all header rows)] | [Complex multi-level headers ...]
      [Row label — short or long]                | [Data values ...]
    """
    domain = random.choice(["financial", "stat", "generic"])
    roots = _build_header_tree(header_levels, domain)
    total_depth = _get_header_depth(roots)
    for node in roots:
        _compute_header_spans(node, total_depth)
    n_data_cols = _count_leaf_cols(roots)

    # Index column header label — spans all header rows vertically
    index_col_header = random.choice([
        "Category", "Segment", "Region", "Period", "Product",
        "Department", "Channel", "SKU", "Group", "Entity",
    ])

    # Build thead: index cell + existing multi-level headers
    # We re-render manually so we can prepend the index <th rowspan=total_depth>
    bfs_levels: List[List[HeaderNode]] = []
    current = roots
    for _ in range(total_depth):
        bfs_levels.append(current)
        next_level: List[HeaderNode] = []
        for node in current:
            next_level.extend(node.children)
        current = next_level

    thead_rows = []
    for lvl_idx, level_nodes in enumerate(bfs_levels):
        cells = ""
        if lvl_idx == 0:
            cells += f'<th rowspan="{total_depth}" style="text-align:left">{index_col_header}</th>'
        for node in level_nodes:
            cs = f' colspan="{node.colspan}"' if node.colspan > 1 else ""
            rs = f' rowspan="{node.rowspan}"' if node.rowspan > 1 else ""
            cells += f"<th{cs}{rs}>{node.label}</th>"
        thead_rows.append(f"<tr>{cells}</tr>")
    thead_html = "<thead>" + "".join(thead_rows) + "</thead>"

    align_data = [random.choice(["left", "center", "right"]) for _ in range(n_data_cols)]

    # Body rows — index label + data cells
    n_rows = random.randint(4, 12)
    body_rows = []
    used_labels: set = set()
    for _ in range(n_rows):
        # Pick a unique index label where possible
        label = ContentLibrary.get_index_label()
        attempts = 0
        while label in used_labels and attempts < 10:
            label = ContentLibrary.get_index_label()
            attempts += 1
        used_labels.add(label)

        data_cells = "".join(
            f'<td style="text-align:{align_data[i]}">{ContentLibrary.get_data_value()}</td>'
            for i in range(n_data_cols)
        )
        body_rows.append(
            f'<tr><td style="text-align:left">{label}</td>{data_cells}</tr>'
        )

    table_html = f'<table>{thead_html}<tbody>{"".join(body_rows)}</tbody></table>'
    return {
        "html": table_html,
        "n_cols": 1 + n_data_cols,   # index col + data cols
        "depth": header_levels,
    }


# ---------------------------------------------------------------------------
# Cross-Tab (Combined) Generator
# ---------------------------------------------------------------------------

def _get_nodes_at_level(node: HeaderNode, target: int, cur: int = 0) -> List[HeaderNode]:
    """Return all HeaderNode children that exist at exactly `target` BFS depth."""
    if cur == target:
        return [node] if node.children else []
    result: List[HeaderNode] = []
    for c in node.children:
        result.extend(_get_nodes_at_level(c, target, cur + 1))
    return result


def generate_cross_tab_table(row_depth: int = 3, header_levels: int = 2) -> Dict[str, Any]:
    """
    Cross-Tab: complex nested column headers (top) + BOM-style rowspan tree (left).
    - Top: multi-level colspan hierarchical headers.
    - Left: deep rowspan BOM hierarchy with named, non-empty cells.
    - Center: randomized numeric/status data points.
    """
    domain = random.choice(["financial", "stat"])
    roots = _build_header_tree(header_levels, domain)
    total_depth = _get_header_depth(roots)
    for node in roots:
        _compute_header_spans(node, total_depth)
    n_data_cols = _count_leaf_cols(roots)
    align_list = [random.choice(["left", "center", "right"]) for _ in range(n_data_cols)]

    # Row hierarchy label for the corner spanning cell
    corner_label = random.choice([
        "Component", "BOM Item", "Part", "Assembly", "SKU", "Item",
    ])

    # Build row-span BOM tree
    root_node = TableNode(name=_safe_name(ContentLibrary.get_header_label("bom")), level=0)
    _grow_tree(root_node, 0, row_depth)
    calculate_spans(root_node)
    rows_data = flatten_to_rows(root_node)

    # Build thead: corner cell spans all header rows; then BFS-render column headers
    bfs_levels: List[List[HeaderNode]] = []
    current = roots
    for _ in range(total_depth):
        bfs_levels.append(current)
        next_lv: List[HeaderNode] = []
        for n in current:
            next_lv.extend(n.children)
        current = next_lv

    thead_rows_html = []
    for lvl_idx, level_nodes in enumerate(bfs_levels):
        cells = ""
        if lvl_idx == 0:
            cells += f'<th rowspan="{total_depth}" style="text-align:left">{corner_label}</th>'
        for node in level_nodes:
            cs = f' colspan="{node.colspan}"' if node.colspan > 1 else ""
            rs = f' rowspan="{node.rowspan}"' if node.rowspan > 1 else ""
            cells += f"<th{cs}{rs}>{node.label}</th>"
        thead_rows_html.append(f"<tr>{cells}</tr>")
    thead_html = "<thead>" + "".join(thead_rows_html) + "</thead>"

    # Build body rows — ensure all hierarchy labels are non-empty
    html_rows = []
    for row in rows_data:
        tr = "<tr>"
        for cell in row:
            span_val = cell["span"]
            rs = f' rowspan="{span_val}"' if span_val > 1 else ""
            label = _safe_name(cell["name"], "bom")
            tr += f'<td{rs} style="text-align:left"><strong>{label}</strong></td>'
        for i in range(n_data_cols):
            tr += f'<td style="text-align:{align_list[i]}">{ContentLibrary.get_data_value()}</td>'
        tr += "</tr>"
        html_rows.append(tr)

    table_html = (
        f'<table>{thead_html}'
        f'<tbody>{"".join(html_rows)}</tbody></table>'
    )
    return {
        "html": table_html,
        "n_cols": 1 + n_data_cols,
        "depth": max(row_depth, header_levels),
    }


# ---------------------------------------------------------------------------
# Difficulty → Depth Mapping
# ---------------------------------------------------------------------------

DIFFICULTY_PARAMS = {
    # (row_depth_range, header_levels_range)
    "simple":  ((1, 2), (1, 2)),
    "medium":  ((3, 3), (3, 3)),
    "extreme": ((4, 6), (3, 4)),
}


def get_difficulty() -> str:
    return random.choices(
        ["simple", "medium", "extreme"],
        weights=[0.20, 0.30, 0.50]
    )[0]


def generate_table(mode: str = "row_heavy", difficulty: str = "extreme") -> Dict[str, Any]:
    """
    Unified entry point. Returns:
      { html, n_cols, depth, mode, difficulty, style_cfg, css }
    """
    row_range, hdr_range = DIFFICULTY_PARAMS[difficulty]
    row_depth = random.randint(*row_range)
    hdr_levels = random.randint(*hdr_range)

    if mode == "row_heavy":
        result = generate_row_heavy_table(depth=row_depth)
    elif mode == "col_heavy":
        result = generate_col_heavy_table(header_levels=hdr_levels)
    else:  # cross_tab
        result = generate_cross_tab_table(row_depth=row_depth, header_levels=hdr_levels)

    style_cfg = StyleEngine.generate(num_cols=result["n_cols"])
    css = StyleEngine.to_css(style_cfg)

    result["mode"] = mode
    result["difficulty"] = difficulty
    result["css"] = css
    return result