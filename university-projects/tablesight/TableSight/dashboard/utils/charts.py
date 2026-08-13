"""Plotly chart helpers for the dashboard."""
from __future__ import annotations

from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# UChicago-maroon palette — accepts both display names ("SPARTAN")
# and internal keys ("spartan") so chart calls don't need to translate.
MODEL_COLOR = {
    "florence":    "#8d6e63",   # warm brown — leather book
    "unitable":    "#486f7d",   # steel blue
    "spartan":     "#800000",   # UChicago maroon (flagship model)
    "Florence-2":  "#8d6e63",
    "UniTable":    "#486f7d",
    "SPARTAN":     "#800000",
}


def metric_bars(metrics_by_model: Dict[str, Dict[str, float]]) -> go.Figure:
    """metrics_by_model: {'Florence-2': {'TEDS': 0.86, 'TEDS-S': 0.91}, ...}"""
    rows = []
    for model, m in metrics_by_model.items():
        for k, v in m.items():
            rows.append({"model": model, "metric": k, "value": float(v)})
    df = pd.DataFrame(rows)
    fig = px.bar(df, x="metric", y="value", color="model", barmode="group",
                 color_discrete_map=MODEL_COLOR, text="value")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside",
                       marker_line_color="rgba(0,0,0,0.15)",
                       marker_line_width=0.8)
    fig.update_yaxes(range=[0, max(1.0, df["value"].max() * 1.15)],
                     gridcolor="rgba(0,0,0,0.06)",
                     title_text=None)
    fig.update_xaxes(title_text=None)
    fig.update_layout(
        height=360,
        margin=dict(t=20, r=12, b=12, l=12),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(family="Inter, sans-serif", size=12, color="#1c1917"),
        legend=dict(title=None, orientation="h", yanchor="bottom", y=1.02,
                     xanchor="right", x=1, font=dict(size=11)),
    )
    return fig


def teds_distribution(df_results: pd.DataFrame) -> go.Figure:
    """Distribution of per-sample TEDS scores per model (CDF + median lines)."""
    fig = go.Figure()
    for model, sub in df_results.groupby("model"):
        sub_sorted = sub.sort_values("teds")
        n = len(sub_sorted)
        if n == 0: continue
        fig.add_trace(go.Scatter(
            x=sub_sorted["teds"],
            y=[i / n for i in range(1, n + 1)],
            mode="lines",
            name=model,
            line=dict(color=MODEL_COLOR.get(model, None), width=2),
        ))
    fig.update_xaxes(title="TEDS", range=[0, 1])
    fig.update_yaxes(title="Cumulative fraction", range=[0, 1])
    fig.update_layout(height=380, margin=dict(t=30, r=10, b=10, l=10))
    return fig


def failure_pies(failure_counts_by_model: Dict[str, Dict[str, int]]) -> go.Figure:
    """One pie chart per model, side-by-side, of failure category distribution."""
    from plotly.subplots import make_subplots
    models = list(failure_counts_by_model.keys())
    fig = make_subplots(rows=1, cols=len(models),
                        specs=[[{"type": "pie"} for _ in models]],
                        subplot_titles=models)
    for i, m in enumerate(models, start=1):
        d = failure_counts_by_model[m]
        if not d:
            continue
        fig.add_trace(go.Pie(labels=list(d.keys()), values=list(d.values()),
                              hole=0.35, sort=False), row=1, col=i)
    fig.update_layout(height=380, margin=dict(t=40, r=10, b=10, l=10), showlegend=True)
    return fig


def summary_table(df_results: pd.DataFrame) -> pd.DataFrame:
    """Summary stats per model and per image_type, if available."""
    out = []
    grp_cols = ["model"]
    if "image_type" in df_results.columns:
        grp_cols.append("image_type")
    for keys, sub in df_results.groupby(grp_cols):
        if isinstance(keys, tuple):
            row = dict(zip(grp_cols, keys))
        else:
            row = {grp_cols[0]: keys}
        row.update({
            "n":          len(sub),
            "teds_mean":  sub["teds"].mean(),
            "teds_median": sub["teds"].median(),
            "teds_p25":   sub["teds"].quantile(0.25),
            "teds_p75":   sub["teds"].quantile(0.75),
        })
        if "teds_s" in sub.columns:
            row["teds_s_mean"] = sub["teds_s"].mean()
        if "time_s" in sub.columns:
            row["sec_per_sample"] = sub["time_s"].mean()
        out.append(row)
    return pd.DataFrame(out).round(4)


__all__ = ["metric_bars", "teds_distribution", "failure_pies", "summary_table", "MODEL_COLOR"]
