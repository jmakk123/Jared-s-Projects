#!/usr/bin/env python3
"""Visualization of pesticide usage near schools in California (2017-2022)

Produces three types of maps:
1. Statewide overview of pesticide usage with school locations
2. Detailed views for top 10 districts by pesticide usage
3. Risk areas map showing schools near high pesticide use areas
"""

import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from libpysal.weights import Queen
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.utils.clean import clean_df, create_df_from_calpip

project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))
output_dir = Path("output/figures")
output_dir.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", category=UserWarning, message="CRS mismatch")


def load_data() -> tuple[pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load and prepare all required datasets"""
    calpip = pd.read_parquet(Path.home() / "Downloads" / "calpip_full.parquet")
    school_pt = gpd.read_file(project_root / "data" / "school_sites.geojson")
    merged_chem = create_df_from_calpip(calpip)
    return calpip, school_pt, merged_chem


def create_statewide_map(
    merged_chem: gpd.GeoDataFrame,
    school_pt: gpd.GeoDataFrame,
    districts: gpd.GeoDataFrame,
) -> plt.Figure:
    """Create statewide visualization of pesticide usage"""
    fig, ax = plt.subplots(figsize=(20, 15), facecolor="white")

    val_sections = get_adjacent_sections(merged_chem)
    val_sections["log_lbs_used"] = np.log1p(val_sections["lbs_chm_used"].fillna(0))

    val_sections[val_sections["lbs_chm_used"] > 0].plot(
        ax=ax,
        column="log_lbs_used",
        cmap="YlOrRd",
        legend=True,
        edgecolor="black",
        linewidth=0.3,
    )

    districts.boundary.plot(ax=ax, color="black", linewidth=0.5)

    school_pt.plot(
        ax=ax, marker="o", color="blue", markersize=5, alpha=0.7, label="Schools"
    )

    plt.title(
        "Pesticide Usage with School Locations Across California\n(2017-2022)",
        fontsize=16,
    )
    ax.axis("off")

    elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markersize=10,
            label="Schools",
        ),
        Patch(facecolor="#ffffcc", edgecolor="black", label="Low pesticide use"),
        Patch(facecolor="#fd8d3c", edgecolor="black", label="Medium pesticide use"),
        Patch(facecolor="#bd0026", edgecolor="black", label="High pesticide use"),
    ]

    ax.legend(
        handles=elements,
        loc="upper right",
        framealpha=1,
        fontsize=12,
        title_fontsize=14,
    )

    cbar = plt.gcf().axes[-1]
    cbar.set_yticklabels([])
    cbar.set_ylabel("Low --> High", fontsize=12)

    plt.tight_layout()
    return fig


def get_adjacent_sections(merged_chem: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Get sections with pesticide use and their adjacent neighbors"""
    used_sections = merged_chem[merged_chem["lbs_chm_used"] > 0]["CO_MTRS"].unique()
    used_mask = merged_chem["CO_MTRS"].isin(used_sections)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        w = Queen.from_dataframe(merged_chem, use_index=False)

    adj_indices = set()
    for idx in merged_chem[used_mask].index:
        adj_indices.update(w.neighbors.get(idx, []))
    keep_mask = used_mask | merged_chem.index.isin(adj_indices)

    return merged_chem[keep_mask].copy()


def create_district_maps(
    chem_by_district: gpd.GeoDataFrame,
    districts: gpd.GeoDataFrame,
    school_pt: gpd.GeoDataFrame,
) -> list[plt.Figure]:
    """Create detailed district visualizations for top pesticide-using districts"""
    usage_per_district = (
        chem_by_district.groupby("NAME_right")["lbs_chm_used_left"].sum().reset_index()
    )
    top_districts = usage_per_district.nlargest(10, "lbs_chm_used_left")[
        "NAME_right"
    ].tolist()
    figures = []

    for district_name in top_districts:
        fig, ax = plt.subplots(figsize=(10, 8))
        district_geom = districts[districts["NAME"] == district_name]
        chem_in_district = chem_by_district[
            chem_by_district["NAME_right"] == district_name
        ]
        schools_in_district = gpd.sjoin(
            school_pt, district_geom, how="inner", predicate="within"
        )

        district_geom.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1)

        if not chem_in_district.empty:
            chem_in_district.plot(
                column="lbs_chm_used_left",
                ax=ax,
                cmap="YlOrRd",
                legend=True,
                legend_kwds={"label": "Pounds of Pesticide Used"},
                markersize=50,
                alpha=0.7,
            )

        if not schools_in_district.empty:
            schools_in_district.plot(
                ax=ax, color="blue", marker="o", markersize=50, label="Schools"
            )

        ax.set_title(f"Pesticide Usage in {district_name}", fontsize=14)
        ax.legend()
        ax.set_axis_off()

        if not chem_in_district.empty:
            cbar = plt.gcf().axes[-1]
            cbar.set_ylabel("Pounds of Pesticide", fontsize=12)

        plt.tight_layout()
        figures.append(fig)

    return figures


def create_risk_area_map(
    merged_chem: gpd.GeoDataFrame, school_pt: gpd.GeoDataFrame
) -> plt.Figure:
    """Create map showing schools near high pesticide use areas"""
    high_use_threshold = merged_chem[merged_chem["lbs_chm_used"] > 0][
        "lbs_chm_used"
    ].quantile(0.75)

    w = Queen.from_dataframe(merged_chem, use_index=False)

    high_use_mask = merged_chem["lbs_chm_used"] >= high_use_threshold
    high_use_sections = merged_chem[high_use_mask]

    adjacent_indices = set()
    for idx in high_use_sections.index:
        adjacent_indices.update(w.neighbors.get(idx, []))

    risk_area_mask = high_use_mask | merged_chem.index.isin(adjacent_indices)
    risk_areas = merged_chem[risk_area_mask]

    schools_in_risk_areas = gpd.sjoin(
        school_pt, risk_areas, how="inner", predicate="within"
    )

    total_schools = len(school_pt)
    schools_in_risk = len(schools_in_risk_areas)
    percentage_in_risk = (schools_in_risk / total_schools) * 100

    print(f"Schools within 1 mile of high pesticide areas: {schools_in_risk}")
    print(f"Percentage of schools in risk areas: {percentage_in_risk:.2f}%")

    fig, ax = plt.subplots(figsize=(15, 10), facecolor="white")

    merged_chem.plot(ax=ax, color="lightgrey", edgecolor="white", linewidth=0.2)

    risk_areas.plot(
        ax=ax,
        color="red",
        edgecolor="darkred",
        linewidth=0.3,
        alpha=0.3,
        label="High pesticide risk areas (1 mile radius)",
    )

    school_pt.plot(
        ax=ax, marker="o", color="blue", markersize=5, alpha=0.7, label="Schools"
    )

    schools_in_risk_areas.plot(
        ax=ax,
        marker="o",
        color="yellow",
        markersize=8,
        edgecolor="black",
        linewidth=0.5,
        label="Schools in risk areas",
    )

    plt.title(
        f"Schools Near High Pesticide Use Areas\n{percentage_in_risk:.1f}% of schools within 1 mile",
        fontsize=16,
    )
    ax.axis("off")

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markersize=10,
            label="All Schools",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="yellow",
            markersize=10,
            markeredgecolor="black",
            label="Schools in Risk Areas",
        ),
        Patch(
            facecolor="red",
            edgecolor="darkred",
            alpha=0.3,
            label="High Pesticide Risk Areas",
        ),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper right",
        framealpha=1,
        fontsize=12,
        title_fontsize=14,
    )

    plt.tight_layout()
    return fig


def main():
    """Main execution function"""
    calpip, school_pt, merged_chem = load_data()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        districts = clean_df(
            start_year=2017,
            start_month=1,
            end_year=2022,
            end_month=12,
            endpoint="school",
        )

    statewide_fig = create_statewide_map(merged_chem, school_pt, districts)
    risk_area_fig = create_risk_area_map(merged_chem, school_pt)
    chem_by_district = gpd.sjoin(
        merged_chem, districts, how="inner", predicate="within"
    )
    district_figs = create_district_maps(chem_by_district, districts, school_pt)

    for i, fig in enumerate(district_figs):
        fig.savefig(output_dir / f"district_{i}.png", bbox_inches="tight", dpi=300)
        plt.close(fig)

    statewide_fig.savefig(output_dir / "statewide.png", bbox_inches="tight", dpi=300)
    risk_area_fig.savefig(output_dir / "risk_areas.png", bbox_inches="tight", dpi=300)
    plt.close(statewide_fig)
    plt.close(risk_area_fig)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
