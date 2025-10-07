"""Utility functions for data analysis and visualization."""

import sys

sys.path.append("/project/notebooks")


import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from matplotlib.patches import Patch
from shapely import wkb

# pesticide docs info
BASE_URL = "https://d3lsdszfx9jqxt.cloudfront.net/data-query/"
"""
The below endpoints aggregate data to different geospatial units.
For this project, School Districts may be the most useful.
Counties are the largest unit, and sections (1x1 sq mi) are the smallest.
"""
AGGREGATE_ENDPOINTS = {
    "school": "674518c90507860008cda24f",
    "tract": "6744f2bbdb91810008024956",
    "zip": "674516220507860008cda24c",
    "county": "674513830507860008cda249",
    "section": "67451c010507860008cda252",
    "township": "6744f63adb91810008024959",
}


def top_n_df(data, level, sort_by, n=10):
    """Returns a DataFrame containing the top `n` rows from the input data, sorted by the specified column.

    Parameters:
        data (pd.DataFrame): The input DataFrame containing the data to be processed.
        level (str): The level of aggregation or grouping.
        sort_by (str): The column name to sort the data by. Must be one of ['ai_intensity', 'prd_intensity'].
        n (int, optional): The number of top rows to return. Defaults to 10.

    Returns:
        pd.DataFrame: A DataFrame containing the top `n` rows, with columns ["Area Name", "FIPS", sort_by].

    Raises:
        ValueError: If sort_by is not one of ['ai_intensity', 'prd_intensity'].

    Notes:
        - The function resets the index of the sorted DataFrame before slicing the top `n` rows.
        - Only the columns ["Area Name", "FIPS", sort_by] are included in the returned DataFrame.
    """
    if sort_by not in ["ai_intensity", "prd_intensity"]:
        raise ValueError(
            f"sort_by must be one of ['ai_intensity', 'prd_intensity'], got {sort_by}"
        )
    top_n = data.sort_values(sort_by, ascending=False).reset_index()[:n][
        ["Area Name", "FIPS", sort_by]
    ]
    return top_n


def show_dem_histograms(data, dem_cols):
    """Displays histograms for a list of demographic columns from a given dataset.

    This function generates histograms for each column in dem_cols using the
    same x-axis range for consistency. The histograms are displayed in a grid
    layout with up to 3 columns per row.

    Args:
        data (pd.DataFrame): The dataset containing the demographic columns.
        dem_cols (list of str): A list of column names in the dataset for which
            histograms will be generated.

    Returns:
        None: The function displays the histograms and does not return any value.

    Notes:
        - Missing values (NaN) in the columns are dropped before plotting.
        - The x-axis range for all histograms is determined by the minimum and
          maximum values across all specified columns.
    """
    all_values = pd.concat([data[col].dropna() for col in dem_cols])
    x_min, x_max = all_values.min(), all_values.max()

    num_cols = len(dem_cols)
    cols_per_row = 3
    num_rows = (num_cols + cols_per_row - 1) // cols_per_row

    plt.figure(figsize=(5 * cols_per_row, 4 * num_rows))

    for idx, dem in enumerate(dem_cols):
        plt.subplot(num_rows, cols_per_row, idx + 1)
        plt.hist(data[dem].dropna(), bins=30, alpha=0.7, range=(x_min, x_max))
        plt.title(dem)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.xlim(x_min, x_max)

    plt.tight_layout()
    plt.show()


def plot_correlations(corr_df, sort=True, figsize=(10, 8)):
    """Plots horizontal bar charts for correlation values stored in a DataFrame.

    This function visualizes correlation coefficients using horizontal bar charts.
    It assumes the input DataFrame contains two columns of correlation values, with
    the variable names stored in the index. Positive correlations are displayed in
    blue, while negative correlations are displayed in red.

    Args:
        corr_df (pd.DataFrame): A DataFrame containing two columns of correlation
                                values. The index should represent variable names.
        sort (bool, optional): If True, sorts the bars by the absolute value of the
                               correlation coefficients. Defaults to True.
        figsize (tuple, optional): A tuple specifying the size of each plot.
                                   Defaults to (10, 8).

    Returns:
        None: The function displays the plots but does not return any value.

    Behavior:
        - For each column in the DataFrame, a separate bar chart is created.
        - Bars are color-coded: blue for positive correlations and red for negative.
        - The correlation coefficient is displayed as text on each bar.
        - A vertical line at 0 is added for reference.
        - The y-axis is inverted to display the highest correlations at the top.
    """
    for corr_col in corr_df.columns:
        data = corr_df[[corr_col]].copy()
        if sort:
            data["abs_corr"] = data[corr_col].abs()
            data = data.sort_values(by="abs_corr", ascending=False)

        values = data[corr_col]
        colors = [
            "#4ba3c3" if v > 0 else "#d62839" for v in values
        ]  # wanted red if positive, blue if negative

        plt.figure(figsize=figsize)
        bars = plt.barh(data.index, values, color=colors)
        plt.axvline(0, color="gray", linewidth=0.8)

        for bar in bars:
            width = bar.get_width()
            plt.text(
                width + 0.02 * np.sign(width),
                bar.get_y() + bar.get_height() / 2,
                f"{width:.2f}",
                va="center",
                fontsize=9,
            )

        ax = plt.gca()
        for spine in ["right", "left"]:
            ax.spines[spine].set_visible(False)

        plt.xlabel("Correlation Coefficient")
        plt.title(f"Correlation with {corr_col}")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()


def into_gdf(data, geometry_col):
    """Converts a DataFrame into a GeoDataFrame by interpreting a specified column as geometry.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data.
        geometry_col (str): The name of the column in the DataFrame that contains geometry data
                            in WKB (Well-Known Binary) format.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame with the specified geometry column converted to geometries.
    """
    gdf = gpd.GeoDataFrame(data, geometry=data[geometry_col].apply(wkb.loads, hex=True))
    return gdf


def create_corr_df(data, cols_1, cols_2):
    """Creates a DataFrame with correlation values between two sets of columns.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        cols_1 (list): The first set of columns to correlate.
        cols_2 (list): The second set of columns to correlate.

    Returns:
        pd.DataFrame: A DataFrame with correlation values.
    """
    corr_df = pd.DataFrame(index=cols_1, columns=cols_2)
    for col_1 in cols_1:
        for col_2 in cols_2:
            corr_df.loc[col_1, col_2] = data[col_1].corr(data[col_2])
    return corr_df


def demographic_intensity_correlations(data, targetVariable):
    """Constructs dataframe and plot for demographic variables.

    Uses the functions create_corr_df() and plot_correlations()

    Input:
        data (pd.DataFrame) : pandas dataframe that contains target variable and demographic variables
        targetVariable (str) : the chosen variable

    Returns:
        corrDf (pd.DataFrame) : correlation dataframe
        Also shows the correlation plot

    """
    demographic_data = [col for col in data.columns if col.startswith("Pct")]

    corrDf = create_corr_df(data, demographic_data, [targetVariable])

    plot_correlations(corrDf, sort=True, figsize=(8, 6))

    return corrDf


def create_CA_visualization(data, targetVariable, year):
    """Creates visualization of California and the targeted variable

    Args:
        data: (pd.DataFrame)
        targetVariable: (string) the variable you are visualizing
        year: (string) the year (or years) that are being displayed

    Returns:
        None: Displays the visualization but has no return
    """
    _, ax = plt.subplots(figsize=(10, 8))

    data.plot(
        ax=ax,
        column=targetVariable,
        cmap="Blues",
        legend=True,
        edgecolor="black",
        linewidth=0.2,
        missing_kwds={"color": "slategray", "label": "No Reported Data"},
    )

    plt.title(f"{targetVariable} in California \n ({year})", fontsize=14)
    plt.axis("off")

    cAx = plt.gcf().axes[-1]
    cAx.set_yticklabels([])
    cAx.set_ylabel("Low → High", fontsize=14)

    missing = Patch(color="slategray", label="No reported Data")
    ax.legend(handles=[missing], loc="lower left", fontsize=12, frameon=False)

    plt.show()


def trendPlot(
    target_variable, api_query=None, endpoint="school", measurement="ai_intensity"
):
    """This function plots the 5 year trends for a target variable, utilizing get_aggregate_results() to do so.

    Args:
        target_variable (str): String representation of target varaible for the plot title
        api_query (list of tuples) : list of tuples containing the parameters for get_aggregate_results()
            Defaults to no parameters/all results (empty list)
        endpoint (str): target geospatial unit from AGGREGATE_ENDPOINT dict. Defaults to "school"
        measurement (str) : the column used for measurement. Defaults to "ai_intensity"

    Returns:
        None: Displays the visualization but has no return

    Example usage:
        trendPlot("All Pesticides") : returns plot of total annual AI pesticide usage
        trendPlot("Pyrethroid", [("ai_categories", ["PYR"])], measurement = "prd_intensity") : returns plot of annual
            pyrethroid product intensity
    """
    # getting df of each year
    years = [2017, 2018, 2019, 2020, 2021, 2022]

    total_usage_sum = []

    if api_query is None:
        api_query = []

    params = dict(api_query)

    for y in years:
        annual_df = get_aggregate_results(
            endpoint, start_year=y, start_month=1, end_year=y, end_month=12, **params
        )

        total_usage_sum.append((y, annual_df[measurement].sum()))

    # plotting
    data = pd.DataFrame(total_usage_sum, columns=["year", "measurement"])

    sns.set_style("whitegrid", {"grid.linestyle": ":", "axes.edgecolor": "0.5"})
    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=data,
        x="year",
        y="measurement",
        marker="o",
        markersize=8,
        linewidth=2.5,
        color="#4ba3c3",  # wanted to make it cohesive with other plots on utils
    )

    plt.ylabel(f"Annual {measurement} ({target_variable})", fontsize=12)
    plt.xlabel("Year", fontsize=12, labelpad=10)
    plt.title(
        f"Trend in {target_variable} Use (2017-2022)",
        fontsize=14,
        fontweight="semibold",
    )

    plt.xticks(data["year"], fontsize=10)
    plt.yticks(fontsize=10)

    sns.despine(left=True, bottom=True)

    plt.tight_layout()
    plt.show()


# will change
def top_10_barplot(top_n_df):
    """(Jenny I know you have a pretty one of these so I didn't spend a ton of time on this one, put yours here instead!)

    (Or I can edit this one whatever!)
    """
    colors = sns.color_palette("crest", n_colors=len(top_n_df))

    sns.barplot(
        x="ai_intensity", y="Area Name", data=top_n_df, palette=colors, hue="Area Name"
    )

    plt.show()


def difference_in_years(df2017, df2022, targetVariable, otherVariables, mergeOn="FIPS"):
    """This function looks at the changes of a given variable between 2017 and 2022

    Inputs:
        df2017 (pd.DataFrame) : aggregate dataframe obtained from get_aggregate_results() for 2017
        df2022 (pd.DataFrame) : aggregate dataframe obtained from get_aggregate_results() for 2022
        targetVariable (str) : chosen variable
        otherVariables (lst) : list of other variables to keep.
        mergeOn (str) : column you are merging the two dataframes on. Defaults to "FIPS" for the school level

    Returns:
        (pd.DataFrame) : dataframe that contains the merged on variable, 2017 target variable, 2022 target variable,

    """
    df2017 = df2017.rename(columns={targetVariable: targetVariable + "_2017"})
    df2022 = df2022.rename(columns={targetVariable: targetVariable + "_2022"})

    df2022 = df2022.merge(df2017[[mergeOn, f"{targetVariable}_2017"]], on=mergeOn)

    df2022[f"{targetVariable}_change"] = (
        df2022[f"{targetVariable}_2022"] - df2022[f"{targetVariable}_2017"]
    )

    return df2022[
        otherVariables
        + [
            mergeOn,
            f"{targetVariable}_2017",
            f"{targetVariable}_2022",
            f"{targetVariable}_change",
        ]
    ]


# pesticide docs function
def get_df_from_api(endpoint: str, string_columns: list[str] = None) -> pd.DataFrame:
    """This function returns a DataFrame from the API.

    Args:
        endpoint (str) : one of the end points from AGGREGATE_ENDPOINTS dict
        string_columns (lst) : list on columns to be contained in returned dataframe

    Returns:
        DataFrame (pd.DataFrame) : Dataframe contained queried information
    """
    converters = {col: str for col in string_columns}
    return pd.read_csv(f"{BASE_URL}{endpoint}?format=csv", converters=converters)


# pesticide docs function
def get_aggregate_results(
    endpoint: str,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    active_ingredients: None | list[str] = None,
    products: None | list[str] = None,
    sites: None | list[str] = None,
    health_and_env_risks: None | list[str] = None,
    ai_categories: None | list[str] = None,
    county_codes: None | list[str] = None,
    as_json: bool = False,
):
    """Returns the aggregate results of pesticide usage based on the parameters provided.

    The API is configured to summarize data that matches any of the conditions,
    so if you pass in multiple active ingredients, it will return the aggregate results for the aggregate usage
    matching *any* of the provided active ingredients.

    Parameters:
        endpoint (str): The endpoint to query for aggregate results.
        start_year (int): The starting year for the query.
        start_month (int): The starting month for the query.
        end_year (int): The ending year for the query.
        end_month (int): The ending month for the query.
        active_ingredients (None | List[str], optional): List of active ingredients to filter by. Defaults to None.
        products (None | List[str], optional): List of products to filter by. Defaults to None.
        sites (None | List[str], optional): List of sites to filter by. Defaults to None.
        health_and_env_risks (None | List[str], optional): List of health and environmental risks to filter by. Defaults
            to None.
        ai_categories (None | List[str], optional): List of active ingredient categories to filter by. Defaults to None.
        county_codes (None | List[str], optional): List of county codes to filter by. Defaults to None.
        as_json (bool, optional): If True, returns the data as JSON. If False, returns the data as a pandas DataFrame.
                Defaults to False.

    Returns:
        dict or pandas.DataFrame: The aggregate results in JSON format if `as_json` is True, otherwise a pandas DataFrame.
    """
    params = {
        "start": f"{start_year}-" + f"{start_month:02d}",
        "end": f"{end_year}-" + f"{end_month:02d}",
    }
    if active_ingredients:
        params["chemical"] = active_ingredients
    if products:
        params["product"] = products
    if sites:
        params["site"] = sites
    if health_and_env_risks:
        params["health"] = health_and_env_risks
    if ai_categories:
        params["category"] = ai_categories
    if county_codes:
        params["county"] = county_codes
    data = requests.get(
        BASE_URL + AGGREGATE_ENDPOINTS[endpoint], params=params, timeout=60
    ).json()
    if as_json:
        return data
    return pd.DataFrame(data)
