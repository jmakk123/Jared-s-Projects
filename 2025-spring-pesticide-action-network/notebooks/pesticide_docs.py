# %%
import requests
import pandas as pd
import geopandas as gpd
from typing import List
# %%
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
  "township": "6744f63adb91810008024959"
}
def get_df_from_api(
    endpoint: str,
    string_columns: List[str] = []
) -> pd.DataFrame:
    """
    This function returns a DataFrame from the API.
    """ 
    converters = {col: str for col in string_columns}
    return pd.read_csv(f"{BASE_URL}{endpoint}?format=csv", converters=converters)

def get_aggregate_results(
    endpoint: str,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    active_ingredients: None | List[str] = None,
    products: None | List[str] = None,
    sites: None | List[str] = None,
    health_and_env_risks: None | List[str] = None,
    ai_categories: None | List[str] = None,
    county_codes: None | List[str] = None,
    as_json: bool = False
):
  """
  Returns the aggregate results of pesticide usage based on the parameters provided. 
  The API is configured to summarize data that matches any of the conditions,
  so if you pass in multiple active ingredients, it will return the aggregate results for the aggregate usage
  matching *any* of the provided active ingredients.

  Parameters:
  - endpoint (str): The endpoint to query for aggregate results.
  - start_year (int): The starting year for the query.
  - start_month (int): The starting month for the query.
  - end_year (int): The ending year for the query.
  - end_month (int): The ending month for the query.
  - active_ingredients (None | List[str], optional): List of active ingredients to filter by. Defaults to None.
  - products (None | List[str], optional): List of products to filter by. Defaults to None.
  - sites (None | List[str], optional): List of sites to filter by. Defaults to None.
  - health_and_env_risks (None | List[str], optional): List of health and environmental risks to filter by. Defaults to None.
  - ai_categories (None | List[str], optional): List of active ingredient categories to filter by. Defaults to None.
  - county_codes (None | List[str], optional): List of county codes to filter by. Defaults to None.
  - as_json (bool, optional): If True, returns the data as JSON. If False, returns the data as a pandas DataFrame. Defaults to False.

  Returns:
  - dict or pandas.DataFrame: The aggregate results in JSON format if `as_json` is True, otherwise a pandas DataFrame.
  """
  params = {
      "start": f"{start_year}-" + f"{start_month:02d}",
      "end": f"{end_year}-" + f"{end_month:02d}"
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
  data = requests.get(BASE_URL + AGGREGATE_ENDPOINTS[endpoint], params=params).json()
  if (as_json):
      return data
  return pd.DataFrame(data)
# %%
##### QUERY PARAMETER DFs #####
"""
An API query parameter is a key-value pair that is appended to the end of a URL to pass data to the server. 
These parameters are used to filter, sort, or modify the data that is returned by the API. 
They are typically included in the URL after a question mark (?) and are separated by an ampersand (&) if there are multiple parameters.

In the provided code, several query parameters are used to fetch data from an API. Here are the query parameters and their explanations:

- county: This parameter is used to filter data based on the county code. CountyCode is distinct from Census FIPS, unfortunately. 
  It is included in the county_codes DataFrame, which contains the name, county code, and Census FIPS code for each county.
- site: This parameter is used to filter data based on the site code. It is included in the site_codes DataFrame, 
  which contains the site code, site name, and average pounds used per year between 2017 and 2022.
- category: This parameter is used to filter data based on the category ID. It is included in the ai_categories DataFrame, 
  which contains the label (name of the category) and the ID.
- chemical: This parameter is used to filter data based on the chemical code. It is included in the active_ingredients DataFrame, 
  which contains the chemical code, chemical name, and yearly average.
- product: This parameter is used to filter data based on the product code. It is included in the products DataFrame, 
  which contains the product code, product name, and yearly average.
- health: This parameter is used to filter data based on the health and environmental risk ID. It is included in the health_and_env_risks DataFrame,

These query parameters are obtain from `get_df_from_api` and can be used with `get_aggregate_results` to filter results.
"""

"""
Name, CountyCode (query param column), and Census FIPS code for each county.
"""
county_codes = get_df_from_api("66db44c7f96f070008c08e39", string_columns=["CountyCode", "FIPS"])
"""
site_code (query param column), site_name, and average pounds used per year between 2017 and 2022
"""
site_codes = get_df_from_api("66d8a22692868e000864e898", string_columns=["site_code", "site_name"])
"""
label (name of category), id (query param column)
"""
ai_categories = get_df_from_api("66e1e112c640880008ba68f2")
"""
chem_code (query param column), chem_name, yearly_average
"""
active_ingredients = get_df_from_api("66d88452ae7ce10008a9473f")
"""
product_code (query param column), product_name, yearly_average
"""
products = get_df_from_api("66d8a20a92868e000864e897")
"""
label, id (query param column)
"""
health_and_env_risks = get_df_from_api("66d8a25592868e000864e899")
# %%
##### EXAMPLE USAGE #####
"""
Let's get some simple, county level data within the Amide or Azole categories for the year 2020:
"""
amide_azole_codes = ai_categories[ai_categories["label"].isin(["Amide", "Azole"])]

category_county_df = get_aggregate_results(
  "county", 
  start_year=2020, 
  start_month=1, 
  end_year=2020, 
  end_month=12, 
  ai_categories=amide_azole_codes['id'].tolist(),
)
"""
Note that dates are inclusive, so starting with 1 (January) and ending with 12 (December) will give you the entire year.
"""
# %%
"""
Next, let's get Census tract level data for Census tracts in napa county for the year 2020:
"""
napa_code = county_codes[county_codes["Name"].str.contains("Napa")]
napa_tract_data_df = get_aggregate_results(
    "tract", 
    start_year=2020, 
    start_month=1, 
    end_year=2020, 
    end_month=12, 
    county_codes=napa_code["CountyCode"].tolist()
)
"""
Note that you will get all demography, regardless of if there is pesticide data or not.
Census tract "GEOIDS" are heirarchical, meaning that the first five digits are the county FIPS code.
So, we can easily filter down to just relevant data:
"""
# %%
napa_tract_data_df = napa_tract_data_df[napa_tract_data_df.GEOID.str.startswith(napa_code['FIPS'].values[0])].reset_index(drop=True)
# %%
"""
KEY DATA COLUMN DESCRIPTIONS:
GEOID                string  # Unique identifier for the geographic area (Alternatively, FIPS, ZCTA, etc.)
Area Name            string  # Name of the geographic area
lbs_chm_used        float64  # Total pounds of chemicals used
lbs_prd_used        float64  # Total pounds of products used
ai_intensity        float64  # Intensity of active ingredients used (pounds per square mile)
prd_intensity       float64  # Intensity of products used (pounds per square mile)
Median HH Income    float64  # Median household income in the area
Pop Total             int64  # Total population in the area
Pop NH Black          int64  # Total population of Non-Hispanic Black individuals
Pct NH Black        float64  # Percentage of Non-Hispanic Black individuals
Pop Hispanic          int64  # Total population of Hispanic individuals
Pct Hispanic        float64  # Percentage of Hispanic individuals
Pop NH White          int64  # Total population of Non-Hispanic White individuals
Pct NH White        float64  # Percentage of Non-Hispanic White individuals
Pop NH Asian          int64  # Total population of Non-Hispanic Asian individuals
Pct NH Asian        float64  # Percentage of Non-Hispanic Asian individuals
Pop NH AIAN           int64  # Total population of Non-Hispanic American Indian and Alaska Native individuals
Pct NH AIAN         float64  # Percentage of Non-Hispanic American Indian and Alaska Native individuals
Pct NH NHPI         float64  # Percentage of Non-Hispanic Native Hawaiian and Other Pacific Islander individuals
Pct Agriculture     float64  # Percentage of land used for agriculture
"""