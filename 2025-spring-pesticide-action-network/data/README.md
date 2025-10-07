### Data

#### Geospatial data

- School districts `ca-school.parquet` 2020 census geographies of school distircts. California has a mixed of unified, elementary, secondary, and high school districts, so this file is a slight mixture, but has full coverage of the state
- Tracts `ca-tract.parquet` 2020 census tracts
- Zip Code Tabulation Areas `ca-zip.parquet` 2020 census zip code tabulation areas. Note - important - this do !not! cover the whole state. Not everywhere has a zip code. ~2% of pesticide data is not accounted for.
- Counties `cal_counties.parquet` 2020 census counties
- Sections `sections.parquet` California "section" units, which are roughly 1 mile x 1 mile squares, the atomic unit of the pesticide data.