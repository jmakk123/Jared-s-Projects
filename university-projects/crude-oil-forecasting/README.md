# Crude Oil Price Forecasting

**Time Series Analysis & Forecasting · MS-ADS · Winter 2026**
Team: Danny Mendoza, Nick Dhaliwal, Jared Maksoud
Original repo: https://github.com/dannymendoza1/Time-Series-Final-Project

Daily crude oil prices for both WTI and Brent from the US Energy Information
Administration: roughly 9,700 observations spanning more than 35 years. Because oil markets
are globally integrated, these benchmarks move equities, currencies, and inflation
worldwide.

### Three questions

1. Can classical time-series models forecast prices and returns over a 63-day horizon,
   roughly one business quarter?
2. Can GARCH models capture the volatility clustering plainly visible in oil returns?
3. How many dollars per barrel did major geopolitical shocks actually move prices?

### Running it

Open the `.Rmd` files in RStudio and knit. Start with
`TS_Final_Project_EDA.Rmd`, then the modeling notebooks.
"Modeling (Fast Version)" skips the slower model searches.

### Data (not committed)

- **WTI and Brent daily spot prices** — EIA series `PET_PRI_SPT_S1_D`.
  https://www.eia.gov/dnav/pet/pet_pri_spt_s1_d.htm

Download the Excel file and place it in the project root as `PET_PRI_SPT_S1_D.xls`
(and in `global-commodities-dashboard/data/` for the dashboard).

`README.upstream.md` is the original team README.
