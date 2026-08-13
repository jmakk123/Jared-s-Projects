# Forward 12-Month Drawdown Prediction

**Machine Learning II · MS-ADS · Spring 2026**
Team: Nick Dhaliwal, Jared Maksoud, Nicholas Mikhail, Yung Chyi Yang

A dual-stream neural network forecasting the maximum peak-to-trough drawdown a US public
company will experience over the next 12 months. An LSTM reads five years of Compustat
accounting ratios, an MLP reads seven CRSP-derived price features, and the two streams fuse
into a single forecast bounded in `[-1, 0]`.

### Results

Test fold fyear 2020–2023, 15,311 firm-years, 3-seed ensemble. Beats the volatility-only
baseline on every primary metric.

| Metric | Value |
|---|---|
| MAE | 0.121 |
| RMSE | 0.161 |
| R² | 0.410 |
| PR-AUC at −30% | 0.852 |
| Brier score | 0.226 |
| Within-year Spearman | 0.666 |
| Top-decile precision | 0.487 |

### Where the code lives

This project has its own repository with full history and a deployed evaluation site:

- **Live site:** https://jmakk123.github.io/MLII-Final/
- **Source:** https://github.com/jmakk123/MLII-Final

It is not duplicated here.

### Data

Not redistributable — both sources are license-restricted and accessed through WRDS.

- **Compustat North America** (annual fundamentals) via [WRDS](https://wrds-www.wharton.upenn.edu/)
- **CRSP Daily Stock File** (prices, returns) via [WRDS](https://wrds-www.wharton.upenn.edu/)

A WRDS subscription is required. See the project repo for the extraction queries.
