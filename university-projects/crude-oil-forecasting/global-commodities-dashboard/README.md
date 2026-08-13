# Time-Series-Final-Project
# Global Commodities Dashboard

A FastAPI-based dashboard for exploring energy commodity spot prices, recent market headlines, and short-horizon forecast context in one place.

The app combines:

- historical commodity price series from the U.S. Energy Information Administration (EIA)
- interactive frontend charts and summary cards
- headline ingestion and NLP-based market signal scoring
- forecast overlays and commodity-specific context

## Covered commodities

- WTI crude
- Brent crude
- Conventional gasoline
- RBOB gasoline
- Heating oil
- ULSD diesel
- Jet fuel
- Propane

## Features

- Interactive dashboard UI served from a FastAPI app
- Historical commodity pricing loaded from the bundled EIA workbook
- Forecast summaries and analytics generated server-side
- Commodity news retrieval and ranking
- NLP interpretation of headline tone and price direction
- Cached bootstrap/data loading for faster startup

## Project structure

```text
global-commodities-dashboard/
├── app/
│   ├── config.py
│   ├── forecast.py
│   ├── loader.py
│   ├── news.py
│   ├── source.py
│   ├── state.py
│   ├── utils.py
│   └── web.py
├── data/
│   └── PET_PRI_SPT_S1_D.xls
├── static/
│   └── index.html
├── main.py
└── requirements.txt
```

Requirements

```text
Python 3.10+
pip
```

Installation

```text
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Run locally
uvicorn main:app --reload
Then open http://127.0.0.1:8000.
```

Data source
```text
The dashboard uses EIA petroleum spot price data. A workbook snapshot is bundled in data/PET_PRI_SPT_S1_D.xls.

NLP headline interpretation
Headline scoring is handled in app/news.py and estimates:

price-direction signal for the commodity
overall market tone from the headline set
This is decision-support tooling, not trading advice.
```

Notes
```text
Notes
Some news and NLP functionality depends on external requests and model availability.
Transformer inference may require additional local system support depending on your Python and Torch environment.
The bundled workbook allows the dashboard to start without a fresh data download.
``` 