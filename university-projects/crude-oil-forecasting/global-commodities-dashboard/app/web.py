from contextlib import asynccontextmanager
from datetime import date
import json
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app import state
from app.config import COMMODITY_BY_ID, DEFAULT_FORECAST_RANGE, INDEX_TEMPLATE_PATH, RANGE_LABEL_TO_DAYS, ZERO_SHOT_MODEL_ID
from app.forecast import apply_news_risk_to_forecast, build_forecast_for_range
from app.loader import load_store
from app.news import fetch_news_for_commodity, summarize_news_history, zero_shot_classifier
from app.source import refresh_eia_source
from app.utils import cache_metadata, find_data_file


def render_index_html() -> str:
    payload = state.STORE.bootstrap_cache or state.STORE.bootstrap_payload()
    bootstrap_script = (
        "<script id=\"bootstrap-data\">"
        f"window.__BOOTSTRAP__ = {json.dumps(payload)};"
        "</script>"
    )
    return INDEX_TEMPLATE_PATH.read_text().replace("</head>", f"{bootstrap_script}\n  </head>", 1)


def ensure_store_current(force_source_refresh: bool = False) -> None:
    info = refresh_eia_source(force=force_source_refresh)
    current_file = find_data_file()
    current_meta = cache_metadata(current_file)
    loaded_meta = cache_metadata(state.STORE.file_path) if state.STORE.file_path and state.STORE.file_path.exists() else None
    if not state.STORE.datasets or loaded_meta != current_meta:
        state.STORE = load_store()
    state.STORE.source_info = info


@asynccontextmanager
async def lifespan(_: FastAPI):
    zero_shot_classifier()
    refresh_eia_source(force=False)
    state.STORE = load_store()
    state.STORE.source_info = refresh_eia_source(force=False)
    print("Global commodities dashboard ready at http://localhost:8000")
    for commodity_id, dataset in state.STORE.datasets.items():
        print(f"- {commodity_id}: {dataset.stats['points']} points loaded")
    yield


app = FastAPI(title="Global Commodities Dashboard", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index() -> HTMLResponse:
    ensure_store_current(force_source_refresh=False)
    return HTMLResponse(render_index_html())


@app.get("/api/commodities")
def commodities() -> list[dict[str, Any]]:
    ensure_store_current(force_source_refresh=False)
    return [{**dataset.meta, "stats": dataset.stats} for dataset in state.STORE.datasets.values()]


@app.get("/api/bootstrap")
def bootstrap() -> dict[str, Any]:
    ensure_store_current(force_source_refresh=False)
    return state.STORE.bootstrap_payload()


@app.get("/api/source")
def source_status() -> dict[str, Any]:
    ensure_store_current(force_source_refresh=False)
    return state.STORE.source_info


@app.post("/api/source/refresh")
def source_refresh() -> dict[str, Any]:
    ensure_store_current(force_source_refresh=True)
    return state.STORE.source_info


@app.get("/api/data/{commodity_id}")
def commodity_data(commodity_id: str, start: Optional[str] = Query(default=None), end: Optional[str] = Query(default=None)) -> list[dict[str, Any]]:
    ensure_store_current(force_source_refresh=False)
    dataset = state.STORE.datasets.get(commodity_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Unknown commodity id")

    start_date = date.fromisoformat(start) if start else None
    end_date = date.fromisoformat(end) if end else None
    output = []
    for point in dataset.series:
        point_date = date.fromisoformat(point["date"])
        if start_date and point_date < start_date:
            continue
        if end_date and point_date > end_date:
            continue
        output.append(point)
    return output


@app.get("/api/stats/{commodity_id}")
def commodity_stats(commodity_id: str) -> dict[str, Any]:
    dataset = state.STORE.datasets.get(commodity_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Unknown commodity id")
    return dataset.stats


@app.get("/api/forecast/{commodity_id}")
def commodity_forecast(commodity_id: str, range_label: str = Query(default=DEFAULT_FORECAST_RANGE, alias="range")) -> dict[str, Any]:
    ensure_store_current(force_source_refresh=False)
    dataset = state.STORE.datasets.get(commodity_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Unknown commodity id")
    if range_label not in RANGE_LABEL_TO_DAYS:
        raise HTTPException(status_code=400, detail="Unsupported range")
    commodity_cache = state.STORE.forecast_cache.setdefault(commodity_id, {})
    if range_label not in commodity_cache:
        commodity_cache[range_label] = build_forecast_for_range(dataset.series, range_label)
    news_context = fetch_news_for_commodity(commodity_id)
    return apply_news_risk_to_forecast(commodity_cache[range_label], news_context)


@app.get("/api/news/{commodity_id}")
def commodity_news(commodity_id: str) -> dict[str, Any]:
    ensure_store_current(force_source_refresh=False)
    if commodity_id not in COMMODITY_BY_ID:
        raise HTTPException(status_code=404, detail="Unknown commodity id")
    return fetch_news_for_commodity(commodity_id)


@app.get("/api/news-history/{commodity_id}")
def commodity_news_history(
    commodity_id: str,
    days: int = Query(default=30, ge=1, le=90),
    top_k: int = Query(default=3, ge=1, le=5),
) -> dict[str, Any]:
    ensure_store_current(force_source_refresh=False)
    if commodity_id not in COMMODITY_BY_ID:
        raise HTTPException(status_code=404, detail="Unknown commodity id")
    payload = fetch_news_for_commodity(commodity_id)
    history = summarize_news_history(state.STORE.news_history.get(commodity_id, []), days=days, top_k=top_k)
    return {
        "commodity_id": commodity_id,
        "days": days,
        "top_k": top_k,
        "daily_history": history,
        "current_event_score": payload.get("event_score"),
        "current_event_label": payload.get("event_label"),
        "transformer_model": ZERO_SHOT_MODEL_ID,
    }
