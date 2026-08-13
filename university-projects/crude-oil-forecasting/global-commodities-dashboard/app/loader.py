from datetime import datetime
from typing import Optional

import xlrd

from app.config import COMMODITIES, COMMODITY_BY_ID, DEFAULT_FORECAST_RANGE
from app.forecast import compute_dataset
from app.news import load_news_history
from app.state import CommodityDataset, CommodityStore
from app.utils import cache_metadata, cache_path, find_data_file, normalize_header, parse_date, read_json, to_float, write_json


def load_cached_store(file_path) -> Optional[CommodityStore]:
    current_cache_path = cache_path()
    if not current_cache_path.exists():
        return None
    try:
        cached = read_json(current_cache_path)
    except Exception:
        return None
    if cached.get("metadata") != cache_metadata(file_path):
        return None
    payload = cached.get("payload")
    if not isinstance(payload, dict):
        return None
    store = CommodityStore()
    store.file_path = file_path
    store.loaded_at = payload.get("loaded_at")
    store.bootstrap_cache = payload
    store.source_info = payload.get("source_info", {})
    for commodity in payload.get("commodities", []):
        meta = {key: commodity[key] for key in COMMODITY_BY_ID[commodity["id"]].keys() if key in commodity}
        store.datasets[commodity["id"]] = CommodityDataset(
            meta=meta,
            series=commodity.get("data", []),
            stats=commodity.get("stats", {}),
            analytics=commodity.get("analytics", {}),
            forecast=commodity.get("forecast", {}),
        )
        store.forecast_cache[commodity["id"]] = {DEFAULT_FORECAST_RANGE: commodity.get("forecast", {})}
    return store


def write_cached_store(file_path, store: CommodityStore) -> None:
    payload = store.bootstrap_cache or store.bootstrap_payload()
    write_json(cache_path(), {"metadata": cache_metadata(file_path), "payload": payload})


def find_header_row(sheet: xlrd.sheet.Sheet) -> int:
    for row_idx in range(sheet.nrows):
        first = str(sheet.cell_value(row_idx, 0)).strip().lower()
        if first == "date":
            return row_idx
    raise ValueError("Could not find header row beginning with 'Date'.")


def find_column_index(sheet: xlrd.sheet.Sheet, header_row_idx: int, header_match: str) -> int:
    headers = [normalize_header(sheet.cell_value(header_row_idx, col_idx)) for col_idx in range(sheet.ncols)]
    normalized_match = normalize_header(header_match)
    for idx, header in enumerate(headers):
        if normalized_match == header:
            return idx
    raise ValueError(f"Column not found for {header_match!r} in sheet {sheet.name!r}.")


def load_store() -> CommodityStore:
    file_path = find_data_file()
    cached_store = load_cached_store(file_path)
    if cached_store is not None:
        cached_store.news_history = load_news_history()
        return cached_store

    book = xlrd.open_workbook(file_path.as_posix())
    rows_by_commodity: dict[str, list[tuple]] = {commodity["id"]: [] for commodity in COMMODITIES}
    for commodity in COMMODITIES:
        sheet = book.sheet_by_name(commodity["sheet"])
        header_row_idx = find_header_row(sheet)
        col_idx = find_column_index(sheet, header_row_idx, commodity["header_match"])
        for row_idx in range(header_row_idx + 1, sheet.nrows):
            current_date = parse_date(book, sheet.cell_value(row_idx, 0))
            if current_date is None:
                continue
            value = to_float(sheet.cell_value(row_idx, col_idx))
            if value != value:
                continue
            rows_by_commodity[commodity["id"]].append((current_date, value))

    store = CommodityStore()
    store.file_path = file_path
    store.loaded_at = datetime.utcnow().isoformat() + "Z"
    store.news_history = load_news_history()
    for commodity in COMMODITIES:
        dataset = compute_dataset(commodity, rows_by_commodity[commodity["id"]])
        store.datasets[commodity["id"]] = dataset
        store.forecast_cache[commodity["id"]] = {DEFAULT_FORECAST_RANGE: dataset.forecast}
    store.bootstrap_payload()
    write_cached_store(file_path, store)
    return store
