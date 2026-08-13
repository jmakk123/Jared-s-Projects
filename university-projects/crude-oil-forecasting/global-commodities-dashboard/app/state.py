from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class CommodityDataset:
    meta: dict[str, Any]
    series: list[dict[str, Any]]
    stats: dict[str, Any]
    analytics: dict[str, Any]
    forecast: dict[str, Any]


class CommodityStore:
    def __init__(self) -> None:
        self.file_path: Optional[Path] = None
        self.loaded_at: Optional[str] = None
        self.datasets: dict[str, CommodityDataset] = {}
        self.bootstrap_cache: Optional[dict[str, Any]] = None
        self.forecast_cache: dict[str, dict[str, Any]] = {}
        self.source_info: dict[str, Any] = {}
        self.news_cache: dict[str, dict[str, Any]] = {}
        self.news_history: dict[str, list[dict[str, Any]]] = {}

    def bootstrap_payload(self) -> dict[str, Any]:
        payload = {
            "source_file": str(self.file_path) if self.file_path else None,
            "loaded_at": self.loaded_at,
            "source_info": self.source_info,
            "commodities": [
                {
                    **dataset.meta,
                    "stats": dataset.stats,
                    "analytics": dataset.analytics,
                    "forecast": dataset.forecast,
                    "data": dataset.series,
                }
                for dataset in self.datasets.values()
            ],
        }
        self.bootstrap_cache = payload
        return payload


STORE = CommodityStore()
ZERO_SHOT_CLASSIFIER = None
ZERO_SHOT_LOAD_ERROR: Optional[str] = None
HEADLINE_SIGNAL_CACHE: dict[tuple[str, str], dict[str, Any]] = {}
