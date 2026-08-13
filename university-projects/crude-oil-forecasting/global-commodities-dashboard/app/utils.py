from datetime import date, datetime
from email.utils import parsedate_to_datetime
import json
import math
from pathlib import Path
from typing import Any, Optional
from urllib.request import Request, urlopen

import xlrd

from app.config import (
    BOOTSTRAP_CACHE_FILENAME,
    CACHE_VERSION,
    DATA_FILENAME,
    EIA_SOURCE_PAGE_URL,
    EIA_SOURCE_XLS_URL,
    NEWS_HISTORY_FILENAME,
)


def normalize_header(value: Any) -> str:
    text = str(value or "").strip().lower()
    for token in ["($/barrel)", "($/gallon)", ",", ".", "-", "(", ")", "/"]:
        text = text.replace(token, " ")
    return " ".join(text.split())


def candidate_data_paths() -> list[Path]:
    home = Path.home()
    cwd = Path.cwd()
    return [
        cwd / "data" / DATA_FILENAME,
        cwd / DATA_FILENAME,
        home / "Downloads" / DATA_FILENAME,
    ]


def cache_path() -> Path:
    return Path("data") / BOOTSTRAP_CACHE_FILENAME


def news_history_path() -> Path:
    return Path("data") / NEWS_HISTORY_FILENAME


def cache_metadata(file_path: Path) -> dict[str, Any]:
    stat = file_path.stat()
    return {
        "cache_version": CACHE_VERSION,
        "source_file": str(file_path),
        "source_mtime_ns": stat.st_mtime_ns,
        "source_size": stat.st_size,
    }


def http_get_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="ignore")


def http_last_modified(url: str) -> Optional[datetime]:
    request = Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        last_modified = response.headers.get("Last-Modified")
        if not last_modified:
            return None
        parsed = parsedate_to_datetime(last_modified)
        return parsed.astimezone().replace(tzinfo=None)


def parse_published_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
        return parsed.astimezone().replace(tzinfo=None)
    except Exception:
        return None


def parse_eia_dates(page_html: str) -> dict[str, Any]:
    import re

    release_match = re.search(r"Release Date:\s*([0-9/]+)", page_html, re.IGNORECASE)
    next_match = re.search(r"Next Release Date:\s*([0-9/]+)", page_html, re.IGNORECASE)
    release_date = release_match.group(1) if release_match else None
    next_release = next_match.group(1) if next_match else None
    return {
        "source_page": EIA_SOURCE_PAGE_URL,
        "download_url": EIA_SOURCE_XLS_URL,
        "release_date": release_date,
        "next_release_date": next_release,
        "notes": "EIA calculates weekly, monthly, and annual prices as unweighted averages of daily closing spot prices.",
        "cadence": "Weekly",
    }


def download_latest_xls(targets: list[Path]) -> None:
    request = Request(EIA_SOURCE_XLS_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=60) as response:
        payload = response.read()
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)


def find_data_file() -> Path:
    for path in candidate_data_paths():
        if path.exists():
            return path
    candidates = "\n".join(str(path) for path in candidate_data_paths())
    raise FileNotFoundError(f"Could not locate {DATA_FILENAME}. Checked:\n{candidates}")


def parse_date(book: xlrd.book.Book, cell_value: Any) -> Optional[date]:
    if isinstance(cell_value, (int, float)) and not math.isnan(cell_value):
        try:
            return xlrd.xldate_as_datetime(cell_value, book.datemode).date()
        except (ValueError, OverflowError):
            return None
    if isinstance(cell_value, str):
        text = cell_value.strip()
        if not text:
            return None
        for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y"):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue
    return None


def to_float(value: Any) -> float:
    if value in ("", None):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    try:
        cleaned = str(value).replace(",", "").strip()
        return float(cleaned) if cleaned else float("nan")
    except ValueError:
        return float("nan")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload))
