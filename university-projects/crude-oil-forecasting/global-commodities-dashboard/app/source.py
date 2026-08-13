from datetime import datetime
from pathlib import Path
from typing import Any

from app import state
from app.config import DATA_FILENAME, EIA_SOURCE_PAGE_URL, EIA_SOURCE_XLS_URL, SOURCE_REFRESH_TTL
from app.utils import candidate_data_paths, download_latest_xls, find_data_file, http_get_text, http_last_modified, parse_eia_dates


def refresh_eia_source(force: bool = False) -> dict[str, Any]:
    downloads_path = Path.home() / "Downloads" / DATA_FILENAME
    workspace_path = Path.cwd() / "data" / DATA_FILENAME
    current_path = find_data_file() if any(path.exists() for path in candidate_data_paths()) else workspace_path
    now = datetime.utcnow()
    cached_checked_at = state.STORE.source_info.get("checked_at")
    if cached_checked_at and not force:
        try:
            checked_at = datetime.fromisoformat(cached_checked_at.replace("Z", ""))
            if now - checked_at < SOURCE_REFRESH_TTL:
                return state.STORE.source_info
        except ValueError:
            pass

    source_info = {
        "source_page": EIA_SOURCE_PAGE_URL,
        "download_url": EIA_SOURCE_XLS_URL,
        "checked_at": now.isoformat() + "Z",
        "cadence": "Weekly",
    }
    try:
        page_html = http_get_text(EIA_SOURCE_PAGE_URL)
        source_info.update(parse_eia_dates(page_html))
        remote_last_modified = http_last_modified(EIA_SOURCE_XLS_URL)
        local_mtime = datetime.utcfromtimestamp(current_path.stat().st_mtime) if current_path.exists() else None
        if force or (remote_last_modified and (local_mtime is None or remote_last_modified > local_mtime)):
            download_latest_xls([downloads_path, workspace_path])
            source_info["downloaded_latest"] = True
        else:
            source_info["downloaded_latest"] = False
    except Exception as exc:
        source_info["error"] = str(exc)
    state.STORE.source_info = source_info
    return source_info
