"""Dashboard-side helper: load all three models with graceful failure tracking."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from TableSight.models.inference import TableExtractor


def load_config(path: str = "dashboard/config.yaml") -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        # try relative to this file's package root
        alt = Path(__file__).resolve().parents[2] / "dashboard" / "config.yaml"
        if alt.exists():
            p = alt
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text()) or {}


def load_models(cfg: Dict[str, Any]) -> Tuple[Dict[str, Optional[TableExtractor]], Dict[str, str]]:
    """Returns (extractors, status_messages).

    `status_messages[name]` is "" on success or the exception string on failure.
    """
    device = cfg.get("device", "auto")
    ck = cfg.get("checkpoints", {})

    extractors: Dict[str, Optional[TableExtractor]] = {}
    status: Dict[str, str] = {}
    for name in ("florence", "unitable", "spartan"):
        ckpt = ck.get(name)
        try:
            extractors[name] = TableExtractor(name, checkpoint_path=ckpt, device=device,
                                              **cfg.get(f"{name}_kwargs", {}))
            status[name] = ""
        except Exception as exc:
            extractors[name] = None
            status[name] = f"{type(exc).__name__}: {exc}"
    return extractors, status


__all__ = ["load_config", "load_models"]
