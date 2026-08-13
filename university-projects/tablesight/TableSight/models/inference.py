"""
Unified TableExtractor interface.

All three model runners (Florence, UniTable, Pytesseract) conform to this
interface so the dashboard and the batch-evaluation harness can call them
identically.

    extractor = TableExtractor('florence', checkpoint_path='/path/to/adapter')
    result    = extractor.predict(pil_image)
    # result = {'html': '<table>...</table>', 'time_s': 0.42, 'metadata': {...}}

Runners are lazily imported inside `__init__` so that loading one model does
not require dependencies of the others (Florence needs `transformers + peft`,
UniTable needs the poloclub model definitions, SPARTAN needs cv2 +
pytesseract).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from PIL import Image

ModelName = Literal["florence", "unitable", "spartan"]


class TableExtractor:
    """Common front-end to the three table-extraction models."""

    def __init__(
        self,
        model_name: ModelName,
        checkpoint_path: Optional[str] = None,
        device: str = "auto",
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.device = self._resolve_device(device)
        self._runner = self._build_runner(**kwargs)

    # ── runner construction ──────────────────────────────────────────────

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _build_runner(self, **kwargs: Any):
        if self.model_name == "florence":
            from .florence.pipeline import FlorenceRunner
            return FlorenceRunner(
                checkpoint_path=self.checkpoint_path,
                device=self.device,
                **kwargs,
            )
        if self.model_name == "unitable":
            from .unitable.pipeline import UniTableRunner
            return UniTableRunner(
                checkpoint_path=self.checkpoint_path,
                device=self.device,
                **kwargs,
            )
        if self.model_name == "spartan":
            from .spartan.pipeline import SpartanRunner
            return SpartanRunner(
                checkpoint_path=self.checkpoint_path,
                device=self.device,
                **kwargs,
            )
        raise ValueError(f"Unknown model_name: {self.model_name!r}")

    # ── inference ────────────────────────────────────────────────────────

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Run inference on a single PIL image.

        Returns
        -------
        dict with keys:
            html      str  : predicted HTML <table>
            time_s    float: wall-clock inference time
            metadata  dict : runner-specific extras (cell counts, raw tokens, …)
        """
        t0 = time.perf_counter()
        out = self._runner.predict(image)
        elapsed = time.perf_counter() - t0

        html = out.get("html", "") if isinstance(out, dict) else str(out)
        meta = out.get("metadata", {}) if isinstance(out, dict) else {}
        return {
            "html": html,
            "time_s": float(elapsed),
            "metadata": {
                "model": self.model_name,
                "device": self.device,
                **meta,
            },
        }

    def predict_batch(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Run inference on a batch. Default impl is per-image; runners that
        support real batch can override `predict_batch` on their side."""
        if hasattr(self._runner, "predict_batch"):
            t0 = time.perf_counter()
            outs = self._runner.predict_batch(images)
            elapsed = time.perf_counter() - t0
            results = []
            n = max(1, len(outs))
            for o in outs:
                html = o.get("html", "") if isinstance(o, dict) else str(o)
                meta = o.get("metadata", {}) if isinstance(o, dict) else {}
                results.append({
                    "html": html,
                    "time_s": elapsed / n,
                    "metadata": {"model": self.model_name, "device": self.device, **meta},
                })
            return results
        return [self.predict(img) for img in images]

    # ── ergonomics ───────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self.model_name

    def __repr__(self) -> str:
        return f"TableExtractor(model={self.model_name!r}, device={self.device!r})"


def load_all(
    checkpoints: Dict[str, Optional[str]],
    device: str = "auto",
) -> Dict[str, TableExtractor]:
    """Convenience helper used by the dashboard's startup path.

    Parameters
    ----------
    checkpoints : dict mapping model_name → checkpoint_path (or None)
    device      : 'auto', 'cuda', 'mps', 'cpu'

    Returns
    -------
    {'florence': TableExtractor | None, 'unitable': …, 'spartan': …}
        Any model that fails to load is recorded as None and a warning is
        printed (the dashboard surfaces this as a ✗ indicator).
    """
    out: Dict[str, TableExtractor] = {}
    for name, ckpt in checkpoints.items():
        try:
            out[name] = TableExtractor(name, checkpoint_path=ckpt, device=device)
            print(f"✓ {name} loaded ({device})")
        except Exception as exc:
            out[name] = None
            print(f"✗ {name} failed to load: {exc}")
    return out


__all__ = ["TableExtractor", "load_all", "ModelName"]
