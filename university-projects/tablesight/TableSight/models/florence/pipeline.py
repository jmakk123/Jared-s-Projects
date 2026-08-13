"""
Florence-2 runner — fine-tuned LoRA on `microsoft/Florence-2-base`.

Two prompts:
  <PARSE_TABLE_STRUCTURE>  → skeleton HTML (used by the fine-tuned adapter)
  <OCR_WITH_REGION>        → text + bbox for content injection

Pipeline:
  1. parse_structure(image)   → skeleton HTML (no text)
  2. ocr_with_region(image)   → list of (text, bbox)
  3. inject_content(skeleton, ocr_results) → final HTML

The LoRA adapter is loaded with PEFT in eval mode (no merging).
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


# ── helpers ─────────────────────────────────────────────────────────────────

BASE_MODEL_ID = "microsoft/Florence-2-base"
PROMPT_STRUCT = "<PARSE_TABLE_STRUCTURE>"
PROMPT_OCR    = "<OCR_WITH_REGION>"


def _extract_html_table(text: str) -> str:
    """Pull the first <table>...</table> block out of generated text.

    Florence sometimes wraps output in extra commentary; this is a robust
    grabber that returns '' if nothing table-like is found.
    """
    if not text:
        return ""
    m = re.search(r"<table[\s\S]*?</table>", text, flags=re.IGNORECASE)
    if m:
        return m.group(0)
    # Some checkpoints emit <tr>...</tr> directly; wrap them.
    if "<tr" in text.lower():
        body = text[text.lower().find("<tr"):]
        end = body.lower().rfind("</tr>")
        if end >= 0:
            body = body[: end + len("</tr>")]
        return f"<table>{body}</table>"
    return ""


def _inject_content(skeleton_html: str, ocr_items: List[Tuple[str, Tuple[float, float, float, float]]]) -> str:
    """Place OCR text into the skeleton's <td>/<th> cells by spatial order.

    Strategy: skeleton order ≈ row-major reading order. We sort OCR items by
    (y_center, x_center) and assign them one-per-cell into the skeleton.
    Cells we don't have text for stay empty.
    """
    from lxml import html as lhtml

    if not skeleton_html or not ocr_items:
        return skeleton_html or ""
    try:
        tbl = lhtml.fromstring(skeleton_html)
    except Exception:
        return skeleton_html

    cells = tbl.xpath(".//td|.//th")
    items = sorted(
        ocr_items,
        key=lambda it: ((it[1][1] + it[1][3]) / 2, (it[1][0] + it[1][2]) / 2),
    )
    for cell, (text, _) in zip(cells, items):
        cell.text = text or ""
    return lhtml.tostring(tbl, encoding="unicode")


# ── runner ──────────────────────────────────────────────────────────────────


class FlorenceRunner:
    """Lazy-loads Florence-2-base + LoRA adapter."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        base_model: str = BASE_MODEL_ID,
        do_ocr_injection: bool = True,
        max_new_tokens: int = 1024,
        num_beams: int = 3,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.base_model = base_model
        self.do_ocr_injection = do_ocr_injection
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM

        self._torch = torch
        # Use fp16 only on CUDA. MPS + fp16 + PEFT triggers
        # "Input type (float) and bias type (c10::Half) should be the same"
        # because PEFT adapter layers stay fp32 while the base is fp16.
        # CPU/MPS use fp32 throughout for correctness.
        self._dtype = torch.float16 if device == "cuda" else torch.float32
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                trust_remote_code=True,
                torch_dtype=self._dtype,
            )

        # Load LoRA adapter on top if a checkpoint path is provided
        if checkpoint_path and Path(checkpoint_path).exists():
            try:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, checkpoint_path)
                self.has_adapter = True
            except Exception as exc:
                warnings.warn(f"Florence: failed to attach LoRA adapter ({exc}); using base model.")
                self.has_adapter = False
        else:
            self.has_adapter = False

        # Force every parameter & buffer to the same dtype after PEFT wrapping
        if self._dtype == torch.float32:
            self.model = self.model.float()
        else:
            self.model = self.model.half()
        self.model.to(device).eval()

    # ── core generation ─────────────────────────────────────────────────

    @property
    def _device(self):
        return next(self.model.parameters()).device

    def _generate(self, image: Image.Image, prompt: str) -> str:
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self._device)
        # Ensure pixel_values has the same dtype as the model weights so we
        # never hit "Input type (float) and bias type (c10::Half) should be
        # the same".
        pixel_values = inputs["pixel_values"].to(self._dtype)
        with self._torch.no_grad():
            generated = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=pixel_values,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                do_sample=False,
            )
        text = self.processor.batch_decode(generated, skip_special_tokens=False)[0]
        try:
            parsed = self.processor.post_process_generation(
                text, task=prompt, image_size=(image.width, image.height)
            )
            # `parsed` is typically  {task_prompt: <result>}.  For OCR_WITH_REGION
            # the inner value is a dict like {'quad_boxes': [[8 floats]], 'labels': [str]}.
            # Return it as-is so `_parse_ocr_with_region` can extract from the dict path.
            if isinstance(parsed, dict) and prompt in parsed:
                inner = parsed[prompt]
                return inner   # may be dict OR str — downstream handles both
            if isinstance(parsed, dict):
                return parsed
            return str(parsed)
        except Exception:
            return text

    # ── public API ───────────────────────────────────────────────────────

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        if image.mode != "RGB":
            image = image.convert("RGB")

        # If no LoRA adapter, the `<PARSE_TABLE_STRUCTURE>`
        # prompt is meaningless to base Florence-2 and it dumps raw `<poly><loc_*>`
        # garbage.  Instead, use the standard `<OCR_WITH_REGION>` task (which the
        # base model is natively trained on) and cluster the resulting word boxes
        # into a table grid — same approach as SPARTAN, but with Florence's OCR.
        if not self.has_adapter:
            return self._predict_ocr_fallback(image)

        # ── Adapter path ─────────────────────
        struct_raw = self._generate(image, PROMPT_STRUCT)
        skeleton = _extract_html_table(struct_raw if isinstance(struct_raw, str) else "")
        meta: Dict[str, Any] = {
            "has_adapter": True,
            "mode": "structure_then_inject",
            "prompt_struct": PROMPT_STRUCT,
            "raw_struct_preview": (struct_raw[:200] if isinstance(struct_raw, str) else ""),
        }
        final_html = skeleton
        if self.do_ocr_injection and skeleton:
            ocr_raw = self._generate(image, PROMPT_OCR)
            ocr_items = self._parse_ocr_with_region(ocr_raw)
            meta["ocr_items"] = len(ocr_items)
            if ocr_items:
                final_html = _inject_content(skeleton, ocr_items)
        return {"html": final_html, "metadata": meta}

    # ── OCR-with-cluster fallback (no adapter present) ──────────────────────

    def _predict_ocr_fallback(self, image: Image.Image) -> Dict[str, Any]:
        """Build the table from Florence-2's `<OCR_WITH_REGION>` output via
        the same grid-clustering logic used in SPARTAN.  Produces a valid
        <table> with structure even when the LoRA adapter isn't available."""
        ocr_raw = self._generate(image, PROMPT_OCR)
        items = self._parse_ocr_with_region(ocr_raw)
        meta: Dict[str, Any] = {
            "has_adapter": False,
            "mode": "ocr_with_region_cluster",
            "n_ocr_items": len(items),
        }
        if not items:
            return {"html": "<table></table>",
                    "metadata": {**meta, "reason": "no_ocr_output"}}

        # Convert OCR items → word dicts compatible with the SPARTAN clusterers
        words = []
        for text, (x1, y1, x2, y2) in items:
            words.append({
                "x":  float(x1), "y":  float(y1),
                "w":  float(x2 - x1), "h": float(y2 - y1),
                "cx": float((x1 + x2) / 2), "cy": float((y1 + y2) / 2),
                "text": text, "conf": 1.0,
            })

        # Re-use SPARTAN's helpers (already imported as soon as we need them)
        from ..spartan.pipeline import (
            _cluster_rows, _estimate_n_cols, _column_centers,
            _assign_to_grid, _detect_header_rows, _detect_spans,
            _grid_to_html,
        )
        rows = _cluster_rows(words, tol_frac=0.55)
        n_cols = _estimate_n_cols(rows)
        col_centers = _column_centers(rows, n_cols)
        grid = _assign_to_grid(rows, col_centers)
        header_rows = _detect_header_rows(grid, max_rows=3)
        spans = _detect_spans(rows, col_centers)
        html = _grid_to_html(grid, header_rows=header_rows, spans=spans)

        meta.update({
            "n_rows":      len(grid),
            "n_cols":      n_cols,
            "n_cells":     sum(1 for r in grid for c in r if c.strip()),
            "header_rows": header_rows,
            "n_spans":     len(spans),
        })
        return {"html": html, "metadata": meta}

    # ── parsing utilities ────────────────────────────────────────────────

    @staticmethod
    def _parse_ocr_with_region(raw) -> List[Tuple[str, Tuple[float, float, float, float]]]:
        """Florence post-process for OCR_WITH_REGION can return either:
            - a string with <loc_*> tokens interleaved
            - a dict {'quad_boxes': [...], 'labels': [...]}
        """
        items: List[Tuple[str, Tuple[float, float, float, float]]] = []
        if isinstance(raw, dict):
            quads = raw.get("quad_boxes") or raw.get("polygons") or []
            labels = raw.get("labels") or raw.get("texts") or []
            for q, t in zip(quads, labels):
                if not q:
                    continue
                xs = q[0::2]; ys = q[1::2]
                items.append((str(t), (min(xs), min(ys), max(xs), max(ys))))
            return items
        if isinstance(raw, str):
            for m in re.finditer(
                r"(?P<txt>[^<]+)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>",
                raw,
            ):
                txt = m.group("txt").strip()
                x1, y1, x2, y2 = (int(m.group(i)) for i in (2, 3, 4, 5))
                items.append((txt, (x1, y1, x2, y2)))
        return items


__all__ = ["FlorenceRunner"]
