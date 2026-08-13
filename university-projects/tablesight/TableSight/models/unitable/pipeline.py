"""UniTable runner — uses the canonical poloclub/unitable inference recipe.

This is a faithful port of `notebooks/full_pipeline.ipynb` from the cloned
poloclub/unitable repo at:
    TableSight/models/unitable/unitable/

Three model passes:
    1. Structure   prefix=[html],  size=448x448,  whitelist=VALID_HTML_TOKEN
    2. BBox        prefix=[bbox],  size=448x448,  whitelist=VALID_BBOX_TOKEN[:449]
    3. Content     prefix=[cell],  size=112x448 PER CELL,  blacklist=INVALID_CELL_TOKEN

Critical implementation details that the earlier vendored arch got wrong:
    - prefix is JUST [task] (not <sos>[task]) — the model treats the bare
      task token as the SOS implicitly because pos-embedding starts at 0
    - decode signature is decode(memory, tgt, tgt_mask, tgt_padding_mask)
      with memory FIRST.  generator is a SEPARATE call after decode.
    - encode adds pos_embed to the IMAGE patches before running the encoder
      and applies a top-level norm to the memory.  This is what was missing
      from the vendored reverse-engineering — without it the encoder produces
      constant features and both downstream decoders collapse.
    - content model uses 112x448 cell crops (aspect 1:4), NOT 448x448.

The fine-tuned content checkpoint from Ryan replaces the original.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image


# ──────────────────────── path setup ────────────────────────────────────────

HERE = Path(__file__).resolve()
HANDOFF_DIR = HERE.parent                                  # this folder is the bundle
UNITABLE_REPO = HANDOFF_DIR / "source"          # poloclub clone

# Make `src.*` importable (the poloclub repo's package root)
if UNITABLE_REPO.exists() and str(UNITABLE_REPO) not in sys.path:
    sys.path.insert(0, str(UNITABLE_REPO))
if str(HANDOFF_DIR) not in sys.path:
    sys.path.insert(0, str(HANDOFF_DIR))


# ──────────────────────── image preprocessing ───────────────────────────────

def _image_to_tensor(image: Image.Image, size: Tuple[int, int], device):
    """Match poloclub's `image_to_tensor` exactly (incl. FinTabNet stats)."""
    import torch
    import torchvision.transforms as T
    if image.mode != "RGB":
        image = image.convert("RGB")
    tfm = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=[0.86597056, 0.88463002, 0.87491087],
                    std=[0.20686628, 0.18201602, 0.18485524]),
    ])
    return tfm(image).to(device).unsqueeze(0)


# ──────────────────────── autoregressive decode ─────────────────────────────

def _autoregressive_decode(model, image_tensor, prefix_ids: Sequence[int],
                            max_decode_len: int, eos_id: int, device,
                            token_whitelist: Optional[Sequence[int]] = None,
                            token_blacklist: Optional[Sequence[int]] = None):
    """Faithful port of poloclub's `autoregressive_decode`."""
    import torch
    from src.utils.data import subsequent_mask, pred_token_within_range, greedy_sampling

    model.eval()
    with torch.no_grad():
        memory = model.encode(image_tensor)
        context = torch.tensor(prefix_ids, dtype=torch.int32).repeat(
            image_tensor.shape[0], 1).to(device)

    for _ in range(max_decode_len):
        if all(eos_id in k.tolist() for k in context):
            break
        with torch.no_grad():
            causal_mask = subsequent_mask(context.shape[1]).to(device)
            logits = model.decode(memory, context, tgt_mask=causal_mask,
                                   tgt_padding_mask=None)
            logits = model.generator(logits)[:, -1, :]

        logits = pred_token_within_range(
            logits.detach(),
            white_list=list(token_whitelist) if token_whitelist else None,
            black_list=list(token_blacklist) if token_blacklist else None,
        )
        _, next_tokens = greedy_sampling(logits)
        context = torch.cat([context, next_tokens], dim=1)
    return context


# ──────────────────────── HTML assembly ─────────────────────────────────────

def _build_html(structure_tokens: List[str], cell_contents: List[str]) -> str:
    """Use poloclub's build_table_from_html_and_cell, then wrap in <table>."""
    from src.utils.data import build_table_from_html_and_cell
    html_list = build_table_from_html_and_cell(structure_tokens, cell_contents)
    # build_table_from_html_and_cell returns a list of HTML fragments
    body = "".join(html_list)
    if not body.lower().startswith("<table"):
        body = f"<table>{body}</table>"
    return body


# ──────────────────────── runner ────────────────────────────────────────────

class UniTableRunner:
    """3-pass UniTable inference, canonical poloclub recipe."""

    def __init__(self,
                  checkpoint_path: Optional[str] = None,
                  device: str = "cpu",
                  use_cpu: bool = True,
                  max_struct_tokens: int = 512,
                  max_bbox_tokens: int = 1024,
                  max_content_tokens: int = 200,
                  **_: Any):
        self.checkpoint_path = checkpoint_path
        self.device_str = device
        self.use_cpu = use_cpu
        self.max_struct_tokens = max_struct_tokens
        self.max_bbox_tokens = max_bbox_tokens
        self.max_content_tokens = max_content_tokens

        bundle_root = Path(checkpoint_path) if checkpoint_path else HANDOFF_DIR
        if not bundle_root.exists():
            raise FileNotFoundError(
                f"UniTable bundle not found at {bundle_root}. "
                f"Set checkpoints.unitable in dashboard/config.yaml.")

        # Defer torch import; helper does the real work
        import torch
        self._torch = torch

        from loader import load_unitable_models  # type: ignore
        (self.struct_model, self.bbox_model, self.content_model,
         self.vocab_s, self.vocab_b, self.vocab_c) = load_unitable_models(
            base_dir=str(bundle_root), use_cpu=use_cpu)
        self.device = "cuda" if (not use_cpu and torch.cuda.is_available()) else "cpu"

        # Precompute token whitelists/blacklists (mirrors trainer.utils)
        from src.vocab.constant import (HTML_TOKENS, BBOX_TOKENS, TASK_TOKENS,
                                          RESERVED_TOKENS)
        self.VALID_HTML_TOKEN = ["<eos>"] + HTML_TOKENS
        self.VALID_BBOX_TOKEN_449 = (["<eos>"] + BBOX_TOKENS)[:449]
        self.INVALID_CELL_TOKEN = (["<sos>", "<pad>", "<empty>", "<sep>"]
                                    + TASK_TOKENS + RESERVED_TOKENS)
        self._white_html = [self.vocab_s.token_to_id(t) for t in self.VALID_HTML_TOKEN
                             if self.vocab_s.token_to_id(t) is not None]
        self._white_bbox = [self.vocab_b.token_to_id(t) for t in self.VALID_BBOX_TOKEN_449
                             if self.vocab_b.token_to_id(t) is not None]
        self._black_cell = [self.vocab_c.token_to_id(t) for t in self.INVALID_CELL_TOKEN
                             if self.vocab_c.token_to_id(t) is not None]

    # ── inference ─────────────────────────────────────────────────────────

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        if image.mode != "RGB":
            image = image.convert("RGB")

        # ── Stage 1: structure ──────────────────────────────────────────
        img_tensor = _image_to_tensor(image, size=(448, 448), device=self.device)
        struct_ids = _autoregressive_decode(
            self.struct_model, img_tensor,
            prefix_ids=[self.vocab_s.token_to_id("[html]")],
            max_decode_len=self.max_struct_tokens,
            eos_id=self.vocab_s.token_to_id("<eos>"),
            device=self.device,
            token_whitelist=self._white_html,
        )
        struct_str = self.vocab_s.decode(
            struct_ids.detach().cpu().numpy()[0].tolist(),
            skip_special_tokens=False)
        from src.utils.data import html_str_to_token_list, cell_str_to_token_list
        struct_tokens = html_str_to_token_list(struct_str)

        # ── Stage 2: bbox ───────────────────────────────────────────────
        bbox_ids = _autoregressive_decode(
            self.bbox_model, img_tensor,
            prefix_ids=[self.vocab_b.token_to_id("[bbox]")],
            max_decode_len=self.max_bbox_tokens,
            eos_id=self.vocab_b.token_to_id("<eos>"),
            device=self.device,
            token_whitelist=self._white_bbox,
        )
        bbox_str = self.vocab_b.decode(
            bbox_ids.detach().cpu().numpy()[0].tolist(),
            skip_special_tokens=False)
        bboxes = self._parse_bboxes(bbox_str, image.size)

        # ── Stage 3: content ────────────────────────────────────────────
        cell_contents: List[str] = []
        if bboxes:
            import torch
            cell_tensors = [
                _image_to_tensor(image.crop(b), size=(112, 448),
                                  device=self.device)
                for b in bboxes
            ]
            batch = torch.cat(cell_tensors, dim=0)
            cell_ids = _autoregressive_decode(
                self.content_model, batch,
                prefix_ids=[self.vocab_c.token_to_id("[cell]")],
                max_decode_len=self.max_content_tokens,
                eos_id=self.vocab_c.token_to_id("<eos>"),
                device=self.device,
                token_blacklist=self._black_cell,
            )
            cell_strs = self.vocab_c.decode_batch(
                cell_ids.detach().cpu().numpy().tolist(),
                skip_special_tokens=False)
            cell_contents = [cell_str_to_token_list(s) for s in cell_strs]
            # poloclub also normalises "1. 23" → "1.23" in their notebook
            import re
            cell_contents = [re.sub(r'(\d)\.\s+(\d)', r'\1.\2', s) for s in cell_contents]

        # ── assemble ─────────────────────────────────────────────────────
        html = _build_html(struct_tokens, cell_contents)
        return {
            "html": html,
            "metadata": {
                "n_struct_tokens": len(struct_tokens),
                "n_bboxes":        len(bboxes),
                "n_cells_filled":  sum(1 for c in cell_contents if c.strip()),
                "image_size":      list(image.size),
                "mode":            "full_three_pass",
            },
        }

    # ── helpers ──────────────────────────────────────────────────────────

    def _parse_bboxes(self, bbox_str: str, image_size: Tuple[int, int]
                       ) -> List[Tuple[int, int, int, int]]:
        """Convert decoded bbox token stream to pixel coordinates.
        bbox tokens are quantised on a 448-grid (the inference image size)."""
        import re
        bbox_str = bbox_str.split("<eos>")[0]
        nums = [int(m.group(1)) for m in re.finditer(r"bbox-(\d+)", bbox_str)]
        if len(nums) < 4:
            return []
        W, H = image_size
        # Rescale from 448 grid to original image dimensions
        sx = W / 448.0
        sy = H / 448.0
        out = []
        for i in range(0, len(nums) - 3, 4):
            x1, y1, x2, y2 = nums[i:i + 4]
            if x2 <= x1 or y2 <= y1:
                continue
            out.append((int(x1 * sx), int(y1 * sy),
                         int(x2 * sx), int(y2 * sy)))
        return out


__all__ = ["UniTableRunner"]
