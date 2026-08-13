#!/usr/bin/env python3
"""
debug_build_table_alignment.py - Verify build_table_robust Alignment Issues

Features:
- Select 20 random samples from test split
- Run full inference (Structure + BBox + Content) for each sample
- Output 4 files per sample:
  - {sample}_gt.html — Ground Truth HTML
  - {sample}_pred_fine_tuned.html — Fine-tuned model prediction
  - {sample}_pred_original.html — Original model prediction
  - {sample}_summary.txt — Brief comparison summary

Usage:
```bash
# Run in Colab
python debug_build_table_alignment.py

# Run locally
python experiments/v2/debug_build_table_alignment.py
```
"""

# ──────────────────────────────────────────────────────────────────────
# Path Configuration
# ──────────────────────────────────────────────────────────────────────
import os
import sys
import json
import re
import time
import random
from pathlib import Path
from functools import partial

# ── Detect Colab Environment ──────────────────────────────────────────
IS_COLAB = os.path.exists('/content/drive') or 'google.colab' in sys.modules

if IS_COLAB:
    BASE_COLAB = Path('/content/drive/MyDrive/Computer_Vision_Team8')
    UNITABLE_DIR = Path('/content/unitable')
    CSV_PATH = BASE_COLAB / 'data' / 'dataset_splits_ids_1300sample.csv'
    IMAGES_DIR = Path('/content/data/data/processed/images')
    MODEL_DIR = UNITABLE_DIR / "experiments" / "unitable_weights"
    CHECKPOINT_DIR = BASE_COLAB / 'experiments' / 'v2' / 'checkpoints'
    OUTPUT_DIR = BASE_COLAB / 'experiments' / 'v2' / 'outputs' / 'debug_samples'
else:
    BASE_DIR = Path(__file__).parent.parent.parent  # CV_Test/
    UNITABLE_DIR = BASE_DIR / "unitable"
    DATA_DIR = BASE_DIR / "data"
    CSV_PATH = DATA_DIR / "dataset_splits_ids_1300sample.csv"
    IMAGES_DIR = DATA_DIR / "data" / "processed" / "images"
    MODEL_DIR = UNITABLE_DIR / "experiments" / "unitable_weights"
    CHECKPOINT_DIR = BASE_DIR / "experiments" / 'v2' / 'checkpoints'
    OUTPUT_DIR = BASE_DIR / "experiments" / "v2" / "outputs" / 'debug_samples'

sys.path.insert(0, str(UNITABLE_DIR))

# ──────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import tokenizers as tk
from PIL import Image
from torchvision import transforms
import numpy as np

from src.model import EncoderDecoder, ImgLinearBackbone, Encoder, Decoder
from src.utils import (
    subsequent_mask,
    pred_token_within_range,
    greedy_sampling,
    bbox_str_to_token_list,
    cell_str_to_token_list,
    html_str_to_token_list,
)
from src.vocab.constant import HTML_TOKENS, TASK_TOKENS, RESERVED_TOKENS, BBOX_TOKENS

# ── Token lists ──────────────────────────────────────────────────────
VALID_HTML_TOKEN = ["<eos>"] + HTML_TOKENS
INVALID_CELL_TOKEN = ["<sos>", "<pad>", "<empty>", "<sep>"] + TASK_TOKENS + RESERVED_TOKENS
VALID_BBOX_TOKEN = ["<eos>"] + BBOX_TOKENS[:449]

# ── Image normalization parameters ───────────────────────────────────
MEAN = [0.86597056, 0.88463296, 0.87491087]
STD = [0.20686570, 0.18201602, 0.18485524]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ── Configuration ────────────────────────────────────────────────────
NUM_SAMPLES = 20           # Number of samples to output
SEED = 42                  # Random seed (for reproducibility)
MAX_SEQ_LEN = 200          # Content model maximum sequence length


# ──────────────────────────────────────────────────────────────────────
# Utility Functions
# ──────────────────────────────────────────────────────────────────────
def preprocess_for_teds(pred_html: str, gt_html: str) -> tuple:
    """Preprocessing for TEDS evaluation"""
    has_structure_tags = ('<thead>' in gt_html or '<tbody>' in gt_html or '<tfoot>' in gt_html)
    if not has_structure_tags:
        for tag in ['<thead>', '</thead>', '<tbody>', '</tbody>', '<tfoot>', '</tfoot>']:
            pred_html = pred_html.replace(tag, '')
    pred_html = re.sub(r'(\d)\.\s+(\d)', r'\1.\2', pred_html)
    return pred_html, gt_html


def load_model(weight_path: str) -> EncoderDecoder:
    """Load model by automatically inferring hyperparameters from state_dict"""
    state = torch.load(weight_path, map_location=device, weights_only=False)

    d_model = state['token_embed.embedding.weight'].shape[1]
    vocab_size = state['generator.weight'].shape[0]
    max_seq_len = state['pos_embed.embedding.weight'].shape[0]
    padding_idx = 2

    enc_keys = [k for k in state if k.startswith('encoder.encoder.layers.')]
    dec_keys = [k for k in state if k.startswith('decoder.decoder.layers.')]
    nlayer_enc = max(int(k.split('.')[3]) for k in enc_keys) + 1 if enc_keys else 2
    nlayer_dec = max(int(k.split('.')[3]) for k in dec_keys) + 1 if dec_keys else 4

    ff_dim = state['encoder.encoder.layers.0.linear1.weight'].shape[0]
    ff_ratio = ff_dim // d_model
    nhead = d_model // 64

    backbone = ImgLinearBackbone(d_model=d_model, patch_size=16)
    encoder = Encoder(d_model=d_model, nhead=nhead, dropout=0.2,
                      activation="gelu", norm_first=True,
                      nlayer=nlayer_enc, ff_ratio=ff_ratio)
    decoder = Decoder(d_model=d_model, nhead=nhead, dropout=0.2,
                      activation="gelu", norm_first=True,
                      nlayer=nlayer_dec, ff_ratio=ff_ratio)
    model = EncoderDecoder(
        backbone=backbone, encoder=encoder, decoder=decoder,
        vocab_size=vocab_size, d_model=d_model,
        padding_idx=padding_idx, max_seq_len=max_seq_len,
        dropout=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def image_to_tensor(image: Image.Image, size: tuple) -> torch.Tensor:
    """Convert image to tensor and move to GPU"""
    T = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    return T(image.convert("RGB")).unsqueeze(0).to(device)


def autoregressive_decode(
    model, image, prefix, max_decode_len, eos_id,
    token_whitelist=None, token_blacklist=None,
):
    """Autoregressive decoding"""
    model.eval()
    with torch.no_grad():
        memory = model.encode(image)
        context = (
            torch.tensor(prefix, dtype=torch.int32)
            .repeat(image.shape[0], 1)
            .to(device)
        )

    for step_idx in range(max_decode_len):
        if all(eos_id in k for k in context):
            break
        with torch.no_grad():
            causal_mask = subsequent_mask(context.shape[1]).to(device)
            logits = model.decode(memory, context, tgt_mask=causal_mask, tgt_padding_mask=None)
            logits = model.generator(logits)[:, -1, :]

        logits = pred_token_within_range(
            logits.detach(),
            white_list=token_whitelist,
            black_list=token_blacklist,
        )
        next_probs, next_tokens = greedy_sampling(logits)
        context = torch.cat([context, next_tokens], dim=1)

    return context


def build_table_robust(html_tokens: list, bboxes: list, cells: list) -> str:
    """Use spatial heuristic alignment to cluster BBoxes into visual rows based on y1 coordinates"""
    items = []
    for bbox, text in zip(bboxes, cells):
        items.append({"bbox": bbox, "text": text, "y1": bbox[1], "x1": bbox[0]})

    if items:
        items.sort(key=lambda x: x["y1"])

        rows_of_bboxes = []
        current_row = [items[0]]

        heights = [item["bbox"][3] - item["bbox"][1] for item in items]
        median_h = sorted(heights)[len(heights)//2] if heights else 10
        threshold_y = median_h * 0.6

        for item in items[1:]:
            avg_y1 = sum(x["y1"] for x in current_row) / len(current_row)
            if abs(item["y1"] - avg_y1) < threshold_y:
                current_row.append(item)
            else:
                current_row.sort(key=lambda x: x["x1"])
                rows_of_bboxes.append(current_row)
                current_row = [item]

        current_row.sort(key=lambda x: x["x1"])
        rows_of_bboxes.append(current_row)
    else:
        rows_of_bboxes = []

    final_html_table = ""
    v_row_idx = 0
    v_col_idx = 0

    for tag in html_tokens:
        if tag == "<tr>":
            final_html_table += tag
            v_col_idx = 0
        elif tag == "</tr>":
            final_html_table += tag
            v_row_idx += 1
        elif tag in ("<td>[]</td>", ">[]</td>"):
            cell_text = ""
            if v_row_idx < len(rows_of_bboxes):
                current_vis_row = rows_of_bboxes[v_row_idx]
                if v_col_idx < len(current_vis_row):
                    cell_text = current_vis_row[v_col_idx]["text"]
                    v_col_idx += 1

            if tag.startswith(">"):
                final_html_table += ">" + cell_text + "</td>"
            else:
                final_html_table += "<td>" + cell_text + "</td>"
        else:
            final_html_table += tag

    return final_html_table


def load_training_data_from_csv(
    csv_path: Path,
    images_dir: Path,
    max_samples: int = -1,
    split: str = "test"
) -> list:
    """Load data from CSV for specified split"""
    import pandas as pd

    df = pd.read_csv(csv_path)

    if split and 'split' in df.columns:
        df = df[df['split'] == split]

    results = []
    for _, row in df.iterrows():
        if max_samples > 0 and len(results) >= max_samples:
            break

        fname = str(row.get('img_id', ''))
        gt_html = str(row.get('html', ''))

        if not fname or not gt_html:
            continue

        img_path = None
        for ext in [".png", ".jpg", ".jpeg", ""]:
            candidate = images_dir / f"{fname}{ext}"
            if candidate.exists():
                img_path = str(candidate)
                break

        if img_path is None:
            continue

        results.append({
            'filename': fname,
            'gt_html': gt_html,
            'image_path': img_path,
        })

    return results


def extract_cell_texts_from_html(html: str) -> list:
    """Extract cell texts from HTML"""
    import html as html_mod
    html_str = html_mod.unescape(html)
    matches = re.findall(r'<td[^>]*>(.*?)</td>', html_str, re.DOTALL)
    texts = []
    for m in matches:
        text = re.sub(r'<[^>]+>', '', m)
        text = re.sub(r'\s+', ' ', text).strip()
        if text:
            texts.append(text)
    return texts


def format_html_for_display(html: str, max_chars: int = 500) -> str:
    """Format HTML for display"""
    # Simplified display
    html = html.strip()
    if len(html) > max_chars:
        return html[:max_chars] + f"\n... ({len(html)} chars total)"
    return html


# ──────────────────────────────────────────────────────────────────────
# Main Logic
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print("Debug build_table_robust Alignment")
    print("=" * 60)

    # ── Create Output Directory ────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {OUTPUT_DIR}")

    # ── Load Vocab ─────────────────────────────────────────────────
    print("\n[Step 1] Loading Vocab...")
    vocab_s = tk.Tokenizer.from_file(str(UNITABLE_DIR / "vocab" / "vocab_html.json"))
    vocab_b = tk.Tokenizer.from_file(str(UNITABLE_DIR / "vocab" / "vocab_bbox.json"))
    vocab_c = tk.Tokenizer.from_file(str(UNITABLE_DIR / "vocab" / "vocab_cell_6k.json"))
    print(f"  vocab_s size: {vocab_s.get_vocab_size()}")
    print(f"  vocab_b size: {vocab_b.get_vocab_size()}")
    print(f"  vocab_c size: {vocab_c.get_vocab_size()}")

    # ── Load Models ────────────────────────────────────────────────
    print("\n[Step 2] Loading models...")

    print("  Loading Structure model...")
    structure_model = load_model(str(MODEL_DIR / "unitable_large_structure.pt"))

    print("  Loading BBox model...")
    bbox_model = load_model(str(MODEL_DIR / "unitable_large_bbox.pt"))

    print("  Loading original Content model...")
    original_content_model = load_model(str(MODEL_DIR / "unitable_large_content.pt"))

    # Load fine-tuned Content model
    fine_tuned_checkpoint = CHECKPOINT_DIR / "content_best_merged.pt"
    if fine_tuned_checkpoint.exists():
        print(f"  Loading fine-tuned Content model: {fine_tuned_checkpoint}")
        fine_tuned_content_model = load_model(str(fine_tuned_checkpoint))
    else:
        print(f"  [WARN] Fine-tuned checkpoint does not exist: {fine_tuned_checkpoint}")
        print("  Using original model as substitute")
        fine_tuned_content_model = original_content_model

    print("  ✅ All models loaded successfully")

    # ── Load Test Data ─────────────────────────────────────────────
    print(f"\n[Step 3] Loading test data...")
    all_samples = load_training_data_from_csv(CSV_PATH, IMAGES_DIR, max_samples=-1, split="test")
    print(f"  Test split has {len(all_samples)} samples in total")

    # Randomly select 20 samples
    if len(all_samples) > NUM_SAMPLES:
        samples = random.sample(all_samples, NUM_SAMPLES)
    else:
        samples = all_samples
    print(f"  Randomly selected {len(samples)} samples for analysis")

    # ── Run Inference for Each Sample ──────────────────────────────
    print(f"\n[Step 4] Starting inference...")
    print(f"{'='*60}")

    for i, sample in enumerate(samples):
        fname = sample['filename']
        gt_html = sample['gt_html']
        img_path = sample['image_path']

        print(f"\n[{i+1}/{len(samples)}] {fname}")
        print(f"  GT cells: {len(extract_cell_texts_from_html(gt_html))}")

        # ── Load Image ─────────────────────────────────────────────
        image = Image.open(img_path).convert("RGB")
        img_tensor = image_to_tensor(image, size=(448, 448))

        # ── Pass 1: Structure ──────────────────────────────────────
        pred_html_ctx = autoregressive_decode(
            model=structure_model, image=img_tensor,
            prefix=[vocab_s.token_to_id("[html]")],
            max_decode_len=512,
            eos_id=vocab_s.token_to_id("<eos>"),
            token_whitelist=[vocab_s.token_to_id(t) for t in VALID_HTML_TOKEN],
        )
        pred_html_str = vocab_s.decode(pred_html_ctx[0].tolist(), skip_special_tokens=False)
        pred_html_tokens = html_str_to_token_list(pred_html_str)

        # ── Pass 2: BBox ───────────────────────────────────────────
        pred_bbox_ctx = autoregressive_decode(
            model=bbox_model, image=img_tensor,
            prefix=[vocab_b.token_to_id("[bbox]")],
            max_decode_len=1024,
            eos_id=vocab_b.token_to_id("<eos>"),
            token_whitelist=[vocab_b.token_to_id(t) for t in VALID_BBOX_TOKEN],
        )
        pred_bbox_str = vocab_b.decode(pred_bbox_ctx[0].tolist(), skip_special_tokens=False)
        unnorm_bboxes = bbox_str_to_token_list(pred_bbox_str)

        # ── Pass 3a: Content (Original) ────────────────────────────
        if len(unnorm_bboxes) == 0:
            pred_cell_original = []
        else:
            w, h = image.size
            cell_imgs = []
            for bbox in unnorm_bboxes:
                x1, y1, x2, y2 = bbox
                x1_px = int(x1 / 448 * w)
                y1_px = int(y1 / 448 * h)
                x2_px = int(x2 / 448 * w)
                y2_px = int(y2 / 448 * h)
                x1_px, y1_px = max(0, x1_px), max(0, y1_px)
                x2_px, y2_px = min(w, x2_px), min(h, y2_px)

                if x2_px <= x1_px or y2_px <= y1_px:
                    cell_crop = image
                else:
                    cell_crop = image.crop((x1_px, y1_px, x2_px, y2_px))
                cell_imgs.append(image_to_tensor(cell_crop, size=(112, 448)))

            cell_tensor = torch.cat(cell_imgs, dim=0)

            pred_cell_ctx = autoregressive_decode(
                model=original_content_model, image=cell_tensor,
                prefix=[vocab_c.token_to_id("[cell]")],
                max_decode_len=200,
                eos_id=vocab_c.token_to_id("<eos>"),
                token_blacklist=[vocab_c.token_to_id(t) for t in INVALID_CELL_TOKEN],
            )
            pred_cell_strs = vocab_c.decode_batch(pred_cell_ctx.tolist(), skip_special_tokens=False)
            pred_cell_original = [cell_str_to_token_list(c) for c in pred_cell_strs]

        # ── Pass 3b: Content (Fine-tuned) ──────────────────────────
        if len(unnorm_bboxes) == 0:
            pred_cell_fine_tuned = []
        else:
            cell_tensor = torch.cat(cell_imgs, dim=0)  # reuse from above

            pred_cell_ctx_ft = autoregressive_decode(
                model=fine_tuned_content_model, image=cell_tensor,
                prefix=[vocab_c.token_to_id("[cell]")],
                max_decode_len=200,
                eos_id=vocab_c.token_to_id("<eos>"),
                token_blacklist=[vocab_c.token_to_id(t) for t in INVALID_CELL_TOKEN],
            )
            pred_cell_strs_ft = vocab_c.decode_batch(pred_cell_ctx_ft.tolist(), skip_special_tokens=False)
            pred_cell_fine_tuned = [cell_str_to_token_list(c) for c in pred_cell_strs_ft]

        # ── Synthesize HTML ────────────────────────────────────────
        final_html_original = build_table_robust(pred_html_tokens, unnorm_bboxes, pred_cell_original)
        final_html_fine_tuned = build_table_robust(pred_html_tokens, unnorm_bboxes, pred_cell_fine_tuned)

        gt_wrapped = f"<html><body><table>{gt_html}</table></body></html>"
        pred_wrapped_original = f"<html><body><table>{final_html_original}</table></body></html>"
        pred_wrapped_fine_tuned = f"<html><body><table>{final_html_fine_tuned}</table></body></html>"

        # ── Save HTML Files ────────────────────────────────────────
        safe_fname = fname.replace('.', '_').replace('/', '_')

        # GT
        gt_path = OUTPUT_DIR / f"{safe_fname}_gt.html"
        with open(gt_path, "w", encoding="utf-8") as f:
            f.write(f"<!-- Ground Truth for {fname} -->\n")
            f.write(gt_wrapped)

        # Original
        orig_path = OUTPUT_DIR / f"{safe_fname}_pred_original.html"
        with open(orig_path, "w", encoding="utf-8") as f:
            f.write(f"<!-- Original Model Prediction for {fname} -->\n")
            f.write(pred_wrapped_original)

        # Fine-tuned
        ft_path = OUTPUT_DIR / f"{safe_fname}_pred_fine_tuned.html"
        with open(ft_path, "w", encoding="utf-8") as f:
            f.write(f"<!-- Fine-tuned Model Prediction for {fname} -->\n")
            f.write(pred_wrapped_fine_tuned)

        # ── Save Summary ───────────────────────────────────────────
        gt_cells = extract_cell_texts_from_html(gt_html)
        orig_cells = extract_cell_texts_from_html(final_html_original)
        ft_cells = extract_cell_texts_from_html(final_html_fine_tuned)

        summary_path = OUTPUT_DIR / f"{safe_fname}_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Debug Summary for {fname}\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"GT cells: {len(gt_cells)}\n")
            f.write(f"Original cells: {len(orig_cells)}\n")
            f.write(f"Fine-tuned cells: {len(ft_cells)}\n\n")

            f.write(f"BBox count: {len(unnorm_bboxes)}\n\n")

            f.write("Ground Truth cells:\n")
            for j, c in enumerate(gt_cells):
                f.write(f"  [{j}] {c}\n")
            f.write("\n")

            f.write("Original Model cells:\n")
            for j, c in enumerate(orig_cells):
                marker = " ← DIFF" if j < len(gt_cells) and c != gt_cells[j] else ""
                f.write(f"  [{j}] {c}{marker}\n")
            f.write("\n")

            f.write("Fine-tuned Model cells:\n")
            for j, c in enumerate(ft_cells):
                marker = " ← DIFF" if j < len(gt_cells) and c != gt_cells[j] else ""
                f.write(f"  [{j}] {c}{marker}\n")
            f.write("\n")

            f.write("HTML Preview (truncated):\n")
            f.write(f"  GT: {format_html_for_display(gt_html, 300)}\n\n")
            f.write(f"  Original: {format_html_for_display(final_html_original, 300)}\n\n")
            f.write(f"  Fine-tuned: {format_html_for_display(final_html_fine_tuned, 300)}\n")

        print(f"  GT cells: {len(gt_cells)}, Orig: {len(orig_cells)}, FT: {len(ft_cells)}")
        print(f"  Saved to: {OUTPUT_DIR}/{safe_fname}_*")

    # ── Completion ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("[DONE] All samples processed successfully!")
    print(f"{'='*60}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nFile list:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {f.name}")