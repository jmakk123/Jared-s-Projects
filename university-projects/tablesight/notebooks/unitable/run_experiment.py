#!/usr/bin/env python3
"""
run_experiment.py - UniTable Content Model Fine-tuning Experiment (Simplified Version)

Version: v2.1
Date: 2026-05-18

Features:
- Encapsulates the training logic of fine_tune_v2.py
- Automatically merges LoRA weights
- Automatically evaluates (original vs. fine-tuned)
- Outputs three metrics: TEDS (Content), TEDS-S (Struct), Text-only Word F1
- Supports skipping training to directly evaluate (saves resources)

Usage:
```bash
# Full pipeline: training + merge + evaluation
python run_experiment.py

# Evaluation only (using existing checkpoint)
# Set SKIP_TRAINING = True and specify LOAD_CHECKPOINT_PATH
python run_experiment.py
```

**Team members only need to modify the numbers in the configuration section, no need to understand the code logic!**
"""

# ═══════════════════════════════════════════════════════════
# [Configuration Section] Modify these parameters to adjust experiment settings
# ═══════════════════════════════════════════════════════════

# ── Training Settings ───────────────────────────────────────────────
TRAIN_EPOCHS = 3           # Number of training epochs
LEARNING_RATE = 5e-5       # Learning rate
BATCH_SIZE = 256            # Batch size
GRADIENT_CLIP = 1.0        # Gradient clipping
MAX_SEQ_LEN = 200          # Maximum sequence length

# ── LoRA Settings ─────────────────────────────────────────────────
USE_LORA = True            # Whether to use LoRA
LORA_RANK = 8              # LoRA rank
LORA_ALPHA = 16.0          # LoRA alpha (scaling = alpha/rank)
LORA_DROPOUT = 0.1         # LoRA dropout

# ── Data Settings ──────────────────────────────────────────────────
TRAIN_MAX_SAMPLES = 850    # Number of training samples (-1 = all 850)
VAL_MAX_SAMPLES = 150      # Number of validation samples (-1 = all 150)

# ── Evaluation Settings ────────────────────────────────────────────
SKIP_TRAINING = False              # True = Skip training, directly load checkpoint for evaluation
LOAD_CHECKPOINT_PATH = None       # Specify checkpoint path (None = use default path experiments/v2/checkpoints/content_best.pt)
COMPARE_WITH_ORIGINAL = True      # True = Compare with original model results
ORIGINAL_TEST_SAMPLES = 10        # Original model test samples (-1 = all)
FINETUNED_TEST_SAMPLES = 10       # Fine-tuned model test samples (-1 = all)

# ── Path Settings ──────────────────────────────────────────────────
USE_COLAB = True           # True = Colab mode, False = Local mode

# ═══════════════════════════════════════════════════════════
# [Do not modify below] Unless you know what you are doing
# ═══════════════════════════════════════════════════════════

import os
import sys
import json
import re
import time
from pathlib import Path

# ── Detect Colab Environment ────────────────────────────────────────────
IS_COLAB = os.path.exists('/content/drive') or 'google.colab' in sys.modules

if USE_COLAB or IS_COLAB:
    BASE_COLAB = Path('/content/drive/MyDrive/Computer_Vision_Team8')
    UNITABLE_DIR = Path('/content/unitable')
    CSV_PATH = BASE_COLAB / 'data' / 'dataset_splits_ids_1300sample.csv'
    IMAGES_DIR = Path('/content/data/data/processed/images')
    MODEL_DIR = UNITABLE_DIR / "experiments" / "unitable_weights"
    CHECKPOINT_DIR = BASE_COLAB / 'experiments' / 'v2' / 'checkpoints'
    OUTPUT_DIR = BASE_COLAB / 'experiments' / 'v2' / 'outputs'
else:
    BASE_DIR = Path(__file__).parent.parent.parent  # CV_Test/
    UNITABLE_DIR = BASE_DIR / "unitable"
    DATA_DIR = BASE_DIR / "data"
    CSV_PATH = DATA_DIR / "dataset_splits_ids_1300sample.csv"
    IMAGES_DIR = DATA_DIR / "data" / "processed" / "images"
    MODEL_DIR = UNITABLE_DIR / "experiments" / "unitable_weights"
    CHECKPOINT_DIR = BASE_DIR / "experiments" / 'v2' / 'checkpoints'
    OUTPUT_DIR = BASE_DIR / "experiments" / "v2" / "outputs"

sys.path.insert(0, str(UNITABLE_DIR))

# ── Imports ──────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tokenizers as tk
from torch.utils.data import Dataset, DataLoader
from functools import partial

try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("[WARN] peft not installed, using standard fine-tuning (no LoRA)")

from PIL import Image
from torchvision import transforms
import numpy as np
from tqdm import tqdm

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
from src.utils.teds import TEDS

# ── Token lists ──────────────────────────────────────────────────
VALID_HTML_TOKEN = ["<eos>"] + HTML_TOKENS
INVALID_CELL_TOKEN = ["<sos>", "<pad>", "<empty>", "<sep>"] + TASK_TOKENS + RESERVED_TOKENS
VALID_BBOX_TOKEN = ["<eos>"] + BBOX_TOKENS[:449]

# ── Image normalization parameters ───────────────────────────────
MEAN = [0.86597056, 0.88463296, 0.87491087]
STD = [0.20686570, 0.18201602, 0.18485524]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")


# ═══════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════
def remove_html_tags(text: str) -> str:
    """Remove all HTML tags, keeping only plain text content"""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compute_text_only_metrics(results: dict) -> tuple:
    """Compute Text-only Word-level F1"""
    per_sample = {}
    f1_scores = []

    for fname, info in results.items():
        pred_text = remove_html_tags(info["pred"])
        gt_text = remove_html_tags(info["gt"])

        pred_words = set(pred_text.lower().split())
        gt_words = set(gt_text.lower().split())

        if len(pred_words) == 0 and len(gt_words) == 0:
            f1 = 1.0
        elif len(pred_words) == 0 or len(gt_words) == 0:
            f1 = 0.0
        else:
            precision = len(pred_words & gt_words) / len(pred_words)
            recall = len(pred_words & gt_words) / len(gt_words)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        f1_scores.append(f1)
        per_sample[fname] = f1

    mean_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    std_f1 = float(np.std(f1_scores)) if f1_scores else 0.0

    return mean_f1, std_f1, per_sample


def preprocess_for_teds(pred_html: str, gt_html: str) -> tuple:
    """Preprocessing for TEDS evaluation"""
    has_structure_tags = ('<thead>' in gt_html or '<tbody>' in gt_html or '<tfoot>' in gt_html)
    if not has_structure_tags:
        for tag in ['<thead>', '</thead>', '<tbody>', '</tbody>', '<tfoot>', '</tfoot>']:
            pred_html = pred_html.replace(tag, '')
    pred_html = re.sub(r'(\d)\.\s+(\d)', r'\1.\2', pred_html)
    return pred_html, gt_html


def load_model(weight_path: str) -> EncoderDecoder:
    """Load model"""
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

    print(f"  arch: d_model={d_model}, nhead={nhead}, "
          f"enc_layers={nlayer_enc}, dec_layers={nlayer_dec}, "
          f"ff_ratio={ff_ratio}, vocab={vocab_size}, max_seq={max_seq_len}")

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
        dropout=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def image_to_tensor(image: Image.Image, size: tuple) -> torch.Tensor:
    T = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    return T(image.convert("RGB")).unsqueeze(0).to(device)


def autoregressive_decode(model, image, prefix, max_decode_len, eos_id,
                          token_whitelist=None, token_blacklist=None):
    model.eval()
    with torch.no_grad():
        memory = model.encode(image)
        context = (torch.tensor(prefix, dtype=torch.int32)
                   .repeat(image.shape[0], 1).to(device))

    for step_idx in range(max_decode_len):
        if all(eos_id in k for k in context):
            break
        with torch.no_grad():
            causal_mask = subsequent_mask(context.shape[1]).to(device)
            logits = model.decode(memory, context, tgt_mask=causal_mask, tgt_padding_mask=None)
            logits = model.generator(logits)[:, -1, :]

        logits = pred_token_within_range(logits.detach(),
                                         white_list=token_whitelist,
                                         black_list=token_blacklist)
        next_probs, next_tokens = greedy_sampling(logits)
        context = torch.cat([context, next_tokens], dim=1)

    return context


def build_table_robust(html_tokens: list, bboxes: list, cells: list) -> str:
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


def load_training_data_from_csv(csv_path: Path, images_dir: Path,
                                 max_samples: int = -1,
                                 phase: str = "phase1",
                                 split: str = "train") -> list:
    """Load data from CSV for training"""
    import pandas as pd

    df = pd.read_csv(csv_path)
    print(f"[INFO] CSV has {len(df)} samples in total")

    if phase:
        df = df[df['phase'] == phase]
        print(f"[INFO] {phase} phase has {len(df)} samples")
    if split and 'split' in df.columns:
        df = df[df['split'] == split]
        print(f"[INFO] {split} split has {len(df)} samples")

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

        # Extract cell texts
        import html as html_mod
        html_str = html_mod.unescape(gt_html)
        matches = re.findall(r'<td[^>]*>(.*?)</td>', html_str, re.DOTALL)
        gt_cells = []
        for m in matches:
            text = re.sub(r'<[^>]+>', '', m)
            text = re.sub(r'\s+', ' ', text).strip()
            if text and isinstance(text, str):
                gt_cells.append(text)

        results.append({
            'filename': fname,
            'gt_html': gt_html,
            'gt_cells': gt_cells,
            'image_path': img_path,
        })

    print(f"[INFO] Successfully loaded {len(results)} samples")
    return results


def load_bbox_checkpoint(split_name: str) -> dict:
    """Load BBox predictions from checkpoint"""
    candidates = []
    if IS_COLAB:
        candidates.append(Path('/content/drive/MyDrive/Computer_Vision_Team8/models/ryan/outputs/checkpoints') / f"bbox_preds_{split_name}.pt")
    else:
        base_dir = Path(__file__).parent.parent.parent
        candidates.append(base_dir / "experiments" / "v1-ft-content" / "outputs" / "checkpoints" / f"bbox_preds_{split_name}.pt")

    for cp in candidates:
        if cp.exists():
            bbox_preds = torch.load(cp, map_location=device, weights_only=False)
            n_samples = len(bbox_preds)
            total_cells = sum(len(v) for v in bbox_preds.values())
            print(f"[CHECKPOINT] Loaded BBox {split_name}: {n_samples} samples, {total_cells} cells")
            return bbox_preds
    return None


# ── Content Fine-tune Dataset ───────────────────────────────────
class ContentFineTuneDataset(Dataset):
    def __init__(self, samples, vocab_c, bbox_predictions=None):
        self.samples = samples
        self.vocab_c = vocab_c
        self.bbox_predictions = bbox_predictions or {}

    def __len__(self):
        return sum(len(s['gt_cells']) for s in self.samples)

    def __getitem__(self, idx):
        cumulative = 0
        for sample in self.samples:
            n_cells = len(sample['gt_cells'])
            if cumulative + n_cells > idx:
                cell_idx = idx - cumulative
                cell_text = sample['gt_cells'][cell_idx]
                img_path = sample['image_path']
                break
            cumulative += n_cells

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            return torch.zeros((3, 112, 448)), torch.tensor([0], dtype=torch.long)

        bboxes = self.bbox_predictions.get(sample['filename'], [])

        if cell_idx < len(bboxes) and bboxes:
            x1, y1, x2, y2 = bboxes[cell_idx]
            w, h = img.size
            x1_px = int(x1 / 448 * w)
            y1_px = int(y1 / 448 * h)
            x2_px = int(x2 / 448 * w)
            y2_px = int(y2 / 448 * h)
            x1_px, y1_px = max(0, x1_px), max(0, y1_px)
            x2_px, y2_px = min(w, x2_px), min(h, y2_px)
            if x2_px <= x1_px or y2_px <= y1_px:
                cell_crop = img
            else:
                cell_crop = img.crop((x1_px, y1_px, x2_px, y2_px))
        else:
            cell_crop = img

        cell_tensor = image_to_tensor(cell_crop, size=(112, 448)).squeeze(0)

        if cell_text is None:
            cell_text = ""
        elif isinstance(cell_text, float):
            import math
            cell_text = "" if (math.isnan(cell_text) or math.isinf(cell_text)) else str(cell_text)
        elif not isinstance(cell_text, str):
            cell_text = str(cell_text)

        cell_tokens = cell_text.split() if cell_text.strip() else []
        target_tokens_str = "[cell] " + " ".join(cell_tokens) + " <eos>"

        if hasattr(self.vocab_c, 'encode'):
            try:
                target_ids = self.vocab_c.encode(target_tokens_str).ids
            except (TypeError, ValueError):
                vocab = self.vocab_c.get_vocab()
                target_ids = [vocab.get(t, 0) for t in target_tokens_str.split()]
        else:
            vocab = self.vocab_c.get_vocab()
            target_ids = [vocab.get(t, 0) for t in target_tokens_str.split()]

        if len(target_ids) > MAX_SEQ_LEN:
            target_ids = target_ids[:MAX_SEQ_LEN]

        return cell_tensor, torch.tensor(target_ids, dtype=torch.long)


def collate_fn(batch):
    cell_tensors = [item[0].cpu() if isinstance(item[0], torch.Tensor) else item[0] for item in batch]
    target_tensors = [item[1].cpu() if isinstance(item[1], torch.Tensor) else item[1] for item in batch]

    cell_images = torch.stack(cell_tensors, dim=0)
    max_len = max(t.size(0) for t in target_tensors)
    batch_size = len(target_tensors)
    padded_targets = torch.zeros(batch_size, max_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

    for i, t in enumerate(target_tensors):
        padded_targets[i, :len(t)] = t
        attention_mask[i, :len(t)] = 1

    return cell_images, padded_targets, attention_mask


# ═══════════════════════════════════════════════════════════
# Evaluation Function (must be defined before main)
# ═══════════════════════════════════════════════════════════
def evaluate_model(content_model, structure_model, bbox_model,
                   vocab_s, vocab_b, vocab_c, samples, model_name):
    """Evaluate model and return results (includes three metrics)"""
    results = {}
    inference_times = []

    for i, sample in enumerate(samples):
        fname = sample['filename']
        gt_html = sample['gt_html']
        img_path = sample['image_path']

        start_time = time.time()

        try:
            image = Image.open(img_path).convert("RGB")
            img_tensor = image_to_tensor(image, size=(448, 448))

            # Pass 1: Structure
            pred_html_ctx = autoregressive_decode(
                model=structure_model, image=img_tensor,
                prefix=[vocab_s.token_to_id("[html]")],
                max_decode_len=512,
                eos_id=vocab_s.token_to_id("<eos>"),
                token_whitelist=[vocab_s.token_to_id(t) for t in VALID_HTML_TOKEN],
            )
            pred_html_str = vocab_s.decode(pred_html_ctx[0].tolist(), skip_special_tokens=False)
            pred_html_tokens = html_str_to_token_list(pred_html_str)

            # Pass 2: BBox
            pred_bbox_ctx = autoregressive_decode(
                model=bbox_model, image=img_tensor,
                prefix=[vocab_b.token_to_id("[bbox]")],
                max_decode_len=1024,
                eos_id=vocab_b.token_to_id("<eos>"),
                token_whitelist=[vocab_b.token_to_id(t) for t in VALID_BBOX_TOKEN],
            )
            pred_bbox_str = vocab_b.decode(pred_bbox_ctx[0].tolist(), skip_special_tokens=False)
            unnorm_bboxes = bbox_str_to_token_list(pred_bbox_str)

            # Pass 3: Content
            if len(unnorm_bboxes) == 0:
                pred_cell = []
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
                    model=content_model, image=cell_tensor,
                    prefix=[vocab_c.token_to_id("[cell]")],
                    max_decode_len=200,
                    eos_id=vocab_c.token_to_id("<eos>"),
                    token_blacklist=[vocab_c.token_to_id(t) for t in INVALID_CELL_TOKEN],
                )
                pred_cell_strs = vocab_c.decode_batch(pred_cell_ctx.tolist(), skip_special_tokens=False)
                pred_cell = [cell_str_to_token_list(c) for c in pred_cell_strs]

            final_html_table = build_table_robust(pred_html_tokens, unnorm_bboxes, pred_cell)
            final_html_table_eval = f"<table>{final_html_table}</table>"

            pred_wrapped = f"<html><body>{final_html_table_eval}</body></html>"
            gt_wrapped = f"<html><body>{gt_html}</body></html>"

            pred_clean, gt_clean = preprocess_for_teds(pred_wrapped, gt_wrapped)
            results[fname] = {"pred": pred_clean, "gt": gt_clean}

        except Exception as e:
            print(f"  [WARN] Inference failed for {fname}: {e}")
            continue

        elapsed = time.time() - start_time
        inference_times.append(elapsed)

        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            print(f"  [{i+1}/{len(samples)}] Completed {fname} ({model_name})")

    # Compute TEDS (Content) - includes structure and content
    teds_full = TEDS(structure_only=False, n_jobs=1)
    scores_full = teds_full.batch_evaluate(results) if results else {}
    full_scores = [info["scores"] for info in scores_full.values()]
    mean_teds_full = float(np.mean(full_scores)) if full_scores else 0.0

    # Compute TEDS-S (Structure-only)
    teds_struct = TEDS(structure_only=True, n_jobs=1)
    scores_struct = teds_struct.batch_evaluate(results) if results else {}
    struct_scores = [info["scores"] for info in scores_struct.values()]
    mean_teds_s = float(np.mean(struct_scores)) if struct_scores else 0.0

    # Compute Text-only Word F1
    mean_f1, std_f1, per_sample_f1 = compute_text_only_metrics(results)

    # Store scores in results for later use
    for fname, info in scores_full.items():
        results[fname]["scores_full"] = info["scores"]
    for fname, info in scores_struct.items():
        results[fname]["scores_struct"] = info["scores"]

    return results, inference_times, mean_teds_full, mean_teds_s, mean_f1, std_f1


# ═══════════════════════════════════════════════════════════
# Main Program
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("UniTable Content Model Fine-tuning Experiment (run_experiment.py v2.1)")
    print("=" * 60)

    # ── Print Configuration ──────────────────────────────────────
    print(f"\n[Configuration Summary]")
    print(f"  SKIP_TRAINING: {SKIP_TRAINING}")
    if SKIP_TRAINING and LOAD_CHECKPOINT_PATH:
        print(f"  LOAD_CHECKPOINT_PATH: {LOAD_CHECKPOINT_PATH}")
    print(f"  TRAIN_EPOCHS: {TRAIN_EPOCHS}")
    print(f"  LEARNING_RATE: {LEARNING_RATE}")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")
    print(f"  USE_LORA: {USE_LORA}")
    if USE_LORA:
        print(f"  LORA_RANK: {LORA_RANK}, LORA_ALPHA: {LORA_ALPHA}")
    print(f"  TRAIN_MAX_SAMPLES: {TRAIN_MAX_SAMPLES}")
    print(f"  VAL_MAX_SAMPLES: {VAL_MAX_SAMPLES}")
    print(f"  ORIGINAL_TEST_SAMPLES: {ORIGINAL_TEST_SAMPLES}")
    print(f"  FINETUNED_TEST_SAMPLES: {FINETUNED_TEST_SAMPLES}")
    print(f"  COMPARE_WITH_ORIGINAL: {COMPARE_WITH_ORIGINAL}")
    print(f"  USE_COLAB: {USE_COLAB}")

    # ── Create Output Directories ────────────────────────────────
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load Vocab ───────────────────────────────────────────────
    print(f"\n[Step 0] Loading Vocab...")
    vocab_c = tk.Tokenizer.from_file(str(UNITABLE_DIR / "vocab" / "vocab_cell_6k.json"))
    vocab_b = tk.Tokenizer.from_file(str(UNITABLE_DIR / "vocab" / "vocab_bbox.json"))
    vocab_s = tk.Tokenizer.from_file(str(UNITABLE_DIR / "vocab" / "vocab_html.json"))
    print(f"  vocab_c size: {vocab_c.get_vocab_size()}")

    # ═══════════════════════════════════════════════════════════
    # Step 1: Training (if SKIP_TRAINING = False)
    # ═══════════════════════════════════════════════════════════
    if not SKIP_TRAINING:
        print(f"\n{'='*60}")
        print(f"[Step 1] Starting training (fine-tuning)...")
        print(f"{'='*60}")

        # Load data
        print(f"\n[Step 1.1] Loading training data...")
        train_samples = load_training_data_from_csv(CSV_PATH, IMAGES_DIR, TRAIN_MAX_SAMPLES, "phase1", "train")
        val_samples = load_training_data_from_csv(CSV_PATH, IMAGES_DIR, VAL_MAX_SAMPLES, "phase1", "val")

        # Load BBox checkpoint
        print(f"\n[Step 1.2] Loading BBox checkpoint...")
        train_bbox_preds = load_bbox_checkpoint("train")
        val_bbox_preds = load_bbox_checkpoint("val")

        if train_bbox_preds is None:
            print("[ERROR] BBox checkpoint not found!")
            print("[HINT] Please run v1 experiment first to generate bbox_preds_train.pt")
            sys.exit(1)

        # Create Dataset and DataLoader
        print(f"\n[Step 1.3] Creating Dataset and DataLoader...")
        train_dataset = ContentFineTuneDataset(train_samples, vocab_c, train_bbox_preds)
        val_dataset = ContentFineTuneDataset(val_samples, vocab_c, val_bbox_preds)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, pin_memory=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=0, pin_memory=True, collate_fn=collate_fn)

        print(f"  Training cells: {len(train_dataset)}")
        print(f"  Validation cells: {len(val_dataset)}")

        # Load Content Model
        print(f"\n[Step 1.4] Loading original Content Model...")
        content_model = load_model(str(MODEL_DIR / "unitable_large_content.pt"))

        # Freeze Backbone and Encoder
        for param in content_model.backbone.parameters():
            param.requires_grad = False
        for param in content_model.encoder.parameters():
            param.requires_grad = False

        # Apply LoRA
        if USE_LORA and HAS_PEFT:
            print(f"\n[Step 1.5] Applying LoRA...")
            lora_config = LoraConfig(
                r=LORA_RANK,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                target_modules=["linear1", "linear2"],
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            content_model = get_peft_model(content_model, lora_config)
            content_model.print_trainable_parameters()
        else:
            trainable_params = [p for n, p in content_model.named_parameters() if p.requires_grad]
            print(f"[INFO] Trainable params: {sum(p.numel() for p in trainable_params):,}")

        # Optimizer
        optimizer = optim.AdamW([p for p in content_model.parameters() if p.requires_grad],
                               lr=LEARNING_RATE, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        # Scheduler
        num_training_steps = len(train_loader) * TRAIN_EPOCHS
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                                    total_iters=10)
        cosine_scheduler = CosineAnnealingLR(optimizer,
                                              T_max=max(1, num_training_steps - 10),
                                              eta_min=LEARNING_RATE * 0.01)
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[10]
        )

        # Training loop
        print(f"\n[Step 1.6] Starting training...")
        print(f"  Epochs: {TRAIN_EPOCHS}")
        print(f"  Steps per epoch: {len(train_loader)}")

        best_val_loss = float('inf')
        for epoch in range(TRAIN_EPOCHS):
            content_model.train()
            total_loss = 0
            n_samples = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{TRAIN_EPOCHS}"):
                cell_images = batch[0].to(device, non_blocking=True)
                target_ids = batch[1].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                tgt_input = target_ids[:, :-1].contiguous()
                tgt_target = target_ids[:, 1:].contiguous()

                causal_mask = subsequent_mask(tgt_input.shape[1]).to(device)
                memory = content_model.encode(cell_images)
                tgt_feature = content_model.pos_embed(content_model.token_embed(tgt_input))
                decoded = content_model.decoder(tgt_feature, memory, causal_mask, None)
                logits = content_model.generator(decoded)

                loss = criterion(logits.view(-1, logits.size(-1)), tgt_target.view(-1))
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    [p for p in content_model.parameters() if p.requires_grad],
                    GRADIENT_CLIP
                )
                optimizer.step()

                total_loss += loss.item() * cell_images.size(0)
                n_samples += cell_images.size(0)

            scheduler.step()
            avg_train_loss = total_loss / n_samples if n_samples > 0 else float('inf')

            # Evaluate
            content_model.eval()
            total_val_loss = 0
            n_val_samples = 0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Evaluating"):
                    cell_images = batch[0].to(device, non_blocking=True)
                    target_ids = batch[1].to(device, non_blocking=True)
                    tgt_input = target_ids[:, :-1].contiguous()
                    tgt_target = target_ids[:, 1:].contiguous()

                    causal_mask = subsequent_mask(tgt_input.shape[1]).to(device)
                    memory = content_model.encode(cell_images)
                    tgt_feature = content_model.pos_embed(content_model.token_embed(tgt_input))
                    decoded = content_model.decoder(tgt_feature, memory, causal_mask, None)
                    logits = content_model.generator(decoded)

                    loss = criterion(logits.view(-1, logits.size(-1)), tgt_target.view(-1))
                    total_val_loss += loss.item() * cell_images.size(0)
                    n_val_samples += cell_images.size(0)

            avg_val_loss = total_val_loss / n_val_samples if n_val_samples > 0 else float('inf')

            print(f"  Epoch {epoch+1}/{TRAIN_EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # Save checkpoint (save entire PEFT model state_dict)
                checkpoint = {
                    'model_state_dict': content_model.state_dict(),
                    'epoch': epoch,
                    'loss': avg_val_loss,
                    'config': {
                        'learning_rate': LEARNING_RATE,
                        'epochs': TRAIN_EPOCHS,
                        'lora_rank': LORA_RANK,
                        'lora_alpha': LORA_ALPHA,
                    }
                }
                best_path = CHECKPOINT_DIR / "content_best.pt"
                torch.save(checkpoint, str(best_path))
                print(f"  [INFO] New best model saved: {best_path}")

    # ═══════════════════════════════════════════════════════════
    # Step 2: Determine the checkpoint path to load
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"[Step 2] Loading model...")
    print(f"{'='*60}")

    if LOAD_CHECKPOINT_PATH:
        # Use specified checkpoint path
        checkpoint_path = Path(LOAD_CHECKPOINT_PATH)
        if not checkpoint_path.exists():
            print(f"[ERROR] Checkpoint does not exist: {checkpoint_path}")
            sys.exit(1)
    else:
        # Use default path
        checkpoint_path = CHECKPOINT_DIR / "content_best.pt"
        if not checkpoint_path.exists():
            print(f"[ERROR] Checkpoint does not exist: {checkpoint_path}")
            print("[HINT] Please run training first or set LOAD_CHECKPOINT_PATH")
            sys.exit(1)

    print(f"  Checkpoint path: {checkpoint_path}")

    # ── Load fine-tuned model ───────────────────────────────────
    # Check if checkpoint contains LoRA keys
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    has_lora_keys = any('lora_' in k for k in ckpt.keys())

    # Check if checkpoint contains 'model_state_dict' key (training save format)
    has_model_state_dict = 'model_state_dict' in ckpt.keys()

    if has_lora_keys and not has_model_state_dict:
        # LoRA checkpoint (not packed), needs merge
        print(f"  [INFO] Detected LoRA checkpoint, needs merge...")
        # Re-create model and load LoRA weights
        print(f"  [INFO] Loading original Content Model...")
        content_model = load_model(str(MODEL_DIR / "unitable_large_content.pt"))

        # Freeze Backbone and Encoder
        for param in content_model.backbone.parameters():
            param.requires_grad = False
        for param in content_model.encoder.parameters():
            param.requires_grad = False

        # Apply LoRA structure
        if USE_LORA and HAS_PEFT:
            lora_config = LoraConfig(
                r=LORA_RANK,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                target_modules=["linear1", "linear2"],
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            content_model = get_peft_model(content_model, lora_config)

        # Load LoRA weights
        content_model.load_state_dict(ckpt)
        print(f"  [INFO] LoRA weights loaded")

        # Merge LoRA weights
        print(f"\n[Step 2.1] Merging LoRA weights...")
        lora_state_dict = content_model.state_dict()
        base_state = torch.load(str(MODEL_DIR / "unitable_large_content.pt"),
                               map_location=device, weights_only=False)
        merged_state_dict = dict(base_state)

        scaling = LORA_ALPHA / LORA_RANK
        print(f"  LoRA rank={LORA_RANK}, alpha={LORA_ALPHA}, scaling={scaling}")

        merged_count = 0
        lora_pairs = 0

        for k in list(lora_state_dict.keys()):
            if '.lora_A.default.weight' in k:
                base_layer_key = k.replace('.lora_A.default.weight', '.base_layer.weight')
                lora_b_key = k.replace('.lora_A.default.weight', '.lora_B.default.weight')

                if base_layer_key in lora_state_dict and lora_b_key in lora_state_dict:
                    lora_a = lora_state_dict[k]
                    lora_b = lora_state_dict[lora_b_key]

                    merged_key = base_layer_key
                    if merged_key.startswith('base_model.model.'):
                        merged_key = merged_key[len('base_model.model.'):]
                    merged_key = merged_key.replace('.base_layer', '')

                    if merged_key in merged_state_dict:
                        original_weight = merged_state_dict[merged_key]
                        lora_update = (lora_b @ lora_a) * scaling

                        if original_weight.shape == lora_update.shape:
                            merged_state_dict[merged_key] = original_weight + lora_update
                            merged_count += 1
                            lora_pairs += 1

        print(f"  Merged {lora_pairs} LoRA pairs → {merged_count} layers")

        # Save merged checkpoint
        merged_path = CHECKPOINT_DIR / "content_best_merged.pt"
        torch.save(merged_state_dict, str(merged_path))
        print(f"  Merged checkpoint saved: {merged_path}")

        # Load merged model
        fine_tuned_model = load_model(str(merged_path))

    elif has_model_state_dict:
        # Training save format (contains model_state_dict, epoch, loss, config)
        # Check if model_state_dict is in LoRA format
        model_sd = ckpt['model_state_dict']
        inner_has_lora = any('lora_' in k for k in model_sd.keys())

        if inner_has_lora:
            print(f"  [INFO] Detected LoRA checkpoint (packed format), needs merge...")
            # Re-create model and load LoRA weights
            print(f"  [INFO] Loading original Content Model...")
            content_model = load_model(str(MODEL_DIR / "unitable_large_content.pt"))

            # Freeze Backbone and Encoder
            for param in content_model.backbone.parameters():
                param.requires_grad = False
            for param in content_model.encoder.parameters():
                param.requires_grad = False

            # Apply LoRA structure
            if USE_LORA and HAS_PEFT:
                lora_config = LoraConfig(
                    r=LORA_RANK,
                    lora_alpha=LORA_ALPHA,
                    lora_dropout=LORA_DROPOUT,
                    target_modules=["linear1", "linear2"],
                    task_type=TaskType.FEATURE_EXTRACTION,
                )
                content_model = get_peft_model(content_model, lora_config)

            # Load LoRA weights
            content_model.load_state_dict(model_sd)
            print(f"  [INFO] LoRA weights loaded")

            # Merge LoRA weights
            print(f"\n[Step 2.1] Merging LoRA weights...")
            lora_state_dict = content_model.state_dict()
            base_state = torch.load(str(MODEL_DIR / "unitable_large_content.pt"),
                                   map_location=device, weights_only=False)
            merged_state_dict = dict(base_state)

            scaling = LORA_ALPHA / LORA_RANK
            print(f"  LoRA rank={LORA_RANK}, alpha={LORA_ALPHA}, scaling={scaling}")

            merged_count = 0
            lora_pairs = 0

            for k in list(lora_state_dict.keys()):
                if '.lora_A.default.weight' in k:
                    base_layer_key = k.replace('.lora_A.default.weight', '.base_layer.weight')
                    lora_b_key = k.replace('.lora_A.default.weight', '.lora_B.default.weight')

                    if base_layer_key in lora_state_dict and lora_b_key in lora_state_dict:
                        lora_a = lora_state_dict[k]
                        lora_b = lora_state_dict[lora_b_key]

                        merged_key = base_layer_key
                        if merged_key.startswith('base_model.model.'):
                            merged_key = merged_key[len('base_model.model.'):]
                        merged_key = merged_key.replace('.base_layer', '')

                        if merged_key in merged_state_dict:
                            original_weight = merged_state_dict[merged_key]
                            lora_update = (lora_b @ lora_a) * scaling

                            if original_weight.shape == lora_update.shape:
                                merged_state_dict[merged_key] = original_weight + lora_update
                                merged_count += 1
                                lora_pairs += 1

            print(f"  Merged {lora_pairs} LoRA pairs → {merged_count} layers")

            # Save merged checkpoint
            merged_path = CHECKPOINT_DIR / "content_best_merged.pt"
            torch.save(merged_state_dict, str(merged_path))
            print(f"  Merged checkpoint saved: {merged_path}")

            # Load merged model
            fine_tuned_model = load_model(str(merged_path))
        else:
            # model_state_dict is already in merged format
            print(f"  [INFO] Checkpoint is already in merged format (packed format), loading directly...")
            # Create model structure and load state_dict
            merged_state = torch.load(str(MODEL_DIR / "unitable_large_content.pt"),
                                     map_location=device, weights_only=False)
            # Override parameters in merged_state with model_state_dict
            for k in model_sd.keys():
                if k in merged_state:
                    merged_state[k] = model_sd[k]
            # Save temporary merged checkpoint
            merged_path = CHECKPOINT_DIR / "content_best_merged.pt"
            torch.save(merged_state, str(merged_path))
            fine_tuned_model = load_model(str(merged_path))
    else:
        # Checkpoint is already in merged state_dict (original format)
        print(f"  [INFO] Checkpoint is already in merged format, loading directly...")
        fine_tuned_model = load_model(str(checkpoint_path))
        merged_path = checkpoint_path

    # Load Structure and BBox models
    print(f"\n[Step 2.2] Loading Structure and BBox models...")
    structure_model = load_model(str(MODEL_DIR / "unitable_large_structure.pt"))
    bbox_model = load_model(str(MODEL_DIR / "unitable_large_bbox.pt"))

    # ═══════════════════════════════════════════════════════════
    # Step 3: Evaluation
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"[Step 3] Evaluating model...")
    print(f"{'='*60}")

    # Load test data
    test_samples = load_training_data_from_csv(CSV_PATH, IMAGES_DIR,
                                                FINETUNED_TEST_SAMPLES, "test", "test")

    if not test_samples:
        print("[ERROR] No test data found!")
        sys.exit(1)

    # Evaluate fine-tuned model
    print(f"\n[Step 3.1] Evaluating fine-tuned Content Model...")
    fine_tuned_results, ft_times, ft_teds_full, ft_teds_s, ft_f1, ft_f1_std = evaluate_model(
        content_model=fine_tuned_model,
        structure_model=structure_model,
        bbox_model=bbox_model,
        vocab_s=vocab_s, vocab_b=vocab_b, vocab_c=vocab_c,
        samples=test_samples,
        model_name="fine_tuned"
    )

    print(f"  fine_tuned: TEDS (Content) = {ft_teds_full:.4f}")
    print(f"  fine_tuned: TEDS-S (Struct)  = {ft_teds_s:.4f}")
    print(f"  fine_tuned: Text-only F1     = {ft_f1:.4f} ± {ft_f1_std:.4f}")

    # Save fine-tuned results
    ft_data = {
        "model": "fine_tuned",
        "checkpoint": str(merged_path),
        "mean_teds_full": ft_teds_full,
        "mean_teds_s": ft_teds_s,
        "mean_f1": ft_f1,
        "std_f1": ft_f1_std,
        "dataset": "v2_1300sample",
        "split": "test",
        "n_samples": len(fine_tuned_results),
    }
    with open(OUTPUT_DIR / "fine_tuned_model_result.json", "w") as f:
        json.dump(ft_data, f, indent=2)

    # Evaluate original model (if comparison needed)
    original_results = None
    if COMPARE_WITH_ORIGINAL:
        print(f"\n[Step 3.2] Evaluating original Content Model...")
        original_model = load_model(str(MODEL_DIR / "unitable_large_content.pt"))
        original_results, orig_times, orig_teds_full, orig_teds_s, orig_f1, orig_f1_std = evaluate_model(
            content_model=original_model,
            structure_model=structure_model,
            bbox_model=bbox_model,
            vocab_s=vocab_s, vocab_b=vocab_b, vocab_c=vocab_c,
            samples=test_samples,
            model_name="original"
        )

        print(f"  original: TEDS (Content) = {orig_teds_full:.4f}")
        print(f"  original: TEDS-S (Struct)  = {orig_teds_s:.4f}")
        print(f"  original: Text-only F1     = {orig_f1:.4f} ± {orig_f1_std:.4f}")

        # Save original results
        orig_data = {
            "model": "original",
            "checkpoint": str(MODEL_DIR / "unitable_large_content.pt"),
            "mean_teds_full": orig_teds_full,
            "mean_teds_s": orig_teds_s,
            "mean_f1": orig_f1,
            "std_f1": orig_f1_std,
            "dataset": "v2_1300sample",
            "split": "test",
            "n_samples": len(original_results),
        }
        with open(OUTPUT_DIR / "original_model_result.json", "w") as f:
            json.dump(orig_data, f, indent=2)

    # ═══════════════════════════════════════════════════════════
    # Step 4: Compare Results
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"[Step 4] Comparing Results")
    print(f"{'='*60}")

    if original_results:
        orig_scores_full = [info.get("scores_full", 0) for info in original_results.values()]
        orig_scores_struct = [info.get("scores_struct", 0) for info in original_results.values()]
        orig_mean_full = float(np.mean(orig_scores_full)) if orig_scores_full else 0.0
        orig_mean_struct = float(np.mean(orig_scores_struct)) if orig_scores_struct else 0.0

        ft_scores_full = [info.get("scores_full", 0) for info in fine_tuned_results.values()]
        ft_scores_struct = [info.get("scores_struct", 0) for info in fine_tuned_results.values()]
        ft_mean_full = float(np.mean(ft_scores_full)) if ft_scores_full else 0.0
        ft_mean_struct = float(np.mean(ft_scores_struct)) if ft_scores_struct else 0.0

        # Compute improvements
        imp_full = ((ft_mean_full - orig_mean_full) / orig_mean_full * 100) if orig_mean_full > 0 else 0
        imp_struct = ((ft_mean_struct - orig_mean_struct) / orig_mean_struct * 100) if orig_mean_struct > 0 else 0
        imp_f1 = ft_f1 - orig_f1

        print(f"\n{'Metric':<30} {'Original Model':<15} {'Fine-tuned':<15} {'Improvement':<10}")
        print(f"{'-'*70}")
        print(f"{'TEDS (Content)':<30} {orig_mean_full:<15.4f} {ft_mean_full:<15.4f} {imp_full:+.2f}%")
        print(f"{'TEDS-S (Structure)':<30} {orig_mean_struct:<15.4f} {ft_mean_struct:<15.4f} {imp_struct:+.2f}%")
        print(f"{'Text-only Word F1':<30} {orig_f1:<15.4f} {ft_f1:<15.4f} {imp_f1:+.4f}")

        print()
        if imp_full > 0:
            print(f"✓ TEDS (Content) improved by {imp_full:+.2f}%, fine-tuning is effective!")
        else:
            print(f"⚠ TEDS (Content) did not improve ({imp_full:+.2f}%)")

        if imp_struct > 0:
            print(f"✓ TEDS-S (Structure) improved by {imp_struct:+.2f}%")
        else:
            print(f"⚠ TEDS-S (Structure) did not improve ({imp_struct:+.2f}%)")

        if imp_f1 > 0:
            print(f"✓ Text-only Word F1 improved by {imp_f1:+.4f}")
        else:
            print(f"⚠ Text-only Word F1 did not improve ({imp_f1:+.4f})")
    else:
        print(f"\n  Skipping comparison (original model not evaluated)")
        print(f"  fine_tuned: TEDS (Content) = {ft_teds_full:.4f}")
        print(f"  fine_tuned: TEDS-S (Struct)  = {ft_teds_s:.4f}")
        print(f"  fine_tuned: Text-only F1     = {ft_f1:.4f} ± {ft_f1_std:.4f}")

    print(f"\n{'='*60}")
    print(f"[DONE] Experiment completed!")
    print(f"{'='*60}")
    print(f"  Checkpoint: {merged_path}")
    print(f"  Results saved to: {OUTPUT_DIR}")