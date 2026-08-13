#!/usr/bin/env python3
"""
run_benchmark_by_type.py - UniTable Benchmark by Image Type

Features:
- Read image_type field from CSV
- Evaluate original and fine-tuned models separately for each image_type
- Compute three metrics for each model:
  - TEDS (Content) — Standard TEDS (includes structure and content)
  - TEDS-S (Structure) — Structure only
  - Text-only Word F1 — Content only

Usage:
```bash
python run_benchmark_by_type.py
```
"""

# ═══════════════════════════════════════════════════════════
# [Configuration Section]
# ═══════════════════════════════════════════════════════════

# ── Evaluation Settings ──────────────────────────────────────
TEST_SAMPLES = -1            # Number of test samples (-1 = all)
EVAL_ORIGINAL = True         # True = Evaluate original model
EVAL_FINETUNED = True        # True = Evaluate fine-tuned model

# ── Path Settings ────────────────────────────────────────────
USE_COLAB = True             # True = Colab mode, False = Local mode

# ═══════════════════════════════════════════════════════════
# [Do not modify below] Unless you know what you are doing
# ═══════════════════════════════════════════════════════════

import os
import sys
import json
import re
import time
from pathlib import Path
from functools import partial

# ── Detect Colab Environment ──────────────────────────────────────
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

# ── Imports ────────────────────────────────────────────────────
import torch
import torch.nn as nn
import tokenizers as tk
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

# ── Token lists ────────────────────────────────────────────────
VALID_HTML_TOKEN = ["<eos>"] + HTML_TOKENS
INVALID_CELL_TOKEN = ["<sos>", "<pad>", "<empty>", "<sep>"] + TASK_TOKENS + RESERVED_TOKENS
VALID_BBOX_TOKEN = ["<eos>"] + BBOX_TOKENS[:449]

# ── Image normalization parameters ─────────────────────────────
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
    mean_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    std_f1 = float(np.std(f1_scores)) if f1_scores else 0.0
    return mean_f1, std_f1


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


@torch.no_grad()
def autoregressive_decode(model, image, prefix, max_decode_len, eos_id,
                          token_whitelist=None, token_blacklist=None):
    """Original autoregressive decode (from run_experiment.py)"""
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
                                 split: str = "test") -> list:
    """Load data from CSV for specified split (including image_type)"""
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
        image_type = str(row.get('image_type', 'unknown'))
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
            'image_type': image_type,
            'image_path': img_path,
        })
    return results


# ═══════════════════════════════════════════════════════════
# Evaluation Function
# ═══════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_model_single(content_model, structure_model, bbox_model,
                          vocab_s, vocab_b, vocab_c, samples):
    """
    Evaluate model using original single-sample inference and return results (includes three metrics)
    """
    results = {}
    for i, sample in enumerate(samples):
        fname = sample['filename']
        gt_html = sample['gt_html']
        img_path = sample['image_path']
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

        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            print(f"    [{i+1}/{len(samples)}] Completed {fname}")

    # Compute TEDS (Content)
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
    mean_f1, std_f1 = compute_text_only_metrics(results)

    return results, mean_teds_full, mean_teds_s, mean_f1, std_f1


# ═══════════════════════════════════════════════════════════
# Main Program
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("UniTable Benchmark by Image Type")
    print("=" * 60)

    # ── Create Output Directory ──────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load Vocab ───────────────────────────────────────────────
    print(f"\n[Step 1] Loading Vocab...")
    vocab_c = tk.Tokenizer.from_file(str(UNITABLE_DIR / "vocab" / "vocab_cell_6k.json"))
    vocab_b = tk.Tokenizer.from_file(str(UNITABLE_DIR / "vocab" / "vocab_bbox.json"))
    vocab_s = tk.Tokenizer.from_file(str(UNITABLE_DIR / "vocab" / "vocab_html.json"))
    print(f"  vocab_s size: {vocab_s.get_vocab_size()}")
    print(f"  vocab_b size: {vocab_b.get_vocab_size()}")
    print(f"  vocab_c size: {vocab_c.get_vocab_size()}")

    # ── Load Models ──────────────────────────────────────────────
    print(f"\n[Step 2] Loading models...")
    print("  Loading Structure model...")
    structure_model = load_model(str(MODEL_DIR / "unitable_large_structure.pt"))
    print("  Loading BBox model...")
    bbox_model = load_model(str(MODEL_DIR / "unitable_large_bbox.pt"))

    # Load Content models
    models_to_eval = {}

    # Original model
    if EVAL_ORIGINAL:
        print(f"  Loading original Content model...")
        original_model = load_model(str(MODEL_DIR / "unitable_large_content.pt"))
        models_to_eval["original"] = original_model
        print("  ✅ original model loaded successfully")

    # Fine-tuned model
    if EVAL_FINETUNED:
        fine_tuned_checkpoint = CHECKPOINT_DIR / "content_best_merged.pt"
        if fine_tuned_checkpoint.exists():
            print(f"  Loading fine-tuned Content model: {fine_tuned_checkpoint}")
            fine_tuned_model = load_model(str(fine_tuned_checkpoint))
            models_to_eval["finetuned"] = fine_tuned_model
        else:
            print(f"  [WARN] Fine-tuned checkpoint does not exist, skipping fine-tuned evaluation")

    if not models_to_eval:
        print("  [ERROR] No Content model to evaluate!")
        sys.exit(1)

    print(f"  ✅ Ready to evaluate {len(models_to_eval)} model(s): {list(models_to_eval.keys())}")

    # ── Load Test Data ───────────────────────────────────────────
    print(f"\n[Step 3] Loading test data...")
    all_samples = load_training_data_from_csv(CSV_PATH, IMAGES_DIR, TEST_SAMPLES, "test")
    print(f"  Test split has {len(all_samples)} samples in total")

    # Group by image_type
    import pandas as pd
    df_all = pd.DataFrame(all_samples)
    if 'image_type' in df_all.columns:
        image_types = df_all['image_type'].unique().tolist()
        print(f"  Image types: {image_types}")
    else:
        image_types = ['all']
        for s in all_samples:
            s['image_type'] = 'all'

    # ── Evaluate for Each image_type ──────────────────────────────
    print(f"\n[Step 4] Starting evaluation...")
    print(f"{'='*60}")

    # Store results for each image_type
    type_results = {}

    overall_t0 = time.time()

    for itype in image_types:
        type_samples = [s for s in all_samples if s['image_type'] == itype]
        print(f"\n  Image type: {itype} ({len(type_samples)} samples)")
        type_results[itype] = {}

        for model_name, content_model in models_to_eval.items():
            print(f"\n    Evaluating {model_name} model...")
            t0 = time.time()
            results, teds_full, teds_s, f1, f1_std = evaluate_model_single(
                content_model=content_model,
                structure_model=structure_model,
                bbox_model=bbox_model,
                vocab_s=vocab_s, vocab_b=vocab_b, vocab_c=vocab_c,
                samples=type_samples,
            )
            elapsed = time.time() - t0
            type_results[itype][model_name] = {
                'teds_full': teds_full,
                'teds_s': teds_s,
                'text_f1': f1,
                'text_f1_std': f1_std,
                'n_samples': len(results),
            }
            print(f"      TEDS (Content) = {teds_full:.4f}")
            print(f"      TEDS-S (Struct)  = {teds_s:.4f}")
            print(f"      Text-only F1     = {f1:.4f} ± {f1_std:.4f}")
            print(f"      Elapsed: {elapsed:.1f}s")

    total_elapsed = time.time() - overall_t0
    print(f"\n  Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    # ── Output Results ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Average TEDS by image type:")
    print(f"{'='*60}")

    # Create DataFrame output
    result_rows = []
    for itype in image_types:
        row = {'image_type': itype}
        for model_name in models_to_eval.keys():
            r = type_results[itype][model_name]
            row[f'{model_name}_teds'] = round(r['teds_full'], 4)
            row[f'{model_name}_s'] = round(r['teds_s'], 4)
            row[f'{model_name}_f1'] = round(r['text_f1'], 4)
        result_rows.append(row)
    df_result = pd.DataFrame(result_rows)
    df_result = df_result.set_index('image_type')
    print(df_result.to_string())

    # Overall average
    print(f"\nOverall average TEDS:")
    overall = df_result.mean()
    for col in df_result.columns:
        print(f"{col:<20} {overall[col]:.4f}")

    # ── Save Results ─────────────────────────────────────────────
    result_json = {
        'by_type': {},
        'overall': {},
    }
    for itype in image_types:
        result_json['by_type'][itype] = {}
        for model_name in models_to_eval.keys():
            r = type_results[itype][model_name]
            result_json['by_type'][itype][model_name] = {
                'teds_full': round(r['teds_full'], 4),
                'teds_s': round(r['teds_s'], 4),
                'text_f1': round(r['text_f1'], 4),
                'text_f1_std': round(r['text_f1_std'], 4),
                'n_samples': r['n_samples'],
            }
    result_json['overall'] = {}
    for model_name in models_to_eval.keys():
        result_json['overall'][model_name] = {
            'teds_full': round(float(df_result[f'{model_name}_teds'].mean()), 4),
            'teds_s': round(float(df_result[f'{model_name}_s'].mean()), 4),
            'text_f1': round(float(df_result[f'{model_name}_f1'].mean()), 4),
        }

    result_path = OUTPUT_DIR / "benchmark_by_type_result.json"
    with open(result_path, "w") as f:
        json.dump(result_json, f, indent=2)
    print(f"\nResults saved to: {result_path}")

    # Save CSV
    csv_path = OUTPUT_DIR / "benchmark_by_type_result.csv"
    df_result.to_csv(csv_path)
    print(f"CSV saved to: {csv_path}")

    print(f"\n{'='*60}")
    print(f"[DONE] Benchmark completed! Total elapsed: {total_elapsed:.1f}s")
    print(f"{'='*60}")