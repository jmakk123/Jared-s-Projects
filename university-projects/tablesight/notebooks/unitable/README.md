# UniTable Content Model Fine-tuning Experiments

This repository contains experimental scripts for fine-tuning the **Content Model** of the UniTable architecture, a table recognition system that converts table images to HTML format.

## Overview

UniTable uses a three-stage pipeline for table recognition:
1. **Structure Model** - Predicts the HTML structure (table tags, rows, cells)
2. **BBox Model** - Predicts bounding box coordinates for each cell
3. **Content Model** - Predicts the text content inside each cell

These experiments focus on **fine-tuning the Content Model** using LoRA (Low-Rank Adaptation) to improve text recognition accuracy while keeping the Structure and BBox models frozen.

## Files

### 1. `run_experiment.py`
**Main fine-tuning experiment script**

Encapsulates the complete training pipeline for fine-tuning the Content Model:
- Training with configurable epochs, learning rate, batch size
- LoRA weight application and automatic merging
- Automatic evaluation comparing fine-tuned vs. original models
- Outputs three key metrics

**Key Features:**
- Supports both Colab and local execution environments
- Configurable training parameters via the configuration section at the top
- Automatic LoRA weight merging to produce a standalone checkpoint
- Comparison with original model baseline

**Usage:**
```bash
# Full pipeline: training + merge + evaluation
python run_experiment.py

# Evaluation only (using existing checkpoint)
# Set SKIP_TRAINING = True and specify LOAD_CHECKPOINT_PATH
python run_experiment.py
```

### 2. `debug_build_table_alignment.py`
**Debug script for table alignment verification**

Used to diagnose alignment issues in the `build_table_robust` function, which maps predicted cell contents to their correct positions in the HTML table structure.

**Key Features:**
- Selects 20 random samples from the test split
- Runs full inference pipeline (Structure + BBox + Content) for both original and fine-tuned models
- Outputs 4 files per sample:
  - `{sample}_gt.html` — Ground Truth HTML
  - `{sample}_pred_fine_tuned.html` — Fine-tuned model prediction
  - `{sample}_pred_original.html` — Original model prediction
  - `{sample}_summary.txt` — Brief comparison summary showing cell-by-cell differences

**Usage:**
```bash
python debug_build_table_alignment.py
```

### 3. `run_benchmark_by_type.py`
**Benchmark evaluation by image type**

Evaluates both original and fine-tuned models separately for each image type category, providing granular performance analysis.

**Key Features:**
- Groups samples by `image_type` field from the dataset CSV
- Evaluates both original and fine-tuned models per type
- Outputs per-type and overall average metrics
- Results saved as both JSON and CSV

**Usage:**
```bash
python run_benchmark_by_type.py
```

## Architecture

```
Input Image (448x448)
        │
        ├──→ Structure Model → HTML tokens (<tr>, <td>, etc.)
        │
        ├──→ BBox Model → Bounding box coordinates
        │
        └──→ Content Model → Cell text content
                │
                ↓
        build_table_robust()
                │
                ↓
        Final HTML Table
```

### Model Details

| Model | Purpose | Vocabulary Size |
|-------|---------|-----------------|
| Structure | Predicts HTML structure tags | vocab_html.json |
| BBox | Predicts normalized bounding box coordinates | vocab_bbox.json |
| Content | Predicts cell text content | vocab_cell_6k.json |

### Content Model Architecture

- **Backbone:** ImgLinearBackbone (d_model=768, patch_size=16)
- **Encoder:** Transformer encoder (6 layers, 12 attention heads)
- **Decoder:** Transformer decoder (8 layers, 12 attention heads)
- **FF Ratio:** 4 (feed-forward dimension = 3072)

## Evaluation Metrics

| Metric | Description | What it Measures |
|--------|-------------|------------------|
| **TEDS (Content)** | Tree Edit Distance Similarity (full) | Overall similarity including structure and content |
| **TEDS-S (Structure)** | Tree Edit Distance Similarity (structure-only) | Structure accuracy only |
| **Text-only Word F1** | Word-level Precision/Recall/F1 | Text content accuracy only |

## Configuration

All scripts have a configuration section at the top. Key parameters:

### Training Parameters (`run_experiment.py`)
```python
TRAIN_EPOCHS = 3           # Number of training epochs
LEARNING_RATE = 5e-5       # Learning rate
BATCH_SIZE = 256            # Batch size
USE_LORA = True            # Enable LoRA fine-tuning
LORA_RANK = 8              # LoRA rank
LORA_ALPHA = 16.0          # LoRA alpha
```

### Evaluation Parameters
```python
SKIP_TRAINING = False      # Skip training, evaluate only
COMPARE_WITH_ORIGINAL = True  # Compare with original model
TEST_SAMPLES = -1          # Number of test samples (-1 = all)
```

## Output Files

### Checkpoints
- `content_best.pt` — Best validation checkpoint (packed format with metadata)
- `content_best_merged.pt` — Merged checkpoint (LoRA weights merged into base model)

### Results
- `fine_tuned_model_result.json` — Fine-tuned model evaluation results
- `original_model_result.json` — Original model evaluation results
- `benchmark_by_type_result.json` — Per-type benchmark results
- `benchmark_by_type_result.csv` — Per-type benchmark results (CSV format)

### Debug Samples (`debug_build_table_alignment.py`)
- `{sample}_gt.html` — Ground truth
- `{sample}_pred_original.html` — Original model prediction
- `{sample}_pred_fine_tuned.html` — Fine-tuned model prediction
- `{sample}_summary.txt` — Cell-by-cell comparison

## Requirements

- Python 3.8+
- PyTorch
- `peft` library for LoRA support
- `tokenizers` for vocabulary management
- `Pillow` for image processing
- `torchvision` for image transforms

## Directory Structure

```
github/
├── run_experiment.py          # Main fine-tuning experiment
├── debug_build_table_alignment.py  # Debug/alignment verification
├── run_benchmark_by_type.py   # Per-type benchmark evaluation
└── Readme.md                  # This file
```

## License

This project uses the UniTable architecture. See the original UniTable repository for license information.