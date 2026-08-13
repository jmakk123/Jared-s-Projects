# UniTable Streamlit Integration Guide

## Quick Start for Teammates

This guide explains how to integrate the fine-tuned UniTable model into your Streamlit application. **No prior experience with the codebase is required.**

---

## 📦 What's Included

```
unitable_bundle/
├── checkpoints/
│   └── content_best_merged.pt      ← Fine-tuned Content Model (LoRA merged)
├── models/
│   ├── unitable_large_structure.pt  ← Structure Prediction Model
│   ├── unitable_large_bbox.pt       ← Bounding Box Prediction Model
│   └── unitable_large_content.pt    ← Original Content Model (for reference)
├── vocab/
│   ├── vocab_cell_6k.json           ← Cell content tokenizer
│   ├── vocab_html.json              ← HTML/structure tokenizer
│   └── vocab_bbox.json              ← Bounding box tokenizer
├── loader.py             ← Model loading utility (copy to your app)
└── README.md             ← This file
```

---

## 🚀 Step 1: Install Dependencies

Run this in your terminal:

```bash
pip install torch torchvision tokenizers pillow numpy pandas streamlit tqdm
```

If you don't have CUDA, install CPU-only PyTorch:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

You also need the `peft` library (for LoRA support during training, if needed later):

```bash
pip install peft
```

---

## 🚀 Step 2: Set Up Your Streamlit App

### Option A: Use the Helper Function (Recommended)

Copy `loader.py` into your Streamlit project directory, then use it like this:

```python
import streamlit as st
from loader import load_unitable_models

st.title("UniTable Table Extraction")

# Load models once at app startup
@st.cache_resource
def load_models():
    return load_unitable_models(
        base_dir="./unitable_bundle",  # Adjust path as needed
        use_cpu=False               # Set True if no GPU
    )

structure_model, bbox_model, content_model, vocab_s, vocab_b, vocab_c = load_models()

# Models are now ready for inference!
st.success("Models loaded successfully!")
```

### Option B: Minimal Manual Loading

If you prefer not to use the helper, here's the minimum code needed:

```python
import torch
import tokenizers as tk
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
vocab_c = tk.Tokenizer.from_file("./unitable_bundle/vocab/vocab_cell_6k.json")

# Load merged checkpoint
checkpoint = torch.load(
    "./unitable_bundle/checkpoints/content_best_merged.pt",
    map_location=device,
    weights_only=False
)

# You'll need the model architecture code from run_experiment.py
# Or use loader.py which handles this automatically
```

---

## 🚀 Step 3: Replace Content Model in Your Existing Code

Find where you load the Content model in your current Streamlit app:

```python
# BEFORE (original model):
content_model = load_model("./models/unitable_large_content.pt")

# AFTER (fine-tuned model):
content_model = load_model("./unitable_bundle/checkpoints/content_best_merged.pt")
```

**That's the only change needed!** The Structure and BBox models remain the same.

---

## 📊 Expected Performance Improvement

| Metric | Original | Fine-tuned | Improvement |
|--------|----------|------------|-------------|
| TEDS (Content) | 0.3389 | 0.3417 | +0.83% |
| TEDS-S (Structure) | 0.8353 | 0.8353 | 0.00% |
| Word F1 | 0.4829 | 0.4928 | +2.05% |

**Key observations:**
- OCR text accuracy improved by ~2%
- Structure prediction unchanged (as expected)
- Improvement is most noticeable on low-quality images

---

## 🔧 Troubleshooting

### Problem: "Cannot find module 'tokenizers'"
**Solution:** Install the tokenizers package:
```bash
pip install tokenizers
```

### Problem: "CUDA out of memory"
**Solution:** Use CPU mode or reduce batch size:
```python
structure_model, bbox_model, content_model, vocab_s, vocab_b, vocab_c = load_unitable_models(
    base_dir="./unitable_bundle",
    use_cpu=True  # Force CPU mode
)
```

### Problem: "Checkpoint file not found"
**Solution:** Verify the path is correct:
```python
# Use absolute path if relative path doesn't work
import os
handoff_path = os.path.abspath("./unitable_bundle")
models = load_unitable_models(base_dir=handoff_path)
```

### Problem: "Model architecture mismatch"
**Solution:** The `load_model_from_state()` function in `loader.py` automatically infers architecture from the checkpoint. Make sure you're using this helper function rather than loading manually.

---

## 📝 Full Inference Example

Here's a complete example of running inference with the fine-tuned model:

```python
import streamlit as st
from PIL import Image
import torch
from loader import load_unitable_models

# ── Load Models ──────────────────────────────────────────────
@st.cache_resource
def load_models():
    return load_unitable_models(base_dir="./unitable_bundle", use_cpu=False)

structure_model, bbox_model, content_model, vocab_s, vocab_b, vocab_c = load_models()

# ── Upload Image ─────────────────────────────────────────────
st.title("UniTable Table Extraction")
uploaded_file = st.file_uploader("Upload a table image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Table", width=600)
    
    if st.button("Extract Table"):
        with st.spinner("Processing..."):
            # Run your three-pass pipeline here using the loaded models
            # Pass 1: Structure → HTML tags
            # Pass 2: BBox → Coordinates
            # Pass 3: Content → Cell text (using fine-tuned content_model)
            result = run_unitable_pipeline(image, structure_model, bbox_model, content_model)
            st.html(result)  # Display HTML table
```

---

## 📁 Directory Structure Reference

| File | Purpose | Size (approx) |
|------|---------|---------------|
| `content_best_merged.pt` | Fine-tuned Content Model (LoRA merged) | ~1.2 GB |
| `unitable_large_structure.pt` | Structure prediction model | ~1.2 GB |
| `unitable_large_bbox.pt` | Bounding box prediction model | ~1.2 GB |
| `unitable_large_content.pt` | Original Content model (before fine-tuning) | ~1.2 GB |
| `vocab_cell_6k.json` | Cell content vocabulary (6000 tokens) | ~50 KB |
| `vocab_html.json` | HTML tag vocabulary | ~50 KB |
| `vocab_bbox.json` | Bounding box vocabulary | ~10 KB |

---

## 🆘 Getting Help

If you encounter issues:

1. Check that all files in `unitable_bundle/` are present
2. Verify Python packages are installed (`pip list | grep -E "torch|tokenizers|streamlit"`)
3. Make sure the `unitable` source code is accessible (add to `sys.path` if needed)
4. Check the `loader.py` file for detailed function documentation

---

## 📋 Checklist

- [ ] Installed Python dependencies (`torch`, `tokenizers`, `streamlit`, etc.)
- [ ] Copied `loader.py` to your project
- [ ] Verified all files in `unitable_bundle/` are present
- [ ] Updated Content model path to use `content_best_merged.pt`
- [ ] Tested model loading with a simple script
- [ ] Ran inference on a test image

---

**Last updated:** 2026-05-19
**Model version:** v2 (LoRA fine-tuned, rank=8, alpha=16, 3 epochs)