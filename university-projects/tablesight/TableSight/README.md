# TableSight

A side-by-side dashboard for hierarchical table extraction. Drop a table image in, see what three different models do with it, and score them against ground truth.

## The models

- **Florence-2** — vision-language model, LoRA fine-tuned on FinTabNet. Falls back to its native OCR-with-region task and clusters the boxes into a grid when no adapter is loaded.
- **UniTable** — three-pass encoder-decoder: predicts the table structure, then the bounding boxes, then per-cell text. Content model fine-tuned with LoRA.
- **SPARTAN** — an OCR-first heuristic: EasyOCR finds the words, a 1D KMeans clusters them into columns, a small set of rules infers the header rows. Tuned parameters live in `models/spartan/*.json`.

## Quick start

```bash
pip install -r TableSight/requirements.txt
brew install tesseract        # macOS — on Linux: apt install tesseract-ocr
streamlit run TableSight/dashboard/app.py
```

The dashboard opens at <http://localhost:8501>. Pick a sample image (or upload your own), hit *Run*, and compare.

## Sharing it on your network

Bind to all interfaces so other devices on the same Wi-Fi can reach it:

```bash
streamlit run TableSight/dashboard/app.py \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
```

Find your LAN IP with `ipconfig getifaddr en0` (macOS) or `hostname -I` (Linux) and others can hit `http://<that-ip>:8501`.

## Running it in the background

```bash
nohup streamlit run TableSight/dashboard/app.py \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    > /tmp/tablesight.log 2>&1 &
echo $! > /tmp/tablesight.pid
```

Logs: `tail -f /tmp/tablesight.log`. Stop with `kill $(cat /tmp/tablesight.pid)`.

## Using it from Python

```python
from TableSight.models.inference import TableExtractor
from PIL import Image

ext = TableExtractor("spartan", checkpoint_path="TableSight/models/spartan")
result = ext.predict(Image.open("table.png"))
print(result["html"])
```

Model names: `"florence"`, `"unitable"`, `"spartan"`.

## Retraining SPARTAN

SPARTAN's parameters are JSON files the runner reads at startup. To retune them on FinTabNet:

```bash
python TableSight/notebooks/train_spartan.py --train
```

The script picks up Colab vs local, downloads FinTabNet from HuggingFace if there's no local cache, and writes new `preprocessing_params.json` + `grid_params.json` into `models/spartan/`. Reload the dashboard to apply.

## Model checkpoints

The Florence-2 LoRA adapter (~1 GB) and the UniTable weights (~3 GB total) aren't in this repo. Point `dashboard/config.yaml` at wherever they live on the machine:

```yaml
checkpoints:
  florence: "/path/to/florence2_lora_adapter"   # PEFT adapter directory
  unitable: "/path/to/unitable_weights"         # bundle with models/ + vocab/
  spartan:  "TableSight/models/spartan"         # included
```

The sidebar shows ✓ or ⚠ next to each model based on whether the checkpoint resolved.

## Layout

```
TableSight/
├── README.md
├── requirements.txt
├── dashboard/
│   ├── app.py
│   ├── config.yaml
│   └── utils/                       model loading, TEDS scoring, charts
├── models/
│   ├── inference.py                 unified TableExtractor interface
│   ├── florence_runner.py
│   ├── unitable_runner.py
│   ├── spartan/
│   │   ├── pipeline.py              SPARTAN runner
│   │   ├── prepare_data.py          FinTabNet → local splits
│   │   ├── preprocessing_params.json
│   │   ├── grid_params.json
│   │   └── benchmark.csv
│   └── unitable_bundle/             UniTable source + vocab + loader
└── notebooks/
    └── train_spartan.py             tune SPARTAN params + benchmark
```
