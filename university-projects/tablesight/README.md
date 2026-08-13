# TableSight — Hierarchical Table Extraction

**Advanced Computer Vision · MS-ADS · Spring 2026**
Team: Ryan Chen, Lawrence Lin, Jared Maksoud
Original repo: https://github.com/RyanChenJung/Hierarchical_Table_PDF_Extraction

A side-by-side dashboard for pulling structure out of table images. Drop an image in, see
what three different approaches do with it, and score each against ground truth.

### The three approaches

- **Florence-2** — vision-language model, LoRA fine-tuned on FinTabNet. Falls back to its
  native OCR-with-region task and clusters the boxes into a grid when no adapter is loaded.
- **UniTable** — three-pass encoder-decoder: predicts table structure, then bounding boxes,
  then per-cell text. Content model fine-tuned with LoRA.
- **SPARTAN** — OCR-first heuristic. EasyOCR finds the words, 1D K-Means clusters them into
  columns, and a small rule set infers the header rows. Tuned parameters live in
  `TableSight/models/spartan/*.json`.

### Running it

```bash
pip install -r TableSight/requirements.txt
brew install tesseract          # macOS. Linux: apt install tesseract-ocr
streamlit run TableSight/dashboard/app.py
```

Opens at <http://localhost:8501>.

### Data and model weights (not committed)

- **FinTabNet** — financial table images with structure annotations.
  https://developer.ibm.com/exchanges/data/all/fintabnet/
- **PubTables-1M / TableBank** (optional comparisons) —
  https://github.com/microsoft/table-transformer · https://github.com/doc-analysis/TableBank
- **Florence-2 base weights** — https://huggingface.co/microsoft/Florence-2-base
- **UniTable weights** — https://github.com/poloclub/unitable

Place datasets under `data/` and fine-tuned adapters under `TableSight/models/`. Both
directories are gitignored. Without adapters the models fall back to their base behavior,
and SPARTAN runs with no weights at all.

`README.upstream.md` is the original team README.
