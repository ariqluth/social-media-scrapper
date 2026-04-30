# Fake News Detection Module

Companion module to `social-media-scrapper/`. Implements the ML/DL pipeline for the research project:

> **Systematic Review of Fake News, Propaganda, and Disinformation: Examining Authors, Content, and Social Impact**
> — using machine learning + deep learning for social-media detection.

## Research Framework

Grounded in two references:

1. **Plikynas, D., Rizgeliene, I., & Korvel, G. (2025).** *Systematic Review of Fake News, Propaganda, and Disinformation: Examining Authors, Content, and Social Impact Through Machine Learning.* IEEE Access, 13, 17583–17629. [DOI: 10.1109/ACCESS.2025.3530688](https://doi.org/10.1109/ACCESS.2025.3530688) — PRISMA-based review; basis for the authors/content/social-impact framing.
2. **Devarajan, G. G., et al. (2023).** *AI-Assisted Deep NLP-Based Approach for Prediction of Fake News From Social Media Users.* IEEE Access. [Document 10086954](https://ieeexplore.ieee.org/document/10086954) — baseline deep NLP pipeline (reported F1 = 98.33%).
3. **JOAIR reference paper** (jouair.com Vol./No. 7) — local methodology reference for the Indonesian-language context.

## Pipeline Phases

### Phase 1 — Model Implementation (`notebook/fake_news_detection.ipynb`)
- **Classical ML baselines:** TF-IDF → Logistic Regression, SVM, Random Forest, XGBoost.
- **Deep learning / transformers (Hugging Face):**
  - `mdhugol/indonesia-bert-sentiment-classification` — Indonesian BERT baseline
  - `indolem/indobert-base-uncased` — Indonesian BERT for fine-tuning
  - `Pulk17/Fake-News-Detection-dataset` models
  - `jy46604790/Fake-News-Bert-Detect` — English baseline
  - `hamzab/roberta-fake-news-classification` — RoBERTa baseline

### Phase 2 — Evaluation (`src/metrics.py`)
Standard metrics computed per model:
- Accuracy, Precision, Recall, F1-score (macro + weighted)
- Confusion matrix
- ROC-AUC (binary) / PR curves
- Per-class breakdown for imbalanced data

### Phase 3 — Cross-Reference Against Credible Sources (`src/cross_reference.py`)
Verified Indonesian news + official sources:
- **komdigi.go.id** — Ministry of Communication & Digital Affairs (official hoax-busting)
- **tribratanews.jabar.polri.go.id** — West Java Police hoax bulletin
- **cnbcindonesia.com**
- **cnnindonesia.com**
- **kompas.com**
- **tempo.co**

Method: TF-IDF / sentence-embedding similarity between the input text and scraped article corpora per source.

### Phase 4 — User-Facing App (`app/streamlit_app.py`)
Streamlit UI that lets a user:
- Paste a **text caption** OR upload an **image** (OCR extracts text via EasyOCR)
- Run the selected model(s) → get reliability score + label (Real / Fake / Uncertain)
- See per-metric breakdown (accuracy/precision/recall/F1 from the model's validation set)
- See the top-N cross-reference matches from credible sources with similarity scores
- Export a PDF / CSV report per submission

## Installation

```bash
cd social-media-scrapper
.venv\Scripts\activate                      # reuse the scraper's venv
pip install -r fake_news_detection\requirements-ml.txt
```

The first run of the notebook / app will download model weights from Hugging Face (~1-2 GB total, cached locally).

## Usage

### Run the notebook
```bash
jupyter lab fake_news_detection\notebook\fake_news_detection.ipynb
```

### Run the Streamlit app
```bash
streamlit run fake_news_detection\app\streamlit_app.py
```

### Analyse scraper output directly
Drop a CSV produced by `main.py` (any platform) into `fake_news_detection/data/` and point the notebook's `INPUT_CSV` variable at it — the `caption` column is analysed row-by-row.

## Data Flow

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│ social-media-    │     │ fake_news_       │     │ Verified Sources     │
│ scrapper         │────▶│ detection        │────▶│ komdigi / kompas /   │
│ (captions CSV)   │     │ (classifier)     │◀────│ tempo / cnn / ...    │
└──────────────────┘     └────────┬─────────┘     └──────────────────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Streamlit App    │
                         │ user uploads     │
                         │ text or image    │
                         └──────────────────┘
```

## Module Layout

```
fake_news_detection/
├── notebook/fake_news_detection.ipynb   # end-to-end pipeline
├── src/
│   ├── preprocessing.py                 # text normalisation (ID + EN)
│   ├── models.py                        # HF + classical ML wrappers
│   ├── metrics.py                       # evaluation metrics
│   ├── cross_reference.py               # verify against credible sources
│   └── ocr.py                           # image → text extraction
├── app/streamlit_app.py                 # user-facing UI
├── data/                                # drop CSVs / datasets here
└── requirements-ml.txt
```

## License

MIT — inherits from the parent project.
