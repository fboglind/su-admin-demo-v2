# SU Admin Course Classifier — Web Demo

A web application for exploring and interacting with automated classification of Swedish university courses into disciplinary domains (*utbildningsområden*, UO).

Built as a demonstration of applied NLP, semantic search, and full-stack development.

**Models & data:** [Zenodo DOI: 10.5281/zenodo.18256018](https://zenodo.org/records/18256018)

## Features

1. **Classification Tool** — Paste a course description, get predicted UO distribution (TF-IDF live + pre-computed BERT comparison)
2. **Model Comparison** — Side-by-side view of baseline vs BERT predictions across the validation set
3. **Semantic Search** — Natural language search over 9,770 course plans using ChromaDB
4. **Corpus Explorer** — Browse, filter, and inspect the full course corpus

## Architecture

```
React (Vite)  ←→  Flask API  ←→  TF-IDF model (live)
                                  ChromaDB (vector search)
                                  Pre-computed BERT predictions
```

## Quick Start

### 1. Download models & data from Zenodo

```bash
cd scripts/
python download_zenodo.py
```

### 2. Pre-compute BERT predictions & build ChromaDB index

```bash
python precompute.py
```

This runs once (~30-60 min on CPU). Generates:
- `backend/data/corpus.parquet` — full corpus with all predictions
- `backend/data/chroma_db/` — vector index for semantic search

### 3. Start the backend

```bash
cd backend/
pip install -r requirements.txt
python app.py
```

### 4. Start the frontend

```bash
cd frontend/
npm install
npm run dev
```

Open http://localhost:5173

## UO Categories

| Code | Abbr | Name (Swedish) | Name (English) |
|------|------|----------------|----------------|
| 2434 | HU | Humaniora | Humanities |
| 2436 | JU | Juridik | Law |
| 2438 | LU | Lärarutbildning | Teacher Education |
| 2439 | ME | Medicin | Medicine |
| 2441 | NA | Naturvetenskap | Natural Science |
| 2442 | SA | Samhällsvetenskap | Social Sciences |
| 2444 | TE | Teknik | Technology |
| 2445 | VÅ | Vård | Health Care |
| 2447 | ÖV | Övrigt | Other |
| 2451 | VU | Verksamhetsförlagd utb. | Work-based Education |

## Tech Stack

- **Backend:** Flask, scikit-learn, pandas, gunicorn
- **Frontend:** React (Vite), Recharts, Tailwind CSS
- **Models:** KB-BERT (`KB/sentence-bert-swedish-cased`), TF-IDF + Linear SVC
- **Data:** Zenodo archive (DOI: 10.5281/zenodo.18256018)

## Deployment (Render)

The app deploys as a single service on [Render](https://render.com) free tier:

1. Push to GitHub (make sure `backend/data/corpus.parquet` and `backend/models/tfidf_baseline/` are committed)
2. Connect repo on Render → New Web Service
3. Set build command: `./build.sh`
4. Set start command: `cd backend && gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`

Or use the `render.yaml` blueprint for one-click deploy.

### What's included in the deployed app

- **TF-IDF live classification** — runs on CPU, ~instant
- **Pre-computed BERT predictions** — binary + distributional, stored in parquet
- **Corpus browser** — filter by UO, search text, paginated

### What requires additional setup

- **Semantic search** — needs ChromaDB + sentence-transformers (~1GB RAM, paid tier)
- **Live BERT inference** — needs transformers + torch (~2GB RAM)
