# SU Admin Course Classifier — Web Demo

A web application for exploring and interacting with automated classification of Swedish university courses into disciplinary domains (*utbildningsområden*, UO).

Built as a demonstration of applied NLP, and full-stack development.

**Models & data:** [Zenodo DOI: 10.5281/zenodo.18256018](https://zenodo.org/records/18256018)

## Features

1. **Classification Tool** — Paste a course description, get predicted UO distribution (TF-IDF live + pre-computed BERT comparison)
2. **Model Comparison** — Side-by-side view of baseline vs BERT predictions across the validation set
3. ~~**Semantic Search** — Natural language search over 9,770 course plans using ChromaDB~~ [Not yet]
4. **Corpus Explorer** — Browse, filter, and inspect the full course corpus

## Architecture

```
React (Vite) |  Flask API  |  TF-IDF model (live)
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

Generates:
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

## Tech Stack

- **Backend:** Flask, scikit-learn, ChromaDB, sentence-transformers, pandas
- **Frontend:** React (Vite), Recharts, Tailwind CSS
- **Models:** KB-BERT (`KB/sentence-bert-swedish-cased`), TF-IDF + Linear SVC
- **Data:** Zenodo archive (DOI: 10.5281/zenodo.18256018)
