"""
Pre-compute BERT predictions and build ChromaDB vector index.
Run once after downloading models from Zenodo.

Usage:
    python precompute.py [--skip-bert] [--skip-chroma]

Generates:
    backend/data/corpus.parquet       — corpus with all model predictions
    backend/data/chroma_db/           — ChromaDB vector store
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent
BACKEND_DIR = SCRIPT_DIR / ".." / "backend"
DATA_DIR = BACKEND_DIR / "data"
MODEL_DIR = BACKEND_DIR / "models"

# UO code → human-readable mapping
UO_NAMES = {
    2434: "HU",  2436: "JU",  2438: "LU",  2439: "ME",  2441: "NA",
    2442: "SA",  2444: "TE",  2445: "VÅ",  2447: "ÖV",  2451: "VU",
}

UO_FULL_NAMES = {
    2434: "Humaniora",       2436: "Juridik",
    2438: "Lärarutbildning", 2439: "Medicin",
    2441: "Naturvetenskap",  2442: "Samhällsvetenskap",
    2444: "Teknik",          2445: "Vård",
    2447: "Övrigt",          2451: "Verksamhetsförlagd utbildning",
}


def load_corpus():
    """Load preprocessed corpus with train/val splits."""
    preproc_dir = DATA_DIR / "preprocessed"

    # Try to find the CSV files in the preprocessed directory
    train_path = None
    val_path = None
    for f in preproc_dir.rglob("train_ml_export.csv"):
        train_path = f
    for f in preproc_dir.rglob("val_ml_export.csv"):
        val_path = f

    if train_path is None or val_path is None:
        # List what's actually there
        print(f"Contents of {preproc_dir}:")
        for f in sorted(preproc_dir.rglob("*")):
            print(f"  {f}")
        raise FileNotFoundError("Could not find train/val export CSVs in preprocessed data")

    print(f"Loading train: {train_path}")
    print(f"Loading val:   {val_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    train_df["split"] = "train"
    val_df["split"] = "val"

    corpus = pd.concat([train_df, val_df], ignore_index=True)
    print(f"Corpus loaded: {len(corpus)} rows ({len(train_df)} train + {len(val_df)} val)")
    return corpus


def add_tfidf_predictions(corpus):
    """Run TF-IDF baseline on all texts and add predictions."""
    tfidf_dir = MODEL_DIR / "tfidf_baseline"

    # Find the pipeline file
    pipe_path = None
    for f in tfidf_dir.rglob("svc_pipe.joblib"):
        pipe_path = f
    if pipe_path is None:
        for f in tfidf_dir.rglob("*.joblib"):
            if "svc" in f.name.lower():
                pipe_path = f

    if pipe_path is None:
        print("⚠  TF-IDF model not found, skipping baseline predictions")
        return corpus

    # Also find the label list
    label_path = None
    for f in tfidf_dir.rglob("uo_label_list.joblib"):
        label_path = f

    print(f"Loading TF-IDF pipeline: {pipe_path}")
    svc_pipe = joblib.load(pipe_path)

    label_list = None
    if label_path:
        label_list = joblib.load(label_path)
        print(f"Label order: {label_list}")
    else:
        label_list = sorted(UO_NAMES.keys())
        print(f"Using default label order: {label_list}")

    # Run predictions
    texts = corpus["text"].fillna("").tolist()
    print(f"Running TF-IDF predictions on {len(texts)} texts...")
    Y_pred = svc_pipe.predict(texts)

    # Store binary predictions as columns
    for i, uo_code in enumerate(label_list):
        corpus[f"tfidf_pred_{uo_code}"] = Y_pred[:, i]

    print(f"✓ TF-IDF predictions added ({Y_pred.sum()} total positive labels)")
    return corpus


def add_bert_predictions(corpus, model_type="binary"):
    """Load BERT model and run predictions on corpus."""
    if model_type == "binary":
        model_dir = MODEL_DIR / "bert_binary"
    else:
        model_dir = MODEL_DIR / "bert_distributional"

    if not model_dir.exists():
        print(f"⚠  BERT {model_type} model not found at {model_dir}, skipping")
        return corpus

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
    except ImportError:
        print("⚠  transformers/torch not installed, skipping BERT predictions")
        print("   Install with: pip install transformers torch")
        return corpus

    # Find the actual model directory (might be nested)
    model_path = model_dir
    for f in model_dir.rglob("config.json"):
        model_path = f.parent
        break

    print(f"Loading BERT {model_type} model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    label_list = sorted(UO_NAMES.keys())
    texts = corpus["text"].fillna("").tolist()

    print(f"Running BERT {model_type} inference on {len(texts)} texts (this may take a while on CPU)...")

    batch_size = 32
    all_outputs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

            if model_type == "binary":
                probs = torch.sigmoid(logits).numpy()
            else:
                probs = torch.softmax(logits, dim=-1).numpy()

        all_outputs.append(probs)

        if (i // batch_size) % 10 == 0:
            print(f"  Batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")

    all_probs = np.vstack(all_outputs)

    # Store predictions
    prefix = f"bert_{model_type}"
    for i, uo_code in enumerate(label_list):
        if model_type == "binary":
            corpus[f"{prefix}_prob_{uo_code}"] = all_probs[:, i]
            corpus[f"{prefix}_pred_{uo_code}"] = (all_probs[:, i] >= 0.5).astype(int)
        else:
            corpus[f"{prefix}_pct_{uo_code}"] = np.round(all_probs[:, i] * 100, 2)

    print(f"✓ BERT {model_type} predictions added")
    return corpus


def build_chromadb(corpus):
    """Build ChromaDB vector index from corpus texts."""
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("⚠  chromadb or sentence-transformers not installed, skipping")
        print("   Install with: pip install chromadb sentence-transformers")
        return

    chroma_dir = DATA_DIR / "chroma_db"
    if chroma_dir.exists() and any(chroma_dir.iterdir()):
        print(f"ChromaDB already exists at {chroma_dir}, skipping")
        return

    print("Loading sentence embedding model (KB/sentence-bert-swedish-cased)...")
    embedder = SentenceTransformer("KB/sentence-bert-swedish-cased")

    texts = corpus["text"].fillna("").tolist()
    ids = [str(i) for i in range(len(texts))]

    print(f"Encoding {len(texts)} texts (this will take a while on CPU)...")
    embeddings = embedder.encode(
        texts,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,
    )

    print("Building ChromaDB index...")
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    collection = client.get_or_create_collection(
        name="courses",
        metadata={"hnsw:space": "cosine"},
    )

    # Add in batches (ChromaDB has limits)
    batch_size = 500
    for i in range(0, len(texts), batch_size):
        end = min(i + batch_size, len(texts))

        # Prepare metadata for each document
        metadatas = []
        for idx in range(i, end):
            row = corpus.iloc[idx]
            meta = {
                "corpus_idx": idx,
                "split": str(row.get("split", "")),
            }
            # Add the course ID if available
            if "id" in row:
                meta["course_id"] = str(row["id"])
            metadatas.append(meta)

        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end].tolist(),
            documents=texts[i:end],
            metadatas=metadatas,
        )

        if (i // batch_size) % 10 == 0:
            print(f"  Added {end}/{len(texts)} documents")

    print(f"✓ ChromaDB built with {collection.count()} documents at {chroma_dir}")


def main():
    skip_bert = "--skip-bert" in sys.argv
    skip_chroma = "--skip-chroma" in sys.argv

    print(f"\n{'='*60}")
    print("SU Admin Demo — Pre-computation")
    print(f"{'='*60}\n")

    # Step 1: Load corpus
    print("─── Step 1: Load corpus ───")
    corpus = load_corpus()

    # Step 2: TF-IDF predictions
    print("\n─── Step 2: TF-IDF predictions ───")
    corpus = add_tfidf_predictions(corpus)

    # Step 3: BERT predictions
    if not skip_bert:
        print("\n─── Step 3a: BERT binary predictions ───")
        corpus = add_bert_predictions(corpus, model_type="binary")
        print("\n─── Step 3b: BERT distributional predictions ───")
        corpus = add_bert_predictions(corpus, model_type="distributional")
    else:
        print("\n─── Step 3: Skipping BERT predictions (--skip-bert) ───")

    # Step 4: Save corpus
    print("\n─── Step 4: Save enriched corpus ───")
    output_path = DATA_DIR / "corpus.parquet"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(output_path, index=False)
    print(f"✓ Saved to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")

    # Also save as JSON for frontend quick-load of stats
    stats = {
        "total_courses": len(corpus),
        "train_count": int((corpus["split"] == "train").sum()),
        "val_count": int((corpus["split"] == "val").sum()),
        "columns": list(corpus.columns),
        "has_tfidf": any("tfidf_pred_" in c for c in corpus.columns),
        "has_bert_binary": any("bert_binary_pred_" in c for c in corpus.columns),
        "has_bert_dist": any("bert_distributional_pct_" in c for c in corpus.columns),
    }
    stats_path = DATA_DIR / "corpus_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Stats saved to {stats_path}")

    # Step 5: ChromaDB
    if not skip_chroma:
        print("\n─── Step 5: Build ChromaDB index ───")
        build_chromadb(corpus)
    else:
        print("\n─── Step 5: Skipping ChromaDB (--skip-chroma) ───")

    print(f"\n{'='*60}")
    print("Pre-computation complete!")
    print(f"\nGenerated files:")
    print(f"  {output_path}")
    print(f"  {stats_path}")
    if not skip_chroma:
        print(f"  {DATA_DIR / 'chroma_db/'}")
    print(f"\nNext step: cd ../backend && python app.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
