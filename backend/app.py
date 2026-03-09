"""
SU Admin Course Classifier — Flask API

Endpoints:
    POST /api/classify        — Classify new text (TF-IDF live + nearest BERT match)
    GET  /api/course/<id>     — Get single course with all predictions
    POST /api/search          — ChromaDB semantic search
    GET  /api/corpus          — Paginated corpus browser
    GET  /api/stats           — Dataset statistics
    GET  /api/uo              — UO category metadata
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── Configuration ────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"

UO_META = {
    2434: {"abbr": "HU", "name": "Humaniora", "name_en": "Humanities", "color": "#8B5CF6"},
    2436: {"abbr": "JU", "name": "Juridik", "name_en": "Law", "color": "#EC4899"},
    2438: {"abbr": "LU", "name": "Lärarutbildning", "name_en": "Teacher Education", "color": "#F59E0B"},
    2439: {"abbr": "ME", "name": "Medicin", "name_en": "Medicine", "color": "#EF4444"},
    2441: {"abbr": "NA", "name": "Naturvetenskap", "name_en": "Natural Science", "color": "#10B981"},
    2442: {"abbr": "SA", "name": "Samhällsvetenskap", "name_en": "Social Sciences", "color": "#3B82F6"},
    2444: {"abbr": "TE", "name": "Teknik", "name_en": "Technology", "color": "#6366F1"},
    2445: {"abbr": "VÅ", "name": "Vård", "name_en": "Health Care", "color": "#14B8A6"},
    2447: {"abbr": "ÖV", "name": "Övrigt", "name_en": "Other", "color": "#78716C"},
    2451: {"abbr": "VU", "name": "Verksamhetsförlagd utb.", "name_en": "Work-based Education", "color": "#D946EF"},
}

UO_CODES = sorted(UO_META.keys())

# ─── App setup ────────────────────────────────────────────

# In production, serve the built React frontend
STATIC_DIR = Path(__file__).parent / "static"

if STATIC_DIR.exists():
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")
else:
    app = Flask(__name__)

CORS(app)  # Allow React dev server

# ─── Load resources at startup ────────────────────────────

corpus_df = None
tfidf_model = None
chroma_collection = None
label_list = None
sentence_embedder = None
corpus_tfidf_matrix = None      # TF-IDF vectors for corpus similarity search
corpus_vectorizer = None         # Vectorizer for transforming new queries


def load_resources():
    """Load corpus, models, and ChromaDB at startup."""
    global corpus_df, tfidf_model, chroma_collection, label_list, sentence_embedder
    global corpus_tfidf_matrix, corpus_vectorizer

    # 1. Load corpus
    corpus_path = DATA_DIR / "corpus.parquet"
    if corpus_path.exists():
        corpus_df = pd.read_parquet(corpus_path)
        print(f"✓ Corpus loaded: {len(corpus_df)} rows")
    else:
        print(f"⚠ Corpus not found at {corpus_path}")
        print("  Run scripts/precompute.py first")

    # 2. Load TF-IDF model
    tfidf_dir = MODEL_DIR / "tfidf_baseline"
    if tfidf_dir.exists():
        # Show what files are in the directory
        all_files = list(tfidf_dir.rglob("*"))
        print(f"  TF-IDF dir contents ({len(all_files)} files):")
        for f in all_files[:20]:
            print(f"    {f}")

        joblib_files = list(tfidf_dir.rglob("svc_pipe*.joblib"))
        if not joblib_files:
            print(f"⚠ No svc_pipe*.joblib found under {tfidf_dir}")
            any_joblib = list(tfidf_dir.rglob("*.joblib"))
            if any_joblib:
                print(f"  Found other .joblib files: {[str(f) for f in any_joblib]}")
        else:
            try:
                tfidf_model = joblib.load(joblib_files[0])
                print(f"✓ TF-IDF model loaded from {joblib_files[0]}")
            except Exception as e:
                print(f"✗ Failed to load TF-IDF model: {e}")
                print(f"  This usually means a scikit-learn version mismatch.")
                print(f"  Model was trained with sklearn on Kaggle; your version may differ.")

        label_files = list(tfidf_dir.rglob("*label_list*.joblib"))
        if label_files:
            try:
                label_list = joblib.load(label_files[0])
                print(f"✓ Label list loaded: {label_list}")
            except Exception as e:
                print(f"✗ Failed to load label list: {e}")
    else:
        print(f"⚠ TF-IDF directory not found: {tfidf_dir}")
        print(f"  MODEL_DIR is: {MODEL_DIR}")
        print(f"  MODEL_DIR exists: {MODEL_DIR.exists()}")
        if MODEL_DIR.exists():
            print(f"  MODEL_DIR contents: {list(MODEL_DIR.iterdir())}")

    if label_list is None:
        label_list = UO_CODES
        print(f"  Using default label list: {label_list}")

    # 3. Load ChromaDB
    chroma_dir = DATA_DIR / "chroma_db"
    if chroma_dir.exists():
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(chroma_dir))
            chroma_collection = client.get_collection("courses")
            print(f"✓ ChromaDB loaded: {chroma_collection.count()} documents")
        except Exception as e:
            print(f"⚠ ChromaDB not available: {e}")

    # 4. Sentence embedder (loaded lazily on first search)
    print("  Sentence embedder will load on first search request")

    # 5. Build corpus similarity index (lightweight fallback for nearest-match search)
    #    This lets /api/classify work even without ChromaDB or the TF-IDF classifier
    if corpus_df is not None:
        print("  Building corpus similarity index...")
        texts = corpus_df["text"].fillna("").tolist()
        corpus_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=3,
            max_df=0.9,
            max_features=50000,  # cap features for speed
        )
        corpus_tfidf_matrix = corpus_vectorizer.fit_transform(texts)
        print(f"✓ Corpus similarity index built: {corpus_tfidf_matrix.shape}")


# ─── Helpers ──────────────────────────────────────────────

def format_predictions(row, prefix="tfidf_pred"):
    """Extract prediction columns into a dict."""
    preds = {}
    for uo_code in UO_CODES:
        col = f"{prefix}_{uo_code}"
        if col in row.index:
            val = row[col]
            preds[uo_code] = float(val) if not pd.isna(val) else 0.0
    return preds


def format_course(row, idx=None):
    """Format a corpus row as a JSON-serializable dict."""
    result = {
        "idx": int(idx) if idx is not None else None,
        "id": int(row["id"]) if "id" in row and pd.notna(row.get("id")) else None,
        "text": str(row.get("text", ""))[:500],  # Truncate for listing
        "text_full": str(row.get("text", "")),
        "split": str(row.get("split", "")),
        "labels_uo": str(row.get("labels_uo", "")),
        "labels_pct": str(row.get("labels_pct", "")),
    }

    # Add all prediction types if available
    for prefix in ["tfidf_pred", "bert_binary_pred", "bert_binary_prob"]:
        preds = format_predictions(row, prefix)
        if preds:
            result[prefix] = preds

    # Distributional predictions
    dist = {}
    for uo_code in UO_CODES:
        col = f"bert_distributional_pct_{uo_code}"
        if col in row.index and pd.notna(row[col]):
            dist[uo_code] = float(row[col])
    if dist:
        result["bert_dist_pct"] = dist

    return result


def get_sentence_embedder():
    """Lazy-load sentence embedder for ChromaDB queries."""
    global sentence_embedder
    if sentence_embedder is None:
        from sentence_transformers import SentenceTransformer
        print("Loading sentence embedder (first search request)...")
        sentence_embedder = SentenceTransformer("KB/sentence-bert-swedish-cased")
        print("✓ Sentence embedder loaded")
    return sentence_embedder


def find_nearest_in_corpus(text, n=5):
    """Find nearest corpus matches using TF-IDF cosine similarity.
    
    Works without ChromaDB — uses the lightweight corpus index built at startup.
    Returns list of (corpus_index, similarity_score) tuples.
    """
    if corpus_vectorizer is None or corpus_tfidf_matrix is None:
        return []

    query_vec = corpus_vectorizer.transform([text])
    similarities = cosine_similarity(query_vec, corpus_tfidf_matrix).flatten()
    top_indices = similarities.argsort()[::-1][:n]

    return [(int(idx), float(similarities[idx])) for idx in top_indices]


# ─── API Routes ───────────────────────────────────────────

@app.route("/api/debug", methods=["GET"])
def debug_status():
    """Show what resources are loaded — useful for diagnosing issues."""
    return jsonify({
        "corpus_loaded": corpus_df is not None,
        "corpus_rows": len(corpus_df) if corpus_df is not None else 0,
        "corpus_columns": list(corpus_df.columns) if corpus_df is not None else [],
        "tfidf_loaded": tfidf_model is not None,
        "tfidf_type": str(type(tfidf_model)) if tfidf_model else None,
        "label_list": [int(x) for x in label_list] if label_list is not None else None,
        "chroma_loaded": chroma_collection is not None,
        "chroma_count": chroma_collection.count() if chroma_collection is not None else 0,
        "corpus_index_loaded": corpus_tfidf_matrix is not None,
        "corpus_index_shape": list(corpus_tfidf_matrix.shape) if corpus_tfidf_matrix is not None else None,
        "model_dir": str(MODEL_DIR.resolve()),
        "model_dir_exists": MODEL_DIR.exists(),
        "data_dir": str(DATA_DIR.resolve()),
        "data_dir_exists": DATA_DIR.exists(),
    })

@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Return dataset statistics."""
    if corpus_df is None:
        return jsonify({"error": "Corpus not loaded"}), 503

    stats = {
        "total": len(corpus_df),
        "train": int((corpus_df["split"] == "train").sum()),
        "val": int((corpus_df["split"] == "val").sum()),
        "has_tfidf": tfidf_model is not None,
        "has_chroma": chroma_collection is not None,
        "has_bert_binary": any("bert_binary_pred_" in c for c in corpus_df.columns),
        "has_bert_dist": any("bert_distributional_pct_" in c for c in corpus_df.columns),
    }

    # Per-UO label counts (from gold labels)
    for uo_code in UO_CODES:
        col = f"y_{uo_code}"
        if col in corpus_df.columns:
            stats[f"count_{uo_code}"] = int(corpus_df[col].sum())

    return jsonify(stats)


@app.route("/api/uo", methods=["GET"])
def get_uo_meta():
    """Return UO category metadata."""
    return jsonify(UO_META)


@app.route("/api/classify", methods=["POST"])
def classify_text():
    """Classify new text with TF-IDF (live) and find nearest corpus matches."""
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400
    if len(text) < 10:
        return jsonify({"error": "Text too short (min 10 chars)"}), 400

    result = {"text_length": len(text), "models": {}}

    # 1. TF-IDF live prediction (if classifier model is loaded)
    if tfidf_model is not None:
        Y_pred = tfidf_model.predict([text])
        tfidf_labels = {}
        for i, uo_code in enumerate(label_list):
            tfidf_labels[int(uo_code)] = int(Y_pred[0, i])
        result["models"]["tfidf"] = {
            "predictions": tfidf_labels,
            "type": "binary",
            "live": True,
        }

    # 2. Find nearest corpus matches (ChromaDB → TF-IDF fallback)
    nearest = []
    search_method = None

    # 2a. Try ChromaDB first (semantic search)
    if chroma_collection is not None:
        try:
            embedder = get_sentence_embedder()
            query_embedding = embedder.encode([text], normalize_embeddings=True)
            chroma_results = chroma_collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=5,
            )
            for i, (doc_id, distance, doc, meta) in enumerate(zip(
                chroma_results["ids"][0],
                chroma_results["distances"][0],
                chroma_results["documents"][0],
                chroma_results["metadatas"][0],
            )):
                corpus_idx = int(meta.get("corpus_idx", doc_id))
                if corpus_df is not None and corpus_idx < len(corpus_df):
                    course = format_course(corpus_df.iloc[corpus_idx], corpus_idx)
                else:
                    course = {"text": doc[:300]}
                nearest.append({
                    "rank": i + 1,
                    "similarity": round(1 - distance, 4),
                    "course": course,
                })
            search_method = "semantic (ChromaDB)"
        except Exception as e:
            result["search_error"] = str(e)

    # 2b. Fallback: TF-IDF corpus similarity search
    if not nearest and corpus_df is not None:
        matches = find_nearest_in_corpus(text, n=5)
        for rank, (corpus_idx, sim) in enumerate(matches, 1):
            course = format_course(corpus_df.iloc[corpus_idx], corpus_idx)
            nearest.append({
                "rank": rank,
                "similarity": round(sim, 4),
                "course": course,
            })
        search_method = "text similarity (TF-IDF cosine)"

    if nearest:
        result["nearest_matches"] = nearest
        result["search_method"] = search_method

        # 3. Extract pre-computed predictions from the top match
        top_idx = nearest[0]["course"].get("idx")
        if top_idx is not None and corpus_df is not None and top_idx < len(corpus_df):
            top_row = corpus_df.iloc[top_idx]
            top_sim = nearest[0]["similarity"]

            # TF-IDF predictions from corpus (if live model not available)
            if "tfidf" not in result["models"]:
                preds = format_predictions(top_row, "tfidf_pred")
                if preds:
                    result["models"]["tfidf"] = {
                        "predictions": preds,
                        "type": "binary",
                        "live": False,
                        "source": f"nearest match ({search_method}, sim: {top_sim:.3f})",
                    }

            # BERT binary predictions from corpus
            preds = format_predictions(top_row, "bert_binary_pred")
            if preds:
                result["models"]["bert_binary"] = {
                    "predictions": preds,
                    "type": "binary",
                    "live": False,
                    "source": f"nearest match ({search_method}, sim: {top_sim:.3f})",
                }

            # BERT distributional predictions from corpus
            dist = {}
            for uo_code in UO_CODES:
                col = f"bert_distributional_pct_{uo_code}"
                if col in top_row.index and pd.notna(top_row[col]):
                    dist[uo_code] = float(top_row[col])
            if dist:
                result["models"]["bert_dist"] = {
                    "predictions": dist,
                    "type": "distributional",
                    "live": False,
                    "source": f"nearest match ({search_method}, sim: {top_sim:.3f})",
                }

    return jsonify(result)


@app.route("/api/search", methods=["POST"])
def semantic_search():
    """Semantic search over course corpus."""
    data = request.get_json()
    query = data.get("query", "").strip()
    n_results = min(data.get("n", 10), 50)

    if not query:
        return jsonify({"error": "No query provided"}), 400

    if chroma_collection is None:
        return jsonify({"error": "Semantic search not available (ChromaDB not loaded)"}), 503

    try:
        embedder = get_sentence_embedder()
        query_embedding = embedder.encode([query], normalize_embeddings=True)

        results = chroma_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
        )

        courses = []
        for i, (doc_id, distance, doc, meta) in enumerate(zip(
            results["ids"][0],
            results["distances"][0],
            results["documents"][0],
            results["metadatas"][0],
        )):
            corpus_idx = int(meta.get("corpus_idx", doc_id))
            if corpus_df is not None and corpus_idx < len(corpus_df):
                course = format_course(corpus_df.iloc[corpus_idx], corpus_idx)
            else:
                course = {"text": doc[:500]}

            course["similarity"] = round(1 - distance, 4)
            course["rank"] = i + 1
            courses.append(course)

        return jsonify({"query": query, "results": courses, "total": len(courses)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/corpus", methods=["GET"])
def browse_corpus():
    """Paginated corpus browser with filters."""
    if corpus_df is None:
        return jsonify({"error": "Corpus not loaded"}), 503

    page = request.args.get("page", 1, type=int)
    per_page = min(request.args.get("per_page", 20, type=int), 100)
    split_filter = request.args.get("split", None)
    uo_filter = request.args.get("uo", None, type=int)
    search_text = request.args.get("q", None)

    df = corpus_df.copy()

    # Apply filters
    if split_filter:
        df = df[df["split"] == split_filter]
    if uo_filter and f"y_{uo_filter}" in df.columns:
        df = df[df[f"y_{uo_filter}"] == 1]
    if search_text:
        mask = df["text"].str.contains(search_text, case=False, na=False)
        df = df[mask]

    total = len(df)
    start = (page - 1) * per_page
    end = start + per_page
    page_df = df.iloc[start:end]

    courses = [
        format_course(row, idx)
        for idx, (_, row) in zip(range(start, end), page_df.iterrows())
    ]

    return jsonify({
        "courses": courses,
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page,
    })


@app.route("/api/course/<int:idx>", methods=["GET"])
def get_course(idx):
    """Get a single course with full details and all predictions."""
    if corpus_df is None:
        return jsonify({"error": "Corpus not loaded"}), 503

    if idx < 0 or idx >= len(corpus_df):
        return jsonify({"error": "Course not found"}), 404

    row = corpus_df.iloc[idx]
    course = format_course(row, idx)
    course["text"] = course["text_full"]  # Return full text for detail view

    return jsonify(course)


@app.route("/api/compare", methods=["GET"])
def compare_models():
    """Get model comparison data for validation set."""
    if corpus_df is None:
        return jsonify({"error": "Corpus not loaded"}), 503

    # Filter to validation set only
    val_df = corpus_df[corpus_df["split"] == "val"].copy()

    page = request.args.get("page", 1, type=int)
    per_page = min(request.args.get("per_page", 20, type=int), 100)
    disagreements_only = request.args.get("disagreements", "false") == "true"
    uo_filter = request.args.get("uo", None, type=int)

    if uo_filter and f"y_{uo_filter}" in val_df.columns:
        val_df = val_df[val_df[f"y_{uo_filter}"] == 1]

    if disagreements_only:
        # Find rows where TF-IDF and BERT disagree
        mask = pd.Series(False, index=val_df.index)
        for uo_code in UO_CODES:
            tfidf_col = f"tfidf_pred_{uo_code}"
            bert_col = f"bert_binary_pred_{uo_code}"
            if tfidf_col in val_df.columns and bert_col in val_df.columns:
                mask = mask | (val_df[tfidf_col] != val_df[bert_col])
        val_df = val_df[mask]

    total = len(val_df)
    start = (page - 1) * per_page
    end = start + per_page
    page_df = val_df.iloc[start:end]

    courses = []
    for _, row in page_df.iterrows():
        courses.append(format_course(row))

    return jsonify({
        "courses": courses,
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page,
    })


# ─── Frontend catch-all (must be AFTER all /api routes) ───

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    """Serve React frontend — lets React Router handle client-side routes."""
    if path.startswith("api/"):
        return jsonify({"error": "Not found"}), 404
    if STATIC_DIR.exists():
        file_path = STATIC_DIR / path
        if file_path.is_file():
            return app.send_static_file(path)
        return app.send_static_file("index.html")
    return "Frontend not built. Run: cd frontend && npm run build", 404


# ─── Main ─────────────────────────────────────────────────

import os
load_resources()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
