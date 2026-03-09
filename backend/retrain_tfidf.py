"""
Retrain TF-IDF + Linear SVC pipeline from corpus.parquet.

Use this if the Kaggle-trained joblib won't load due to sklearn version mismatch.
Trains the exact same pipeline as the original baseline notebook.

Usage:
    cd backend/
    python retrain_tfidf.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
import joblib

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models" / "tfidf_baseline"

UO_CODES = [2434, 2436, 2438, 2439, 2441, 2442, 2444, 2445, 2447, 2451]

def main():
    # Load corpus
    corpus_path = DATA_DIR / "corpus.parquet"
    if not corpus_path.exists():
        print(f"✗ Corpus not found at {corpus_path}")
        return

    df = pd.read_parquet(corpus_path)
    print(f"Loaded corpus: {len(df)} rows")

    # Split
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # Build label matrix
    texts_train = train_df["text"].fillna("").tolist()
    texts_val = val_df["text"].fillna("").tolist()

    Y_train = np.column_stack([train_df[f"y_{uo}"].values for uo in UO_CODES])
    Y_val = np.column_stack([val_df[f"y_{uo}"].values for uo in UO_CODES])
    print(f"Label matrix: {Y_train.shape}")

    # Train — same hyperparameters as original notebook
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=3,
            max_df=0.9,
        )),
        ("clf", OneVsRestClassifier(LinearSVC())),
    ])

    print("Training...")
    pipe.fit(texts_train, Y_train)

    # Quick eval
    Y_pred = pipe.predict(texts_val)
    subset_acc = np.mean(np.all(Y_pred == Y_val, axis=1))
    print(f"Subset accuracy on val: {subset_acc:.4f}")

    # Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODEL_DIR / "svc_pipe.joblib"
    joblib.dump(pipe, out_path)
    print(f"✓ Saved to {out_path}")

    # Also save label list
    label_path = MODEL_DIR / "uo_label_list.joblib"
    joblib.dump(UO_CODES, label_path)
    print(f"✓ Saved label list to {label_path}")


if __name__ == "__main__":
    main()
