"""
Create a slim corpus.parquet for deployment (Render free tier: 512MB RAM).

Drops:
  - bert_binary_prob_* columns (redundant, pred columns are enough)
  - Truncates text to 1000 chars (full text rarely needed in API)

Usage:
    cd backend/
    python slim_corpus.py
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent /"backend" / "data"

def main():
    corpus_path = DATA_DIR / "corpus.parquet"
    df = pd.read_parquet(corpus_path)
    print(f"Original: {len(df)} rows, {len(df.columns)} columns")

    orig_size = corpus_path.stat().st_size / 1e6
    print(f"Original file size: {orig_size:.1f} MB")

    # Estimate memory
    mem = df.memory_usage(deep=True).sum() / 1e6
    print(f"Original memory: {mem:.1f} MB")

    # Drop prob columns (we have pred columns)
    prob_cols = [c for c in df.columns if "bert_binary_prob_" in c]
    if prob_cols:
        df = df.drop(columns=prob_cols)
        print(f"Dropped {len(prob_cols)} prob columns")

    # Truncate text
    df["text"] = df["text"].str[:1000]

    # Downcast float columns
    float_cols = df.select_dtypes(include=["float64"]).columns
    for col in float_cols:
        df[col] = df[col].astype("float32")

    int_cols = df.select_dtypes(include=["int64"]).columns
    for col in int_cols:
        df[col] = df[col].astype("int32")

    mem_after = df.memory_usage(deep=True).sum() / 1e6
    print(f"Slim memory: {mem_after:.1f} MB")
    print(f"Columns: {list(df.columns)}")

    # Save
    slim_path = DATA_DIR / "corpus.parquet"
    df.to_parquet(slim_path, index=False)
    new_size = slim_path.stat().st_size / 1e6
    print(f"\nSaved: {new_size:.1f} MB (was {orig_size:.1f} MB)")


if __name__ == "__main__":
    main()
