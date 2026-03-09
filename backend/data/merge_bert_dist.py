"""
Merge correct BERT distributional predictions into corpus.parquet.

Usage:
    cd backend/
    python merge_bert_dist.py /path/to/bert_dist_all_predictions.csv
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
UO_CODES = [2434, 2436, 2438, 2439, 2441, 2442, 2444, 2445, 2447, 2451]
UO_NAMES = {2434:"HU",2436:"JU",2438:"LU",2439:"ME",2441:"NA",
            2442:"SA",2444:"TE",2445:"VÅ",2447:"ÖV",2451:"VU"}


def main():
    if len(sys.argv) < 2:
        print("Usage: python merge_bert_dist.py <path_to_predictions_csv>")
        return

    csv_path = Path(sys.argv[1])
    corpus_path = "corpus.parquet"

    print(f"Loading predictions: {csv_path}")
    preds = pd.read_csv(csv_path)
    print(f"  {len(preds)} rows, columns: {list(preds.columns)}")

    print(f"Loading corpus: {corpus_path}")
    df = pd.read_parquet(corpus_path)
    print(f"  {len(df)} rows")

    # Check that IDs match
    corpus_ids = df["id"].tolist()
    pred_ids = preds["id"].tolist()
    print(f"  Corpus IDs: {len(corpus_ids)}, Prediction IDs: {len(pred_ids)}")

    # Merge on id
    dist_cols = [c for c in preds.columns if c.startswith("bert_distributional_pct_")]
    print(f"  Distributional columns: {dist_cols}")

    # Drop old distributional columns from corpus
    old_cols = [c for c in df.columns if c.startswith("bert_distributional_pct_")]
    if old_cols:
        print(f"  Dropping old columns: {old_cols}")
        df = df.drop(columns=old_cols)

    # Merge
    merge_cols = ["id"] + dist_cols
    df = df.merge(preds[merge_cols], on="id", how="left")

    # Check for NaN (courses that didn't get predictions)
    missing = df[dist_cols[0]].isna().sum()
    if missing > 0:
        print(f"  ⚠ {missing} courses have no distributional predictions")
    else:
        print(f"  ✓ All courses have distributional predictions")

    # Verify accuracy on single-label val courses
    val = df[df["split"] == "val"]
    gold_cols = [f"y_{c}" for c in UO_CODES]
    single = val[val[gold_cols].sum(axis=1) == 1]

    pred_peaks = single[dist_cols].values.argmax(axis=1)
    gold_peaks = single[gold_cols].values.argmax(axis=1)
    acc = (pred_peaks == gold_peaks).mean()
    print(f"\n  Top-1 accuracy on single-label val: {acc:.1%} ({int(acc*len(single))}/{len(single)})")

    # Save
    df.to_parquet(corpus_path, index=False)
    print(f"\n✓ Saved to {corpus_path}")


if __name__ == "__main__":
    main()
