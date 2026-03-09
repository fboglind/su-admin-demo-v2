"""
Fix BERT prediction columns in corpus.parquet.

The precompute step assumed BERT neurons map to UO codes in sorted order:
  neuron 0 → 2434(HU), neuron 1 → 2436(JU), ...

But the model was trained with a label order that may differ.
Since config.json only has generic LABEL_0..LABEL_9, this script
derives the correct mapping empirically by correlating each prediction
column with each gold label column.

Usage:
    cd backend/
    python fix_bert_labels.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import linear_sum_assignment

DATA_DIR = Path(__file__).parent / "data"

UO_CODES = [2434, 2436, 2438, 2439, 2441, 2442, 2444, 2445, 2447, 2451]
UO_NAMES = {2434:"HU", 2436:"JU", 2438:"LU", 2439:"ME", 2441:"NA",
            2442:"SA", 2444:"TE", 2445:"VÅ", 2447:"ÖV", 2451:"VU"}


def find_correct_mapping(df, prefix):
    """
    Derive the correct neuron → UO mapping by correlating predictions with gold labels.
    
    Currently column '{prefix}_{sorted_code[i]}' holds neuron i's output.
    We correlate each neuron's output with each gold label to find what
    neuron i actually predicts.
    
    Returns: list of 10 UO codes in the model's actual output order,
             i.e. correct_order[i] = the UO code that neuron i predicts.
    """
    sorted_codes = sorted(UO_CODES)
    
    # Build correlation matrix: corr[neuron_i, gold_j]
    # neuron_i's values are in column '{prefix}_{sorted_codes[i]}'
    # gold_j's values are in column 'y_{UO_CODES[j]}'
    n = len(sorted_codes)
    corr = np.zeros((n, n))
    
    for i, assumed_code in enumerate(sorted_codes):
        pred_col = f"{prefix}_{assumed_code}"
        if pred_col not in df.columns:
            continue
        pred_vals = df[pred_col].values.astype(float)
        
        for j, gold_code in enumerate(sorted_codes):
            gold_col = f"y_{gold_code}"
            if gold_col not in df.columns:
                continue
            gold_vals = df[gold_col].values.astype(float)
            corr[i, j] = np.corrcoef(pred_vals, gold_vals)[0, 1]
    
    print(f"\n  Correlation matrix (neuron × gold label):")
    header = "         " + "  ".join(f"{UO_NAMES[c]:>5}" for c in sorted_codes)
    print(header)
    for i, code in enumerate(sorted_codes):
        row = "  ".join(f"{corr[i,j]:5.2f}" for j in range(n))
        peak = sorted_codes[np.argmax(corr[i, :])]
        print(f"  n{i} ({UO_NAMES[code]:>2}): {row}  -> {UO_NAMES[peak]}")
    
    # Use Hungarian algorithm to find optimal assignment
    # (maximise total correlation = minimise negative correlation)
    row_ind, col_ind = linear_sum_assignment(-corr)
    
    correct_order = [None] * n
    for neuron_idx, gold_idx in zip(row_ind, col_ind):
        correct_order[neuron_idx] = sorted_codes[gold_idx]
    
    return correct_order


def remap_columns(df, prefix, correct_order):
    """Remap prediction columns to correct UO codes."""
    sorted_codes = sorted(UO_CODES)
    
    if correct_order == sorted_codes:
        print(f"  {prefix}: already correct, skipping")
        return df
    
    print(f"\n  Remapping {prefix}:")
    print(f"    Assumed: {[UO_NAMES[c] for c in sorted_codes]}")
    print(f"    Correct: {[UO_NAMES[c] for c in correct_order]}")
    
    # Read raw neuron values (currently stored under assumed codes)
    neuron_values = {}
    for i, assumed_code in enumerate(sorted_codes):
        col = f"{prefix}_{assumed_code}"
        if col in df.columns:
            neuron_values[i] = df[col].values.copy()
    
    # Write back under correct codes
    for i, correct_code in enumerate(correct_order):
        col = f"{prefix}_{correct_code}"
        if i in neuron_values:
            df[col] = neuron_values[i]
    
    return df


def check_accuracy(df, prefix):
    """Check top-1 accuracy on single-label val courses."""
    val = df[df["split"] == "val"]
    gold_cols = [f"y_{c}" for c in UO_CODES]
    single = val[val[gold_cols].sum(axis=1) == 1]
    
    pred_cols = [f"{prefix}_{c}" for c in UO_CODES]
    existing = [c for c in pred_cols if c in df.columns]
    if not existing:
        return 0.0, 0
    
    pred_peaks = single[pred_cols].values.argmax(axis=1)
    gold_peaks = single[gold_cols].values.argmax(axis=1)
    
    correct = (pred_peaks == gold_peaks).sum()
    total = len(single)
    return correct / total if total > 0 else 0.0, total


def main():
    corpus_path = DATA_DIR / "corpus.parquet"
    df = pd.read_parquet(corpus_path)
    print(f"Loaded corpus: {len(df)} rows")
    
    # Check BEFORE
    print("\n=== BEFORE FIX ===")
    for prefix in ["bert_binary_pred", "bert_binary_prob", "bert_distributional_pct"]:
        acc, n = check_accuracy(df, prefix)
        if n > 0:
            print(f"  {prefix}: top-1 accuracy = {acc:.1%} ({int(acc*n)}/{n})")
    
    # Find and apply correct mapping for each BERT model
    for prefix_group, prefixes in [
        ("bert_distributional_pct", ["bert_distributional_pct"]),
        ("bert_binary_prob", ["bert_binary_prob", "bert_binary_pred"]),
    ]:
        pred_cols = [f"{prefix_group}_{c}" for c in UO_CODES]
        if not any(c in df.columns for c in pred_cols):
            print(f"\n  {prefix_group}: no columns found, skipping")
            continue
            
        print(f"\n--- Deriving mapping for {prefix_group} ---")
        correct_order = find_correct_mapping(df, prefix_group)
        print(f"\n  Correct label order: {[UO_NAMES[c] for c in correct_order]}")
        
        for prefix in prefixes:
            df = remap_columns(df, prefix, correct_order)
    
    # Check AFTER
    print("\n=== AFTER FIX ===")
    for prefix in ["bert_binary_pred", "bert_binary_prob", "bert_distributional_pct"]:
        acc, n = check_accuracy(df, prefix)
        if n > 0:
            print(f"  {prefix}: top-1 accuracy = {acc:.1%} ({int(acc*n)}/{n})")
    
    # Save
    backup_path = DATA_DIR / "corpus_before_fix.parquet"
    print(f"\n  Backing up original to {backup_path}")
    pd.read_parquet(corpus_path).to_parquet(backup_path, index=False)
    
    df.to_parquet(corpus_path, index=False)
    print(f"✓ Fixed parquet saved to {corpus_path}")


if __name__ == "__main__":
    main()
