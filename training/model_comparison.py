"""
Compare all trained models on the TEST set.

This script:
  - Loads all CSVs from artifacts/test_summaries/
  - Aggregates metrics per model (mean test_acc, test_f1, test_precision)
  - Saves an aggregated CSV
  - Saves a simple bar plot of mean test accuracy

Run with:
    python -m training.compare_models
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from training.preprocess_data import ARTIFACTS

# Directories
TEST_FS_DIR = ARTIFACTS / "test_summaries"
REPORTS_DIR = ARTIFACTS / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def compare_models(save_plot: bool = True) -> pd.DataFrame | None:
    """
    Load all test summary CSVs and produce an aggregated comparison.

    Returns:
        summary_test DataFrame, or None if no test summaries found.
    """
    df_test = pd.DataFrame()

    csvs = sorted(TEST_FS_DIR.glob("*.csv"))
    if csvs:
        df_test = pd.concat((pd.read_csv(p) for p in csvs), ignore_index=True)

    if df_test.empty:
        print(f"No test summaries found to compare in {TEST_FS_DIR}.")
        return None

    # ---- Aggregation across models on TEST set ----
    agg_kwargs = dict(
        mean_test_acc=("test_acc", "mean"),
        mean_test_f1=("test_f1", "mean"),
        mean_test_precision=("test_precision", "mean"),
    )

    # Optionally track number of folds/runs if column exists
    if "fold" in df_test.columns:
        agg_kwargs["folds"] = ("fold", "nunique")

    summary_test = (
        df_test
        .groupby("model", as_index=False)
        .agg(**agg_kwargs)
        .sort_values("mean_test_acc", ascending=False)
    )

    print("Model comparison on TEST set (aggregated):")
    print(summary_test.to_string(index=False))

    return summary_test


if __name__ == "__main__":
   summary_test =  compare_models()

   summary_out = REPORTS_DIR / "test_summary_aggregated.csv"
   summary_test.to_csv(summary_out, index=False)
   print(f"[saved] aggregated TEST summary → {summary_out}")
