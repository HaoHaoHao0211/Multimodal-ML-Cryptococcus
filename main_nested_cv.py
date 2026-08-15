#!/usr/bin/env python3
"""
Multimodal nested CV + SHAP analysis  --  main entry point (parallel).

- Models: Random Forest and XGBoost only.
- Modality combinations: 4 single + 6 pairwise = 10 total, run in parallel.
- Per combination:
    Phase 1: 5x5 nested CV metrics (mean +/- std)
    Phase 2: Final model + feature importance + SHAP


Environment variables:
    N_PARALLEL  Max parallel workers (default: 10).
"""

import os
import sys
import time
import traceback
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    ALL_MODALITIES, MODALITY_PREFIXES, RESULTS_DIR, RANDOM_STATE,
)
from utils import seed_everything, load_multimodal_data, build_X
from engine import run_rf_nested_cv_shap, run_xgb_nested_cv_shap


def _run_one_combo(combo, data_dict, y, prefixes, out_root):
    """
    Run RF + XGBoost for a single modality combination (independent process).

    Returns a list of summary dicts with per-metric mean/std.
    """
    combo_name = "+".join(combo)
    t_start = time.time()

    X = build_X(data_dict, list(combo), prefixes=prefixes)
    feature_names = X.columns.tolist()

    rf_dir = os.path.join(out_root, combo_name, "RF")
    rf_metrics, _ = run_rf_nested_cv_shap(X, y, feature_names, rf_dir)

    xgb_dir = os.path.join(out_root, combo_name, "XGB")
    xgb_metrics, _ = run_xgb_nested_cv_shap(X, y, feature_names, xgb_dir)

    rows = []
    for model_name, df_m in [("RF", rf_metrics), ("XGBoost", xgb_metrics)]:
        for col in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
            vals = df_m[col].values
            rows.append({
                "Combination":  combo_name,
                "N_Modalities": len(combo),
                "Model":        model_name,
                "Metric":       col,
                "Mean":         float(np.mean(vals)),
                "Std":          float(np.std(vals)),
            })

    elapsed = time.time() - t_start
    print(f"[DONE] {combo_name}  ({elapsed:.0f}s)")
    return rows


def main():
    seed_everything(RANDOM_STATE)

    data_dict, y = load_multimodal_data()

    out_root = os.path.join(RESULTS_DIR, "multimodal_nested_cv")
    os.makedirs(out_root, exist_ok=True)

    # Generate all combinations: 4 single + 6 pairwise = 10
    all_combos = []
    for r in range(1, 3):
        for combo in combinations(ALL_MODALITIES, r):
            all_combos.append(combo)

    max_workers = int(os.environ.get("N_PARALLEL", 10))
    print(f"\nModality combinations: {' + '.join(['+'.join(c) for c in all_combos])}")
    print(f"Parallel workers: {max_workers}")

    # ---- Parallel execution ----
    summary_rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for combo in all_combos:
            combo_name = "+".join(combo)
            print(f"[QUEUE] {combo_name}")
            future = executor.submit(
                _run_one_combo, combo, data_dict, y, MODALITY_PREFIXES, out_root,
            )
            futures[future] = combo_name

        for future in as_completed(futures):
            combo_name = futures[future]
            try:
                rows = future.result()
                summary_rows.extend(rows)
            except Exception:
                print(f"[FAIL] {combo_name}")
                traceback.print_exc()

    total_elapsed = time.time() - t0

    # ---- Summary table ----
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_root, "Multimodal_NestedCV_Summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary table: {summary_path}")

    print("\n" + "=" * 80)
    print("All combinations summary (ROC-AUC mean +/- std)")
    print("=" * 80)
    roc = summary_df[summary_df["Metric"] == "roc_auc"]
    for _, row in roc.iterrows():
        print(f"  {row['Combination']:20s} | {row['Model']:8s} | "
              f"{row['Mean']:.4f} +/- {row['Std']:.4f}")

    print(f"\nAll done! Elapsed: {total_elapsed:.0f}s | Output: {os.path.abspath(out_root)}")


if __name__ == "__main__":
    main()
