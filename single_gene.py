#!/usr/bin/env python3
"""
Single-gene (CNAG_07908) model comparison  --  25-fold nested CV.

Controlled experiment: fixed feature = 1 gene (CNAG_07908 TPM).
Independent variable: model type (LR / RF / XGBoost).

"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Integer, Real

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    PHENO_PATH, RNA_PATH, RANDOM_STATE, RED, BLUE, GRAY,
    N_OUTER_FOLDS, N_OUTER_REPEATS, N_INNER_FOLDS,
)
from utils import setup_arial_font

# ---- Output directory (same as script location) ----
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Font setup ----
setup_arial_font()
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["xtick.major.width"] = 1.0
plt.rcParams["ytick.major.width"] = 1.0

GREEN = "#2e8b57"
N_BAYES_ITER = 15  # 1 gene -> fast convergence, reduced from 25 for speed


# ============================================================
# Load data
# ============================================================
pheno = pd.read_csv(PHENO_PATH)
y = pheno.iloc[:, 6].astype(int).values
sample_ids = pheno.iloc[:, 1].astype(str).values

expr = pd.read_csv(RNA_PATH)
expr_ids = expr.iloc[:, 0].astype(str).values
expr_df = pd.DataFrame({"CDCF_No": expr_ids, "TPM": expr["CNAG_07908"].values}).set_index("CDCF_No")
X_1gene = expr_df.reindex(sample_ids)["TPM"].values.reshape(-1, 1)

print(f"Samples: {len(y)}  (0={sum(y == 0)}, 1={sum(y == 1)})")
print(f"Feature: CNAG_07908 TPM  (median={np.median(X_1gene):.1f})")


# ============================================================
# 25-fold nested CV  --  three models
# ============================================================
outer_cv = RepeatedStratifiedKFold(
    n_splits=N_OUTER_FOLDS, n_repeats=N_OUTER_REPEATS, random_state=RANDOM_STATE,
)

results = {"LR": [], "RF": [], "XGB": []}

for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_1gene, y)):
    X_train, X_test = X_1gene[train_idx], X_1gene[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # --- Logistic Regression ---
    s_lr = StandardScaler()
    lr = LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE,
    )
    lr.fit(s_lr.fit_transform(X_train), y_train)
    proba_lr = lr.predict_proba(s_lr.transform(X_test))[:, 1]
    results["LR"].append(roc_auc_score(y_test, proba_lr))

    # --- Random Forest (with inner BayesSearch) ---
    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1,
        )),
    ])
    inner_cv = StratifiedKFold(
        n_splits=N_INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE,
    )
    rf_search = BayesSearchCV(
        rf_pipe,
        {"clf__n_estimators":      Integer(20, 200),
         "clf__max_depth":         Integer(3, 11),
         "clf__min_samples_split": Integer(2, 10),
         "clf__max_leaf_nodes":    Integer(10, 100),
         "clf__max_features":      Real(0.05, 0.7, prior="log-uniform")},
        n_iter=N_BAYES_ITER, scoring="roc_auc", cv=inner_cv,
        n_jobs=1, random_state=RANDOM_STATE, refit=True,
    )
    rf_search.fit(X_train, y_train)
    proba_rf = rf_search.best_estimator_.predict_proba(X_test)[:, 1]
    results["RF"].append(roc_auc_score(y_test, proba_rf))

    # --- XGBoost (with inner BayesSearch) ---
    spw = sum(y_train == 0) / max(sum(y_train == 1), 1)
    xgb_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            objective="binary:logistic", eval_metric="auc",
            scale_pos_weight=spw, random_state=RANDOM_STATE,
            device="cpu", n_jobs=1, verbosity=0,
        )),
    ])
    xgb_search = BayesSearchCV(
        xgb_pipe,
        {"clf__n_estimators":      Integer(20, 200),
         "clf__learning_rate":     Real(0.001, 0.2, prior="log-uniform"),
         "clf__max_depth":         Integer(3, 11),
         "clf__min_child_weight":  Integer(1, 12),
         "clf__gamma":             Real(1e-8, 1.0, prior="log-uniform"),
         "clf__reg_alpha":         Real(1e-8, 10.0, prior="log-uniform"),
         "clf__reg_lambda":        Real(1e-8, 10.0, prior="log-uniform"),
         "clf__subsample":         Real(0.5, 1.0),
         "clf__colsample_bytree":  Real(0.4, 1.0)},
        n_iter=N_BAYES_ITER, scoring="roc_auc", cv=inner_cv,
        n_jobs=1, random_state=RANDOM_STATE, refit=True,
    )
    xgb_search.fit(X_train, y_train)
    proba_xgb = xgb_search.best_estimator_.predict_proba(X_test)[:, 1]
    results["XGB"].append(roc_auc_score(y_test, proba_xgb))

    if (fold_idx + 1) % 5 == 0:
        print(f"  Fold {fold_idx + 1:2d}/25 | LR={results['LR'][-1]:.4f}  "
              f"RF={results['RF'][-1]:.4f}  XGB={results['XGB'][-1]:.4f}")


# ============================================================
# Summary
# ============================================================
print(f"\n{'=' * 60}")
print(f"Single gene CNAG_07908 - 3-model 25-fold nested CV")
print(f"{'=' * 60}")
print(f"  {'Model':<20s} {'AUC mean':>8s} {'AUC std':>8s} {'Median':>8s} {'Min':>8s} {'Max':>8s}")
print(f"  {'-' * 55}")

for name in ["LR", "RF", "XGB"]:
    vals = np.array(results[name])
    print(f"  {name:<20s} {np.mean(vals):>8.4f} {np.std(vals):>8.4f} "
          f"{np.median(vals):>8.4f} {np.min(vals):>8.4f} {np.max(vals):>8.4f}")

# Paired Wilcoxon signed-rank test
print(f"\n  Paired Wilcoxon test (same folds):")
for a, b in [("LR", "RF"), ("LR", "XGB"), ("RF", "XGB")]:
    diff = np.array(results[a]) - np.array(results[b])
    w, p = stats.wilcoxon(diff)
    print(f"    {a} vs {b}: diff={np.mean(diff):+.4f}, p={p:.4f}")


# ============================================================
# Boxplot
# ============================================================
fig, ax = plt.subplots(figsize=(9, 6))
data_list = [results["LR"], results["RF"], results["XGB"]]
labels = ["Logistic\nRegression", "Random\nForest", "XGBoost"]
colors = [RED, GRAY, BLUE]

bp = ax.boxplot(data_list, patch_artist=True, widths=0.4,
                medianprops={"color": "black", "linewidth": 1.5})
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
for i, (data, color) in enumerate(zip(data_list, colors)):
    jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=len(data))
    ax.scatter(np.full_like(data, i + 1) + jitter, data, alpha=0.5, s=15,
               color=color, edgecolors="white", linewidth=0.3, zorder=3)

ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("ROC-AUC (25-fold Nested CV)", fontsize=12)
ax.set_title("Single Gene (CNAG_07908) - Model Comparison\n"
             f"n=87, positive=11, 1 feature", fontsize=13, pad=12)
ax.axhline(y=0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.3)
ax.set_ylim(0.4, 1.05)

plt.tight_layout()
for fmt in ["png", "pdf"]:
    plt.savefig(os.path.join(OUT_DIR, f"single_gene_model_comparison.{fmt}"),
                dpi=600, bbox_inches="tight")
plt.close()
print(f"\n  -> single_gene_model_comparison.png/pdf")

# Save raw AUC values
pd.DataFrame({k: np.array(v) for k, v in results.items()}).to_csv(
    os.path.join(OUT_DIR, "single_gene_3model_aucs.csv"), index=False,
)

print("\nDone.")
