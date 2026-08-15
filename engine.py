"""
Nested cross-validation + SHAP analysis engine for RF and XGBoost.

Phase 1:  5 x 5 RepeatedStratifiedKFold nested CV  -> robust performance estimates
Phase 2:  Full-data BayesSearchCV -> final model -> feature importance + SHAP
"""

import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from matplotlib.colors import LinearSegmentedColormap

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve,
)
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Integer, Real

from config import (
    RANDOM_STATE, N_OUTER_REPEATS, N_OUTER_FOLDS, N_INNER_FOLDS,
    N_BAYES_ITER_OUTER, N_BAYES_ITER_FINAL, BLUE, RED,
)
from utils import setup_arial_font

warnings.filterwarnings("ignore")

# Initialize Arial font for all figures
setup_arial_font()

# ---- Inner-loop parallelism (auto-scale with outer parallelism) ----
N_JOBS = max(1, 10 // max(1, int(os.environ.get("N_PARALLEL", 5))))

# ---- Colormaps ----
CUSTOM_SHAP_CMAP = LinearSegmentedColormap.from_list(
    "custom_blue_to_warm_red", [BLUE, RED]
)
CUSTOM_BLUE_CMAP = LinearSegmentedColormap.from_list(
    "custom_blue_cmap", ["#eef3fb", BLUE]
)

# =====================================================================
# Hyperparameter search spaces
# =====================================================================
RF_PARAM_SPACE = {
    "clf__n_estimators":      Integer(20, 200),
    "clf__max_depth":         Integer(3, 11),
    "clf__min_samples_split": Integer(2, 10),
    "clf__max_leaf_nodes":    Integer(10, 100),
    "clf__max_features":      Real(0.05, 0.7, prior="log-uniform"),
}

XGB_PARAM_SPACE = {
    "clf__n_estimators":      Integer(20, 200),
    "clf__learning_rate":     Real(0.001, 0.2, prior="log-uniform"),
    "clf__max_depth":         Integer(3, 11),
    "clf__min_child_weight":  Integer(1, 12),
    "clf__gamma":             Real(1e-8, 1.0, prior="log-uniform"),
    "clf__reg_alpha":         Real(1e-8, 10.0, prior="log-uniform"),
    "clf__reg_lambda":        Real(1e-8, 10.0, prior="log-uniform"),
    "clf__subsample":         Real(0.5, 1.0),
    "clf__colsample_bytree":  Real(0.4, 1.0),
}


# =====================================================================
# Helper functions
# =====================================================================
def save_fig_both(base_path, dpi=600):
    """Save current figure as both PNG and PDF."""
    plt.tight_layout()
    plt.savefig(f"{base_path}.png", dpi=dpi, bbox_inches="tight")
    plt.savefig(f"{base_path}.pdf", bbox_inches="tight")


def sanitize_filename(name):
    """Replace characters that are illegal in file names."""
    return re.sub(r'[\\/:*?"<>|]', "_", str(name))


def _setup_output_dirs(out_dir):
    """Create subdirectories: figures/, results/, data/."""
    for sub in ["figures", "results", "data"]:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)


def _save_fig(out_dir, basename):
    """Save figure to figures/ subdirectory."""
    path = os.path.join(out_dir, "figures", basename)
    save_fig_both(path)


def _save_fold_predictions(all_y_true, all_proba, out_dir, prefix):
    """Save per-fold true labels and predicted probabilities to data/."""
    rows = []
    for fold_idx, (yt, yp) in enumerate(zip(all_y_true, all_proba)):
        for i, (t, p) in enumerate(zip(yt, yp)):
            rows.append({
                "fold": fold_idx + 1, "sample": i,
                "y_true": int(t), "y_proba": float(p),
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "data", f"{prefix}_fold_predictions.csv"),
        index=False,
    )


def _save_roc_data(all_fpr, mean_tpr, std_tpr, all_y_true, all_proba,
                   out_dir, prefix):
    """Save aggregated ROC curve data and per-fold AUCs to data/."""
    df_roc = pd.DataFrame({"fpr": all_fpr, "mean_tpr": mean_tpr, "std_tpr": std_tpr})
    df_roc.to_csv(
        os.path.join(out_dir, "data", f"{prefix}_roc_curve_data.csv"), index=False
    )
    aucs = [roc_auc_score(yt, yp) for yt, yp in zip(all_y_true, all_proba)]
    pd.DataFrame({"fold": range(1, len(aucs) + 1), "roc_auc": aucs}).to_csv(
        os.path.join(out_dir, "data", f"{prefix}_fold_aucs.csv"), index=False
    )


# =====================================================================
# Random Forest: nested CV + final model + SHAP
# =====================================================================
def run_rf_nested_cv_shap(X, y, feature_names, out_dir):
    """
    Run Random Forest with 5x5 nested CV and SHAP analysis.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : array-like
        Binary labels.
    feature_names : list[str]
        Column names for X.
    out_dir : str
        Output directory (figures/, results/, data/ will be created).

    Returns
    -------
    df_metrics : pd.DataFrame
        Per-fold metrics from nested CV.
    shap_importance : pd.DataFrame
        Mean absolute SHAP values per feature.
    """
    _setup_output_dirs(out_dir)
    print("\n" + "=" * 60)
    print(f"Random Forest - Nested CV + SHAP")
    print(f"Samples: {X.shape[0]} | Features: {X.shape[1]}")
    print(f"Labels: 0={np.sum(y == 0)}, 1={np.sum(y == 1)}")
    print("=" * 60)

    # ====================
    # Phase 1: Nested cross-validation
    # ====================
    print("\n" + "=" * 60)
    print("Phase 1: Nested CV (5 repeats x 5 folds)")
    print("Inner: BayesSearchCV, 5-fold CV, 25 iterations")
    print("=" * 60)

    outer_cv = RepeatedStratifiedKFold(
        n_splits=N_OUTER_FOLDS, n_repeats=N_OUTER_REPEATS,
        random_state=RANDOM_STATE,
    )

    all_proba, all_y_true, fold_results = [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train_fold = X.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_train_fold = y[train_idx]
        y_test_fold = y[test_idx]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1,
            )),
        ])

        inner_cv = StratifiedKFold(
            n_splits=N_INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE,
        )
        search = BayesSearchCV(
            estimator=pipe, search_spaces=RF_PARAM_SPACE,
            n_iter=N_BAYES_ITER_OUTER, scoring="roc_auc", cv=inner_cv,
            n_jobs=N_JOBS, random_state=RANDOM_STATE, refit=True,
        )
        search.fit(X_train_fold, y_train_fold)

        best_pipe = search.best_estimator_
        y_pred = best_pipe.predict(X_test_fold)
        y_proba = best_pipe.predict_proba(X_test_fold)[:, 1]

        fold_metrics = {
            "fold":        fold_idx + 1,
            "accuracy":    accuracy_score(y_test_fold, y_pred),
            "precision":   precision_score(y_test_fold, y_pred, zero_division=0),
            "recall":      recall_score(y_test_fold, y_pred, zero_division=0),
            "f1":          f1_score(y_test_fold, y_pred, zero_division=0),
            "roc_auc":     roc_auc_score(y_test_fold, y_proba),
            "pr_auc":      average_precision_score(y_test_fold, y_proba),
            "best_params": str(search.best_params_),
        }
        fold_results.append(fold_metrics)
        all_proba.append(y_proba)
        all_y_true.append(y_test_fold)

        print(f"  Fold {fold_idx + 1:2d}/25 | ROC-AUC={fold_metrics['roc_auc']:.4f} | "
              f"PR-AUC={fold_metrics['pr_auc']:.4f} | Recall={fold_metrics['recall']:.4f} | "
              f"n_est={search.best_params_.get('clf__n_estimators', '?')} "
              f"depth={search.best_params_.get('clf__max_depth', '?')}")

    # Summary
    df_metrics = pd.DataFrame(fold_results)
    print(f"\n{'=' * 60}")
    print("Phase 1 summary: Nested CV performance (mean +/- std over 25 folds)")
    print(f"{'=' * 60}")
    for col in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        vals = df_metrics[col].values
        print(f"  {col:12s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    df_metrics.to_csv(
        os.path.join(out_dir, "results", "RF_nested_cv_metrics.csv"), index=False
    )
    _save_fold_predictions(all_y_true, all_proba, out_dir, "RF")

    # ====================
    # Phase 2: Full-data final model + SHAP
    # ====================
    print(f"\n{'=' * 60}")
    print("Phase 2: Full-data training + feature importance + SHAP")
    print(f"{'=' * 60}")

    final_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1,
        )),
    ])

    inner_cv_final = StratifiedKFold(
        n_splits=N_INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE,
    )
    final_search = BayesSearchCV(
        estimator=final_pipe, search_spaces=RF_PARAM_SPACE,
        n_iter=N_BAYES_ITER_FINAL, scoring="roc_auc", cv=inner_cv_final,
        n_jobs=N_JOBS, random_state=RANDOM_STATE, refit=True,
    )
    final_search.fit(X, y)

    best_model = final_search.best_estimator_
    print(f"\nBest hyperparameters: {final_search.best_params_}")

    X_scaled = best_model.named_steps["scaler"].transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names, index=X.index)
    rf_model = best_model.named_steps["clf"]

    # Feature importance (MDI)
    print("\n>>> Computing feature importance (MDI)...")
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": rf_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    fi_df.to_csv(
        os.path.join(out_dir, "results", "RF_feature_importance_all.csv"),
        index=False,
    )
    non_zero_fi = fi_df[fi_df["importance"] > 0]
    print(f"  Features with importance > 0: {len(non_zero_fi)}")
    non_zero_fi.to_csv(
        os.path.join(out_dir, "results", "RF_feature_importance_nonzero.csv"),
        index=False,
    )

    # Top-20 bar plot
    top20_fi = fi_df.head(20)
    plt.figure(figsize=(10, max(6, len(top20_fi) * 0.35)))
    bar_colors = sns.light_palette(BLUE, n_colors=len(top20_fi), reverse=True)
    ax = sns.barplot(
        x="importance", y="feature", data=top20_fi,
        hue="feature", palette=bar_colors, legend=False,
    )
    for i, v in enumerate(top20_fi["importance"]):
        ax.text(v + top20_fi["importance"].max() * 0.01, i,
                f"{v:.4f}", va="center", fontsize=8)
    plt.title("Top 20 Features by RF Importance (MDI)", fontsize=15, pad=20)
    plt.xlabel("Mean Decrease in Impurity", fontsize=12)
    plt.ylabel("")
    plt.xlim(0, top20_fi["importance"].max() * 1.15)
    _save_fig(out_dir, "RF_top20_feature_importance")
    plt.close()

    # SHAP analysis
    print("\n>>> Computing SHAP values...")
    explainer = shap.TreeExplainer(rf_model, data=X_scaled)
    shap_values_raw = explainer.shap_values(X_scaled)

    if isinstance(shap_values_raw, list):
        shap_values = np.asarray(shap_values_raw[1])
    else:
        shap_values = np.asarray(shap_values_raw)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

    print(f"  SHAP matrix shape: {shap_values.shape}")

    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.to_csv(
        os.path.join(out_dir, "data", "RF_shap_values_all_samples.csv"),
        index=False,
    )

    shap_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    shap_importance.to_csv(
        os.path.join(out_dir, "results", "RF_shap_importance.csv"), index=False,
    )
    print("  SHAP Top-10 features:")
    print(shap_importance.head(10).to_string(index=False))

    # SHAP summary plot
    print("\n>>> Plotting SHAP figures...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_scaled_df, feature_names=feature_names,
        max_display=20, cmap=CUSTOM_SHAP_CMAP, show=False,
    )
    plt.title("SHAP Summary Plot - Random Forest (Top 20)", fontsize=15, pad=20)
    _save_fig(out_dir, "RF_shap_summary_top20")
    plt.close()

    # SHAP bar plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_scaled_df, feature_names=feature_names,
        plot_type="bar", max_display=20, color=RED, show=False,
    )
    plt.title("RF Feature Importance Ranking by SHAP", fontsize=15, pad=20)
    _save_fig(out_dir, "RF_shap_bar_top20")
    plt.close()

    # SHAP dependence plot (top feature)
    top_feat_name = shap_importance["feature"].iloc[0]
    top_feat_idx = feature_names.index(top_feat_name)
    plt.figure(figsize=(10, 8))
    shap.dependence_plot(
        top_feat_idx, shap_values, X_scaled_df,
        feature_names=feature_names, cmap=CUSTOM_SHAP_CMAP, show=False,
    )
    plt.title(f"SHAP Dependence Plot - {top_feat_name}", fontsize=15, pad=20)
    _save_fig(out_dir, f"RF_shap_dependence_{sanitize_filename(top_feat_name)}")
    plt.close()

    # Aggregated ROC curve
    print("\n>>> Plotting aggregated ROC curve...")
    _plot_aggregated_roc(all_y_true, all_proba, out_dir, "RF",
                         "Random Forest (5x5 Nested CV)")

    print(f"\nRF complete! Results: {os.path.abspath(out_dir)}")
    return df_metrics, shap_importance


# =====================================================================
# XGBoost: nested CV + final model + SHAP
# =====================================================================
def run_xgb_nested_cv_shap(X, y, feature_names, out_dir):
    """
    Run XGBoost with 5x5 nested CV and SHAP analysis.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : array-like
        Binary labels.
    feature_names : list[str]
        Column names for X.
    out_dir : str
        Output directory (figures/, results/, data/ will be created).

    Returns
    -------
    df_metrics : pd.DataFrame
        Per-fold metrics from nested CV.
    shap_importance : pd.DataFrame
        Mean absolute SHAP values per feature.
    """
    _setup_output_dirs(out_dir)
    print("\n" + "=" * 60)
    print(f"XGBoost - Nested CV + SHAP")
    print(f"Samples: {X.shape[0]} | Features: {X.shape[1]}")
    print(f"Labels: 0={np.sum(y == 0)}, 1={np.sum(y == 1)}")
    print("=" * 60)

    # ====================
    # Phase 1: Nested cross-validation
    # ====================
    print("\n" + "=" * 60)
    print("Phase 1: Nested CV (5 repeats x 5 folds)")
    print("Inner: BayesSearchCV, 5-fold CV, 25 iterations")
    print("=" * 60)

    outer_cv = RepeatedStratifiedKFold(
        n_splits=N_OUTER_FOLDS, n_repeats=N_OUTER_REPEATS,
        random_state=RANDOM_STATE,
    )

    all_proba, all_y_true, fold_results = [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train_fold = X.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_train_fold = y[train_idx]
        y_test_fold = y[test_idx]

        scale_pos_weight = np.sum(y_train_fold == 0) / max(np.sum(y_train_fold == 1), 1)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                objective="binary:logistic", eval_metric="auc",
                scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE,
                device="cpu", n_jobs=1, verbosity=0,
            )),
        ])

        inner_cv = StratifiedKFold(
            n_splits=N_INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE,
        )
        search = BayesSearchCV(
            estimator=pipe, search_spaces=XGB_PARAM_SPACE,
            n_iter=N_BAYES_ITER_OUTER, scoring="roc_auc", cv=inner_cv,
            n_jobs=N_JOBS, random_state=RANDOM_STATE, refit=True,
        )
        search.fit(X_train_fold, y_train_fold)

        best_pipe = search.best_estimator_
        y_pred = best_pipe.predict(X_test_fold)
        y_proba = best_pipe.predict_proba(X_test_fold)[:, 1]

        fold_metrics = {
            "fold":        fold_idx + 1,
            "accuracy":    accuracy_score(y_test_fold, y_pred),
            "precision":   precision_score(y_test_fold, y_pred, zero_division=0),
            "recall":      recall_score(y_test_fold, y_pred, zero_division=0),
            "f1":          f1_score(y_test_fold, y_pred, zero_division=0),
            "roc_auc":     roc_auc_score(y_test_fold, y_proba),
            "pr_auc":      average_precision_score(y_test_fold, y_proba),
            "best_params": str(search.best_params_),
        }
        fold_results.append(fold_metrics)
        all_proba.append(y_proba)
        all_y_true.append(y_test_fold)

        lr_val = search.best_params_.get("clf__learning_rate", "?")
        lr_str = f"{lr_val:.4f}" if isinstance(lr_val, float) else str(lr_val)
        print(f"  Fold {fold_idx + 1:2d}/25 | ROC-AUC={fold_metrics['roc_auc']:.4f} | "
              f"PR-AUC={fold_metrics['pr_auc']:.4f} | Recall={fold_metrics['recall']:.4f} | "
              f"n_est={search.best_params_.get('clf__n_estimators', '?')} "
              f"lr={lr_str}")

    # Summary
    df_metrics = pd.DataFrame(fold_results)
    print(f"\n{'=' * 60}")
    print("Phase 1 summary: Nested CV performance (mean +/- std over 25 folds)")
    print(f"{'=' * 60}")
    for col in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        vals = df_metrics[col].values
        print(f"  {col:12s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    df_metrics.to_csv(
        os.path.join(out_dir, "results", "XGB_nested_cv_metrics.csv"), index=False,
    )
    _save_fold_predictions(all_y_true, all_proba, out_dir, "XGB")

    # ====================
    # Phase 2: Full-data final model + SHAP
    # ====================
    print(f"\n{'=' * 60}")
    print("Phase 2: Full-data training + feature importance + SHAP")
    print(f"{'=' * 60}")

    scale_pos_weight_full = np.sum(y == 0) / max(np.sum(y == 1), 1)

    final_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            objective="binary:logistic", eval_metric="auc",
            scale_pos_weight=scale_pos_weight_full, random_state=RANDOM_STATE,
            device="cpu", n_jobs=1, verbosity=0,
        )),
    ])

    inner_cv_final = StratifiedKFold(
        n_splits=N_INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE,
    )
    final_search = BayesSearchCV(
        estimator=final_pipe, search_spaces=XGB_PARAM_SPACE,
        n_iter=N_BAYES_ITER_FINAL, scoring="roc_auc", cv=inner_cv_final,
        n_jobs=N_JOBS, random_state=RANDOM_STATE, refit=True,
    )
    final_search.fit(X, y)

    best_model = final_search.best_estimator_
    print(f"\nBest hyperparameters: {final_search.best_params_}")

    X_scaled = best_model.named_steps["scaler"].transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names, index=X.index)
    xgb_model = best_model.named_steps["clf"]

    # Feature importance (weight)
    print("\n>>> Computing feature importance (weight)...")
    fi_array = xgb_model.feature_importances_
    all_fi_df = pd.DataFrame({
        "feature": feature_names, "importance": fi_array,
    }).sort_values("importance", ascending=False)

    all_fi_df.to_csv(
        os.path.join(out_dir, "results", "XGB_feature_importance_weight_all.csv"),
        index=False,
    )
    non_zero_fi = all_fi_df[all_fi_df["importance"] > 0]
    print(f"  Features with importance > 0: {len(non_zero_fi)}")
    non_zero_fi.to_csv(
        os.path.join(out_dir, "results", "XGB_feature_importance_weight_nonzero.csv"),
        index=False,
    )

    # Top-20 bar plot
    top20_fi = all_fi_df.head(20)
    plt.figure(figsize=(10, max(6, len(top20_fi) * 0.35)))
    bar_colors = sns.light_palette(BLUE, n_colors=len(top20_fi), reverse=True)
    ax = sns.barplot(
        x="importance", y="feature", data=top20_fi,
        hue="feature", palette=bar_colors, legend=False,
    )
    for i, v in enumerate(top20_fi["importance"]):
        ax.text(v + top20_fi["importance"].max() * 0.01, i,
                f"{v:.4f}", va="center", fontsize=8)
    plt.title("Top 20 Features by XGBoost Importance (Weight)", fontsize=15, pad=20)
    plt.xlabel("Feature Importance (Weight)", fontsize=12)
    plt.ylabel("")
    plt.xlim(0, top20_fi["importance"].max() * 1.15)
    _save_fig(out_dir, "XGB_top20_feature_importance_weight")
    plt.close()

    # Additional importance types: gain, cover
    for imp_type in ["gain", "cover"]:
        fi_alt_dict = xgb_model.get_booster().get_score(importance_type=imp_type)
        if fi_alt_dict:
            fi_alt_df = pd.DataFrame({
                "f_index": list(fi_alt_dict.keys()),
                f"importance_{imp_type}": list(fi_alt_dict.values()),
            })
            fi_alt_df["f_idx"] = fi_alt_df["f_index"].str.replace("f", "").astype(int)
            fi_alt_df["feature"] = fi_alt_df["f_idx"].apply(lambda i: feature_names[i])
            fi_alt_df = fi_alt_df[["feature", f"importance_{imp_type}"]].sort_values(
                f"importance_{imp_type}", ascending=False,
            )
        else:
            fi_alt_df = pd.DataFrame({
                "feature": feature_names, f"importance_{imp_type}": 0.0,
            })
        fi_alt_df.to_csv(
            os.path.join(out_dir, "results", f"XGB_feature_importance_{imp_type}.csv"),
            index=False,
        )

    # SHAP analysis
    print("\n>>> Computing SHAP values...")
    explainer = shap.TreeExplainer(xgb_model, data=X_scaled)
    shap_values_raw = explainer.shap_values(X_scaled)

    if isinstance(shap_values_raw, list):
        shap_values = np.asarray(shap_values_raw[1])
    else:
        shap_values = np.asarray(shap_values_raw)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

    print(f"  SHAP matrix shape: {shap_values.shape}")

    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.to_csv(
        os.path.join(out_dir, "data", "XGB_shap_values_all_samples.csv"),
        index=False,
    )

    shap_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    shap_importance.to_csv(
        os.path.join(out_dir, "results", "XGB_shap_importance.csv"), index=False,
    )
    print("  SHAP Top-10 features:")
    print(shap_importance.head(10).to_string(index=False))

    # SHAP summary plot
    print("\n>>> Plotting SHAP figures...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_scaled_df, feature_names=feature_names,
        max_display=20, cmap=CUSTOM_SHAP_CMAP, show=False,
    )
    plt.title("SHAP Summary Plot - XGBoost (Top 20)", fontsize=15, pad=20)
    _save_fig(out_dir, "XGB_shap_summary_top20")
    plt.close()

    # SHAP bar plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_scaled_df, feature_names=feature_names,
        plot_type="bar", max_display=20, color=RED, show=False,
    )
    plt.title("XGBoost Feature Importance Ranking by SHAP", fontsize=15, pad=20)
    _save_fig(out_dir, "XGB_shap_bar_top20")
    plt.close()

    # SHAP dependence plot (top feature)
    top_feat_name = shap_importance["feature"].iloc[0]
    top_feat_idx = feature_names.index(top_feat_name)
    plt.figure(figsize=(10, 8))
    shap.dependence_plot(
        top_feat_idx, shap_values, X_scaled_df,
        feature_names=feature_names, cmap=CUSTOM_SHAP_CMAP, show=False,
    )
    plt.title(f"SHAP Dependence Plot - {top_feat_name}", fontsize=15, pad=20)
    _save_fig(out_dir, f"XGB_shap_dependence_{sanitize_filename(top_feat_name)}")
    plt.close()

    # Aggregated ROC curve
    print("\n>>> Plotting aggregated ROC curve...")
    _plot_aggregated_roc(all_y_true, all_proba, out_dir, "XGB",
                         "XGBoost (5x5 Nested CV)")

    print(f"\nXGBoost complete! Results: {os.path.abspath(out_dir)}")
    return df_metrics, shap_importance


# =====================================================================
# Shared plotting helper
# =====================================================================
def _plot_aggregated_roc(all_y_true, all_proba, out_dir, prefix, title):
    """Plot mean ROC curve with +/- std shading."""
    plt.figure(figsize=(8, 6))
    all_fpr = np.linspace(0, 1, 100)
    tprs, aucs = [], []
    for yt, yp in zip(all_y_true, all_proba):
        fpr, tpr, _ = roc_curve(yt, yp)
        aucs.append(roc_auc_score(yt, yp))
        interp_tpr = np.interp(all_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    plt.plot(all_fpr, mean_tpr, color=RED, linewidth=2,
             label=f"Mean ROC (AUC = {mean_auc:.3f} +/- {std_auc:.3f})")
    plt.fill_between(all_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                     color=RED, alpha=0.15)
    plt.plot([0, 1], [0, 1], "--", color=BLUE, linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {title}")
    plt.legend(loc="lower right")
    _save_fig(out_dir, f"{prefix}_roc_nested_cv")
    plt.close()

    _save_roc_data(all_fpr, mean_tpr, std_tpr, all_y_true, all_proba, out_dir, prefix)
