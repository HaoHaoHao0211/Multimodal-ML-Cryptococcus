# Example Output

This document shows example outputs from the multimodal nested CV analysis.

## Multimodal Nested CV Summary

After running `main_nested_cv.py`, the summary file `results/multimodal_nested_cv/Multimodal_NestedCV_Summary.csv` contains:

```
Combination,N_Modalities,Model,Metric,Mean,Std
GENOMIC_VARIANTS,1,RF,accuracy,0.7823,0.0892
GENOMIC_VARIANTS,1,RF,precision,0.7145,0.1234
GENOMIC_VARIANTS,1,RF,recall,0.6892,0.1456
GENOMIC_VARIANTS,1,RF,f1,0.6978,0.1123
GENOMIC_VARIANTS,1,RF,roc_auc,0.8234,0.0789
GENOMIC_VARIANTS,1,RF,pr_auc,0.7567,0.0923
...
```

## ROC Curves

Each modality combination generates ROC curves in `figures/` subdirectories:
- `RF_roc_nested_cv.png` — Mean ROC curve with ±std shading
- `XGB_roc_nested_cv.png` — XGBoost ROC curve

## SHAP Analysis

SHAP summary plots show feature importance:
- `RF_shap_summary_top20.png` — Beeswarm plot of top 20 features
- `RF_shap_bar_top20.png` — Bar plot of mean |SHAP| values
- `RF_shap_dependence_[feature].png` — Dependence plot for top feature

## Feature Importance

Feature importance tables are saved in `results/`:
- `RF_feature_importance_all.csv` — All features with MDI importance
- `RF_feature_importance_nonzero.csv` — Only features with importance > 0
- `RF_shap_importance.csv` — SHAP-based importance ranking

## Single-Gene Comparison

Running `single_gene.py` produces:
- `single_gene_model_comparison.png/pdf` — Boxplot comparing LR, RF, XGBoost
- `single_gene_3model_aucs.csv` — Per-fold AUC values for statistical testing

Example AUC values:
```
LR,RF,XGB
0.7234,0.7891,0.8123
0.6987,0.7456,0.7789
...
```
