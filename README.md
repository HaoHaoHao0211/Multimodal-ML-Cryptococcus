# Multimodal Machine Learning for Antifungal Susceptibility Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Reproducible code for multimodal machine-learning analysis combining four omics
data types (genomic variants, RNA expression, mass spectrometry proteomics, and
Raman spectroscopy) for binary phenotype prediction in *Cryptococcus neoformans*
clinical isolates.

## Overview

This repository contains the analysis code for our study on predicting antifungal
susceptibility in *Cryptococcus neoformans* using multimodal data integration.

## Project Structure

```
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── .gitignore
├── run_analysis.sh        # Quick-start shell script
├── example_output.md      # Example outputs documentation
├── config.py              # Centralized paths and constants
├── utils.py               # Data loading, seed control, font setup
├── engine.py              # Nested CV + SHAP engine (RF & XGBoost)
├── main_nested_cv.py      # Multimodal nested CV (parallel)
├── single_gene.py         # Single-gene 3-model comparison
├── data/                  # Input data files
│   ├── pheno_87_samples.csv
│   ├── 87_genomic_variants.raw
│   ├── 87_mean_expression_matrix.csv
│   ├── 87_MS.csv
│   └── 87_RS.csv
└── results/               # Output directory (auto-created)
```

## Installation

### Option 1: Using conda (recommended)

```bash
conda env create -f environment.yml
conda activate multimodal
```

### Option 2: Using pip

```bash
conda create -n multimodal python=3.10
conda activate multimodal
pip install -r requirements.txt
```

### Optional: Arial Font

For publication-quality figures with Arial font:

```bash
export ARIAL_FONT_PATH=/path/to/ARIAL.TTF
```

If not set, the code falls back to system-available sans-serif fonts.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/HaoHaoHao0211/multimodal-antifungal-ml.git
cd multimodal-antifungal-ml

# Install dependencies
conda env create -f environment.yml
conda activate multimodal

# Run full analysis
bash run_analysis.sh
```

## Usage

### 1. Multimodal Nested CV (main analysis)

Runs 4 single-modality + 6 pairwise-modality combinations with Random Forest
and XGBoost, each using 5 x 5 nested cross-validation with Bayesian
hyperparameter optimization and SHAP interpretability analysis.

```bash
# Run all 10 combinations in parallel (default: 10 workers)
python main_nested_cv.py

# Control parallelism
N_PARALLEL=4 python main_nested_cv.py
```

**Output** (`results/multimodal_nested_cv/`):
- `Multimodal_NestedCV_Summary.csv` — aggregated metrics across all combinations
- Per-combination subdirectories with:
  - `RF/` and `XGB/` containing `figures/`, `results/`, `data/`
  - ROC curves, SHAP summary/dependence plots, feature importance tables

### 2. Single-Gene Model Comparison

Controlled experiment using only CNAG_07908 expression as a single feature,
comparing Logistic Regression, Random Forest, and XGBoost.

```bash
python single_gene.py
```

**Output** (same directory as the script):
- `single_gene_model_comparison.png/pdf` — boxplot comparison
- `single_gene_3model_aucs.csv` — per-fold AUC values

### Hyperparameter Search Spaces

| Model      | Hyperparameters                                                |
|------------|----------------------------------------------------------------|
| RF         | n_estimators, max_depth, min_samples_split, max_leaf_nodes, max_features |
| XGBoost    | n_estimators, learning_rate, max_depth, min_child_weight, gamma, reg_alpha, reg_lambda, subsample, colsample_bytree |

### SHAP Analysis

Post-hoc model interpretability using TreeSHAP on the final full-data model:
- Summary plots (beeswarm)
- Bar plots (mean |SHAP|)
- Dependence plots (top feature)

## Data Description

| Modality | File | Features | Description |
|----------|------|----------|-------------|
| Phenotype | `pheno_87_samples.csv` | Binary (0/1) | Antifungal susceptibility class |
| Genomic variants | `87_genomic_variants.raw` | ~15K variants | Imputed by Beagle 5.5, QC + LD-pruned |
| RNA | `87_mean_expression_matrix.csv` | 8,338 genes | Mean TPM expression matrix |
| MS | `87_MS.csv` | Proteomics | Mass spectrometry protein abundance |
| LAM | `87_RS.csv` | Raman | Raman spectroscopy intensity |


## Citation

If you use this code in your research, please cite:

```bibtex
@software{ma2026multimodal,
  author = {Haoran Ma},
  title = {Multimodal-ML-Cryptococcus},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/HaoHaoHao0211/multimodal-antifungal-ml}
}
```

See [CITATION.cff](CITATION.cff) for full citation metadata.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Example Output

See [example_output.md](example_output.md) for sample outputs and visualizations.

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@institution.edu].
