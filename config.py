"""
Centralized configuration for the multimodal nested CV project.

All paths are relative to the project root directory. Font paths can be
overridden via the ``ARIAL_FONT_PATH`` environment variable.
"""

import os

# ---- Project root (two levels up from this file) ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- Data directories ----
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PHENO_PATH = os.path.join(DATA_DIR, "pheno_87_samples.csv")
GENOMIC_VARIANTS_PATH = os.path.join(DATA_DIR, "87_genomic_variants.raw")
RNA_PATH = os.path.join(DATA_DIR, "87_mean_expression_matrix.csv")
MS_PATH = os.path.join(DATA_DIR, "87_MS.csv")
LAM_PATH = os.path.join(DATA_DIR, "87_RS.csv")

# ---- Results root ----
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# ---- Arial font (override via env var if not at default location) ----
ARIAL_FONT_PATH = os.environ.get("ARIAL_FONT_PATH", "")

# ---- Random seed ----
RANDOM_STATE = 42

# ---- Nested CV parameters ----
N_OUTER_REPEATS = 5
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 5
N_BAYES_ITER_OUTER = 25
N_BAYES_ITER_FINAL = 25

# ---- Omics modalities ----
ALL_MODALITIES = ["GENOMIC_VARIANTS", "RNA", "MS", "LAM"]
MODALITY_PREFIXES = {"GENOMIC_VARIANTS": "GENOMIC_VARIANTS__", "RNA": "RNA__", "MS": "MS__", "LAM": "LAM__"}

# ---- Plot colors ----
BLUE = "#3164ad"
RED = "#9f1f24"
GRAY = "#7a7a7a"
GREEN = "#2e8b57"
