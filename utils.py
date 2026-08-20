"""
Utility functions: random seed control, data loading, feature matrix assembly.
"""

import os
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from config import (
    PHENO_PATH, GENOMIC_VARIANTS_PATH, RNA_PATH, MS_PATH, LAM_PATH,
    ARIAL_FONT_PATH, RANDOM_STATE,
)


def seed_everything(seed=RANDOM_STATE):
    """Set global random seeds for reproducibility across all frameworks."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    print(f"[Init] Global random seed: {seed}")


def setup_arial_font():
    """
    Configure matplotlib to use Arial font for figures.

    The font path can be specified via the ``ARIAL_FONT_PATH`` environment
    variable. Falls back to the system default if the file is not found.
    """
    font_path = ARIAL_FONT_PATH
    if font_path and os.path.exists(font_path):
        font_prop = font_manager.FontProperties(fname=font_path)
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = font_prop.get_name()
    else:
        # Try common system locations
        candidates = [
            "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
        for c in candidates:
            if os.path.exists(c):
                font_prop = font_manager.FontProperties(fname=c)
                font_manager.fontManager.addfont(c)
                plt.rcParams["font.family"] = font_prop.get_name()
                break
        else:
            print("[Warning] Arial font not found; using matplotlib default.")

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["axes.linewidth"] = 0.5
    plt.rcParams["xtick.major.width"] = 0.5
    plt.rcParams["ytick.major.width"] = 0.5


def _align_and_check(name, df, id_col, ref_ids):
    """
    Align a dataframe to reference sample IDs; raise on mismatch.

    Parameters
    ----------
    name : str
        Modality name (for error messages).
    df : pd.DataFrame
        Feature dataframe.
    id_col : pd.Series
        Sample identifier column from *df*.
    ref_ids : pd.Index
        Reference sample IDs (from phenotype table).

    Returns
    -------
    pd.DataFrame
        *df* reindexed to match *ref_ids* order.
    """
    df = df.set_index(id_col, drop=True)
    missing = set(ref_ids) - set(df.index)
    extra = set(df.index) - set(ref_ids)
    if missing:
        raise ValueError(f"[{name}] Missing samples vs phenotype: {sorted(missing)}")
    if extra:
        raise ValueError(f"[{name}] Extra samples vs phenotype: {sorted(extra)}")
    if not df.index.equals(ref_ids):
        print(f"[Warning] {name} row order re-aligned to phenotype order.")
    return df.reindex(ref_ids)


def load_multimodal_data():
    """
    Load four omics data types and the binary phenotype label.

    Data files are read from the paths defined in ``src.config``.
    Genomic variants are pre-imputed by Beagle 5.5.

    Returns
    -------
    data_dict : dict[str, pd.DataFrame]
        Keys: ``GENOMIC_VARIANTS``, ``RNA``, ``MS``, ``LAM``.  Each value is a
        DataFrame aligned to the phenotype sample order.
    y : np.ndarray
        Binary phenotype array (0 / 1).
    """
    # --- Phenotype (reference ID order) ---
    pheno = pd.read_csv(PHENO_PATH)
    pheno_ids = pd.Index(pheno.iloc[:, 0].astype(str).values, name="Strain ID")
    y = pheno.iloc[:, 1].astype(int).values
    assert set(y) <= {0, 1}, f"Phenotype is not binary 0/1: {set(y)}"
    print(f"[Phenotype] {len(y)} samples, {sum(y == 0)} control / {sum(y == 1)} case")

    # --- Genomic variants ---
    genome = pd.read_csv(GENOMIC_VARIANTS_PATH, sep=r"\s+")
    genome_ids = genome.iloc[:, 1].astype(str)  # IID column
    genome_data = _align_and_check("GENOMIC_VARIANTS", genome.iloc[:, 6:], genome_ids, pheno_ids)

    # --- RNA ---
    rna = pd.read_csv(RNA_PATH)
    rna_data = _align_and_check("RNA", rna.iloc[:, 1:], rna.iloc[:, 0].astype(str), pheno_ids)

    # --- MS ---
    ms = pd.read_csv(MS_PATH)
    ms_data = _align_and_check("MS", ms.iloc[:, 1:], ms.iloc[:, 0].astype(str), pheno_ids)

    # --- LAM (Raman spectroscopy) ---
    lam = pd.read_csv(LAM_PATH)
    lam_data = _align_and_check("LAM", lam.iloc[:, 1:], lam.iloc[:, 0].astype(str), pheno_ids)

    data = {"GENOMIC_VARIANTS": genome_data, "RNA": rna_data, "MS": ms_data, "LAM": lam_data}
    print(f"Loaded: GENOMIC_VARIANTS={genome_data.shape}, RNA={rna_data.shape}, "
          f"MS={ms_data.shape}, LAM={lam_data.shape}")
    return data, y


def build_X(data_dict, modalities, prefixes=None):
    """
    Concatenate selected modalities with prefixed column names.

    Parameters
    ----------
    data_dict : dict[str, pd.DataFrame]
        Full modality dictionary from :func:`load_multimodal_data`.
    modalities : list[str]
        Which modalities to include (e.g. ``["GENOMIC_VARIANTS", "RNA"]``).
    prefixes : dict[str, str] or None
        Column-name prefixes per modality.  Defaults to
        ``{"GENOMIC_VARIANTS": "GENOMIC_VARIANTS__", "RNA": "RNA__", ...}``.

    Returns
    -------
    pd.DataFrame
        Concatenated feature matrix.
    """
    if prefixes is None:
        prefixes = {"GENOMIC_VARIANTS": "GENOMIC_VARIANTS__", "RNA": "RNA__", "MS": "MS__", "LAM": "LAM__"}
    dfs = []
    for m in modalities:
        df = data_dict[m].copy()
        prefix = prefixes.get(m, f"{m}__")
        df = df.add_prefix(prefix)
        dfs.append(df)
    return pd.concat(dfs, axis=1)
