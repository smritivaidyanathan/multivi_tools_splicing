#!/usr/bin/env python3
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mudata as mu
import scanpy as sc
import scvi


# ==========================
# USER INPUTS
# ==========================

MODEL_DIR = "/gpfs/commons/home/svaidyanathan/splice_vi_partial_vae_sweep/batch_20251105_181440/mouse_trainandtest_REAL_cd=32_mn=50000_ld=25_lr=1e-5_0_scatter_PartialEncoderEDDI_pool=sum/models"

MUDATA_PATH = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/train_70_30_model_ready_combined_gene_expression_aligned_splicing_20251009_024406.h5mu"

LABELS_CSV_PATH = "/gpfs/commons/home/svaidyanathan/analysis/subcluster_outputs/20251111_165619__MVISP_models__SCVI_model/leiden_joint_labels.csv"

# choose one or more subcluster pairs (strings to match label values)
PAIRS = [("6", "7")]  # e.g., [("6","7"), ("0","3")]

# junction_id column value (human-readable, NOT var index)
JUNCTION_ID = "chrX_136615265_136822778_+"  # DS junction of interest

# column names in labels CSV (adjust if yours differ)
CELL_ID_COL = "cell_id"
CLUSTER_COL = "leiden_joint"   # or "leiden_joint" if that’s the column in your CSV

# plotting options
PALETTE_GROUPS = ["#1f77b4", "#ff7f0e"]  # cluster A/B
DPI = 200


# ==========================
# LOGGING
# ==========================

def setup_logger(outdir: Path) -> logging.Logger:
    """
    Create logger that writes to a .out file in outdir and to stdout.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "atse_violin_multivisplice.out"

    logger = logging.getLogger("atse_violin_multivisplice")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # file
    fh = logging.FileHandler(str(log_path))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"Logging to {log_path}")
    return logger


def say(logger: logging.Logger, msg: str):
    logger.info(msg)


# ==========================
# HELPERS
# ==========================

def cell_indices_for_label(labels_df: pd.DataFrame,
                           obs_names: pd.Index,
                           label_value: str,
                           cell_id_col: str,
                           cluster_col: str) -> np.ndarray:
    cell_ids = labels_df.loc[
        labels_df[cluster_col].astype(str) == str(label_value),
        cell_id_col
    ].astype(str)

    idx = pd.Index(obs_names).get_indexer(cell_ids)
    idx = idx[idx >= 0]
    return idx


def pair_title(a, b):
    return f"Cluster {a} vs {b}"


# ==========================
# MAIN
# ==========================

def main():
    labels_csv_path = Path(LABELS_CSV_PATH)
    outdir = labels_csv_path.parent
    logger = setup_logger(outdir)

    say(logger, "=== ATSE violin plotting for MULTIVISPLICE ===")
    say(logger, f"MuData path: {MUDATA_PATH}")
    say(logger, f"Model dir: {MODEL_DIR}")
    say(logger, f"Labels CSV: {LABELS_CSV_PATH}")
    say(logger, f"Output directory: {outdir}")

    # plotting style
    sns.set(style="whitegrid", context="notebook")

    # === LOAD DATA ===
    say(logger, "Loading MuData...")
    mdata = mu.read_h5mu(MUDATA_PATH)
    ad_sp = mdata["splicing"]
    ad_rna = mdata["rna"]
    obs_names = mdata.obs_names.astype(str)

    say(logger, f"MuData loaded with {mdata.n_obs} cells, "
                f"{ad_sp.n_vars} splicing features, {ad_rna.n_vars} RNA features.")

    # ensure junction_id column is string
    if "junction_id" not in ad_sp.var.columns:
        raise KeyError("Expected mdata['splicing'].var['junction_id'] to exist.")

    ad_sp.var["junction_id"] = ad_sp.var["junction_id"].astype(str)

    # find var index corresponding to the chosen junction_id (human-readable)
    j_id = str(JUNCTION_ID)
    say(logger, f"Locating junction_id '{j_id}' in splicing var table...")

    match_rows = ad_sp.var.index[ad_sp.var["junction_id"] == j_id].tolist()
    if len(match_rows) == 0:
        raise ValueError(f"junction_id '{j_id}' not found in ad_sp.var['junction_id']")
    elif len(match_rows) > 1:
        say(logger, f"Warning: junction_id '{j_id}' maps to multiple var rows. Using the first.")

    var_name = match_rows[0]
    say(logger, f"Using var index '{var_name}' for chosen junction.")

    # junction -> ATSE and gene
    if "event_id" not in ad_sp.var.columns:
        raise KeyError("Expected mdata['splicing'].var['event_id'] to exist.")
    if "gene_name" not in ad_sp.var.columns:
        raise KeyError("Expected mdata['splicing'].var['gene_name'] to exist.")

    event_id = ad_sp.var.loc[var_name, "event_id"]
    gene_name = ad_sp.var.loc[var_name, "gene_name"]

    atse_var_names = ad_sp.var.index[ad_sp.var["event_id"] == event_id].tolist()

    say(logger, f"Chosen junction junction_id: {j_id}")
    say(logger, f"Var index: {var_name}")
    say(logger, f"ATSE event_id: {event_id} | #junctions in event: {len(atse_var_names)}")
    say(logger, f"Associated gene: {gene_name}")

    # === LOAD LABELS ===
    say(logger, "Loading Leiden joint labels CSV...")
    labels_df = pd.read_csv(labels_csv_path)

    cell_id_col = CELL_ID_COL
    cluster_col = CLUSTER_COL

    if cell_id_col not in labels_df.columns:
        maybe = [c for c in labels_df.columns if "cell" in c.lower() and "id" in c.lower()]
        if not maybe:
            maybe = [labels_df.columns[0]]
        say(logger, f"[warn] cell_id_col '{cell_id_col}' not found. Using '{maybe[0]}' instead.")
        cell_id_col = maybe[0]

    if cluster_col not in labels_df.columns:
        say(logger, f"[warn] cluster_col '{cluster_col}' not found. Using '{labels_df.columns[-1]}' instead.")
        cluster_col = labels_df.columns[-1]

    labels_df[cell_id_col] = labels_df[cell_id_col].astype(str)
    say(logger, f"Labels loaded: {labels_df.shape[0]} cells before intersecting with MuData obs.")
    # restrict to cells present in MuData
    labels_df = labels_df[labels_df[cell_id_col].isin(obs_names)].copy()

    say(logger, f"Labels loaded: {labels_df.shape[0]} cells after intersecting with MuData obs.")

    # === LOAD MULTIVISPLICE MODEL ===
    say(logger, "Loading MULTIVISPLICE model...")
    model = scvi.model.MULTIVISPLICE.load(MODEL_DIR, adata=mdata)
    say(logger, "Model loaded.")

    # === LOOP OVER PAIRS ===
    for a, b in PAIRS:
        say(logger, f"Processing pair: {a} vs {b}")
        idx_a = cell_indices_for_label(labels_df, obs_names, a, cell_id_col, cluster_col)
        idx_b = cell_indices_for_label(labels_df, obs_names, b, cell_id_col, cluster_col)

        say(logger, f"  Cells in cluster {a}: {len(idx_a)}")
        say(logger, f"  Cells in cluster {b}: {len(idx_b)}")

        if len(idx_a) == 0 or len(idx_b) == 0:
            say(logger, f"[skip] No cells for pair {a} vs {b}")
            continue

        # --- DM-normalized PSI* for the ATSE junction set (by var index) ---
        say(logger, "  Computing DM-normalized PSI* for ATSE junctions... idx_a")
        psi_a = model.get_normalized_splicing_DM(
            adata=mdata,
            indices=idx_a,
            junction_list=atse_var_names,
            use_z_mean=True,
            n_samples=1,
            return_mean=True,
            return_numpy=False,
            silent=True,
        )
        say(logger, "  Computing DM-normalized PSI* for ATSE junctions... idx_b")
        psi_b = model.get_normalized_splicing_DM(
            adata=mdata,
            indices=idx_b,
            junction_list=atse_var_names,
            use_z_mean=True,
            n_samples=1,
            return_mean=True,
            return_numpy=False,
            silent=True,
        )

        # long format for seaborn
        psi_a_m = psi_a.reset_index().melt(
            id_vars=psi_a.index.name or "index",
            var_name="var_id",
            value_name="psi_star",
        )
        psi_a_m["group"] = str(a)

        psi_b_m = psi_b.reset_index().melt(
            id_vars=psi_b.index.name or "index",
            var_name="var_id",
            value_name="psi_star",
        )
        psi_b_m["group"] = str(b)

        psi_long = pd.concat([psi_a_m, psi_b_m], ignore_index=True)

        # put the chosen junction first, then others by mean
        j_means = psi_long.groupby("var_id")["psi_star"].mean().sort_values(ascending=False)
        j_order = [var_name] + [v for v in j_means.index if v != var_name]

        # --- normalized expression for the associated gene (if present) ---
        expr_long = None
        if gene_name in ad_rna.var_names:
            say(logger, f"  Computing normalized expression for gene {gene_name}...")
            expr_a = model.get_normalized_expression(
                adata=mdata,
                indices=idx_a,
                gene_list=[gene_name],
                use_z_mean=True,
                n_samples=1,
                return_mean=True,
                return_numpy=False,
                silent=True,
            )
            expr_b = model.get_normalized_expression(
                adata=mdata,
                indices=idx_b,
                gene_list=[gene_name],
                use_z_mean=True,
                n_samples=1,
                return_mean=True,
                return_numpy=False,
                silent=True,
            )
            ea = expr_a.iloc[:, 0].rename("expr").to_frame()
            ea["group"] = str(a)
            eb = expr_b.iloc[:, 0].rename("expr").to_frame()
            eb["group"] = str(b)
            expr_long = pd.concat([ea, eb], axis=0).reset_index(drop=True)
        else:
            say(logger, f"[warn] Gene {gene_name} not found in RNA var_names; skipping expression violin.")

        # --- PLOTS ---
        ncols = 2 if expr_long is not None else 1
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4), dpi=DPI)
        if ncols == 1:
            axes = [axes]

        # Left: PSI* violins for ATSE
        ax0 = axes[0]
        sns.violinplot(
            data=psi_long,
            x="var_id",
            y="psi_star",
            hue="group",
            order=j_order,
            cut=0,
            inner="quartile",
            scale="width",
            palette=PALETTE_GROUPS,
            ax=ax0,
        )
        ax0.set_title(f"{pair_title(a, b)} — DM-normalized PSI* (ATSE)")
        ax0.set_xlabel("Junctions in ATSE (var index; selected first)")
        ax0.set_ylabel("PSI*")
        ax0.legend(title="Cluster")
        for t in ax0.get_xticklabels():
            t.set_rotation(45)
            t.set_ha("right")

        # Right: normalized expression for the associated gene
        if expr_long is not None:
            ax1 = axes[1]
            sns.violinplot(
                data=expr_long,
                x="group",
                y="expr",
                order=[str(a), str(b)],
                cut=0,
                inner="quartile",
                palette=PALETTE_GROUPS,
                ax=ax1,
            )
            ax1.set_title(f"{pair_title(a, b)} — normalized expression: {gene_name}")
            ax1.set_xlabel("Cluster")
            ax1.set_ylabel("ρ")
            if ax1.legend_ is not None:
                ax1.legend_.remove()

        plt.tight_layout()

        # save figure
        junction_safe = j_id.replace(":", "_").replace("|", "_").replace("/", "_").replace("+", "plus")
        fig_fname = f"atse_violin__{junction_safe}__cluster{a}_vs_{b}.png"
        fig_path = outdir / fig_fname

        fig.savefig(fig_path)
        plt.close(fig)

        say(logger, f"  Saved figure: {fig_path}")

    say(logger, "Done.")


if __name__ == "__main__":
    main()
