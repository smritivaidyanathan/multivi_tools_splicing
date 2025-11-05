#!/usr/bin/env python3
import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import mudata as mu
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Plot defaults
sns.set_context("notebook")
sns.set_style("whitegrid")


def setup_logger(outdir: Path) -> logging.Logger:
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "run.log"
    logger = logging.getLogger("subcluster")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(str(log_path))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def say(logger: logging.Logger, msg: str):
    print(msg, flush=True)
    logger.info(msg)


def parse_args():
    p = argparse.ArgumentParser("Subcluster analysis for MULTIVISPLICE")
    p.add_argument("--model_dir", required=True, help="Path to trained MULTIVISPLICE model directory")
    p.add_argument("--mudata_path", required=True, help="Input MuData .h5mu path")
    p.add_argument("--base_outdir", required=True, help="Base folder where timestamped outputs are created")
    p.add_argument("--leiden_resolution", type=float, default=1.0, help="Leiden resolution for joint space")
    p.add_argument("--preferred_celltype_col", default="medium_cell_type", help="Preferred obs column for cell types")
    p.add_argument("--fallback_celltype_col", default="broad_cell_type", help="Fallback obs column for cell types")
    p.add_argument("--target_celltype", required=True, help="Exact label to subset, e.g. 'Cortical excitatory neuron'")
    p.add_argument("--target_tissues", nargs="*", default=[], help="Optional list of tissue names to include")
    p.add_argument("--run_tsne", action="store_true", help="Also compute tSNE for joint space")
    p.add_argument("--de_delta", type=float, default=0.25, help="Delta for DE with mode=change")
    p.add_argument("--ds_delta", type=float, default=0.10, help="Delta for DS on PSI with mode=change")
    p.add_argument("--fdr", type=float, default=0.05, help="Target FDR")
    p.add_argument("--batch_size_post", type=int, default=512, help="Batch size for decoding")
    p.add_argument("--n_top_show", type=int, default=12, help="How many hits to print for quick checks")
    p.add_argument("--x_layer", default="junc_ratio", help="Splicing PSI layer name")
    p.add_argument("--junction_counts_layer", default="cell_by_junction_matrix")
    p.add_argument("--cluster_counts_layer", default="cell_by_cluster_matrix")
    p.add_argument("--mask_layer", default="psi_mask")
    p.add_argument("--norm_splicing_function", default = "decoder")
    return p.parse_args()


def pick_celltype_key(ad_rna, preferred: str, fallback: str | None) -> str:
    if preferred in ad_rna.obs:
        return preferred
    if fallback and fallback in ad_rna.obs:
        return fallback
    raise ValueError("Could not find a cell type column in obs")

def plot_junction_coverage_vs_variance(mdata, x_layer: str, junc_counts_layer: str, mask_layer: str, outdir: Path, logger: logging.Logger):
    """
    For each junction:
      x = # cells with junc_counts > 0
      y = variance of PSI over observed cells (mask == 1)

    Saves:
      - scatter PNG: outdir / "junction_coverage_vs_variance.png"
      - per-junction CSV: outdir / "junction_coverage_vs_variance.csv"
    """
    say(logger, "[qc] Computing junction coverage vs PSI variance")

    ad_spl = mdata["splicing"]
    X = ad_spl.layers.get(x_layer, ad_spl.X)          # PSI matrix (cells x junctions)
    M = ad_spl.layers.get(mask_layer, None)           # observation mask (same shape)
    JC = ad_spl.layers.get(junc_counts_layer, None)   # junction counts (cells x junctions)

    if JC is None:
        raise RuntimeError(f"Missing junction counts layer '{junc_counts_layer}' in splicing modality")
    if M is None:
        raise RuntimeError(f"Missing mask layer '{mask_layer}' in splicing modality")

    # Convert to compatible types
    import scipy.sparse as sp
    is_sparse_X = sp.issparse(X)
    is_sparse_M = sp.issparse(M)
    is_sparse_JC = sp.issparse(JC)

    # Cells-expressed count per junction: (# cells with count > 0)
    if is_sparse_JC:
        cells_expressed = (JC > 0).sum(axis=0).A1
    else:
        cells_expressed = (JC > 0).sum(axis=0)

    # Variance of PSI across observed cells
    eps = 1e-12
    if is_sparse_X or is_sparse_M:
        # Ensure CSR for efficient row ops then do elementwise products
        X_csr = X.tocsr() if is_sparse_X else sp.csr_matrix(X)
        M_csr = M.tocsr() if is_sparse_M else sp.csr_matrix(M)

        # observed counts per junction
        n_obs = M_csr.sum(axis=0).A1  # length J

        # sum of x and x^2 over observed entries
        X_masked = X_csr.multiply(M_csr)
        sum_x = X_masked.sum(axis=0).A1
        sum_x2 = X_masked.multiply(X_masked).sum(axis=0).A1

        mean = np.divide(sum_x, np.maximum(n_obs, 1), where=(n_obs > 0))
        var = np.divide(sum_x2, np.maximum(n_obs, 1), where=(n_obs > 0)) - mean**2
        var = np.clip(var, 0.0, None)
        var[n_obs == 0] = np.nan
    else:
        # dense
        M_bool = M.astype(bool)
        n_obs = M_bool.sum(axis=0)
        sum_x = (X * M_bool).sum(axis=0)
        sum_x2 = ((X * M_bool) ** 2).sum(axis=0)
        mean = np.divide(sum_x, np.maximum(n_obs, 1), where=(n_obs > 0))
        var = np.divide(sum_x2, np.maximum(n_obs, 1), where=(n_obs > 0)) - mean**2
        var = np.clip(var, 0.0, None)
        var[n_obs == 0] = np.nan

    # Assemble DataFrame
    df = pd.DataFrame({
        "junction": ad_spl.var_names,
        "cells_expressed": cells_expressed.astype(int),
        "psi_var": var,
        "n_obs_mask": n_obs.astype(int),
    }).set_index("junction")

    # Save CSV
    csv_path = outdir / "junction_coverage_vs_variance.csv"
    df.to_csv(csv_path)
    say(logger, f"[qc] Wrote CSV → {csv_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="cells_expressed",
        y="psi_var",
        s=8,
        alpha=0.5,
        ax=ax,
    )
    ax.set_xlabel("# cells with junction count > 0")
    ax.set_ylabel("PSI variance across observed cells")
    ax.set_title("Junction coverage vs PSI variance")
    plt.tight_layout()
    png_path = outdir / "junction_coverage_vs_variance.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    say(logger, f"[qc] Saved plot → {png_path}")


def subset_mudata(mdata, ct_key: str, target_celltype: str, target_tissues: list[str],
                  logger: logging.Logger):
    # Kept for compatibility, not used in this script anymore
    obs_rna = mdata["rna"].obs
    mask = obs_rna[ct_key] == target_celltype
    if target_tissues and "tissue" in obs_rna:
        tt = [t.lower() for t in target_tissues]
        mask &= obs_rna["tissue"].astype(str).str.lower().isin(tt)
    n_keep = int(mask.sum())
    if n_keep == 0:
        raise ValueError(f"No cells matched '{target_celltype}' with tissues={target_tissues}")
    say(logger, f"[subset] Keeping {n_keep} cells for '{target_celltype}', tissues={target_tissues}")
    return mdata[mask]


def run_leiden_umap_joint(model, mdata, ct_key: str, target_celltype: str, target_tissues: list[str],
                          resolution: float, run_tsne: bool,
                          outdir: Path, logger: logging.Logger):
    say(logger, "[latent] Subsetting inside Leiden function and computing joint latent")

    # Build subset mask here
    ad_rna_full = mdata["rna"]
    mask_ct = ad_rna_full.obs[ct_key] == target_celltype
    if target_tissues and "tissue" in ad_rna_full.obs:
        tt = [t.lower() for t in target_tissues]
        mask_ct &= ad_rna_full.obs["tissue"].astype(str).str.lower().isin(tt)

    n_keep = int(mask_ct.sum())
    if n_keep == 0:
        raise ValueError(f"No cells matched '{target_celltype}' with tissues={target_tissues}")
    say(logger, f"[subset] Keeping {n_keep} cells for '{target_celltype}', tissues={target_tissues}")

    # Work on a sliced MuData for the target population
    m_ct = mdata[mask_ct].copy()
    ad_rna = m_ct["rna"]

    # Joint latent for the subset only
    Z_joint_full = model.get_latent_representation(modality="joint")
    Z_subset = Z_joint_full[mask_ct]
    ad_rna.obsm["X_latent_joint"] = Z_subset

    say(logger, f"[latent] Z_joint shape: {Z_subset.shape}")

    # Neighbors + Leiden on the subset
    sc.pp.neighbors(ad_rna, use_rep="X_latent_joint", key_added="neighbors_joint")
    sc.tl.leiden(ad_rna, neighbors_key="neighbors_joint", key_added="leiden_joint", resolution=resolution)

    # Embeddings
    sc.tl.umap(ad_rna, neighbors_key="neighbors_joint")
    ad_rna.obsm["X_umap_joint"] = ad_rna.obsm["X_umap"]

    if run_tsne:
        sc.tl.tsne(ad_rna, use_rep="X_latent_joint", learning_rate=200.0,
                   perplexity=30.0, random_state=0)
        ad_rna.obsm["X_tsne_joint"] = ad_rna.obsm["X_tsne"]

    ncl = int(ad_rna.obs["leiden_joint"].nunique())
    say(logger, f"[leiden] Found {ncl} clusters at res={resolution}")

    # Plots
    say(logger, "[plot] Saving UMAP colored by Leiden")
    fig, ax = plt.subplots(figsize=(8, 6))
    sc.pl.embedding(ad_rna, basis="X_umap_joint", color="leiden_joint",
                    legend_loc="right margin", show=False, ax=ax, frameon=True)
    ax.set_title(f"Joint UMAP | Leiden res={resolution}")
    plt.tight_layout()
    fig.savefig(outdir / "umap_joint_leiden.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    if run_tsne:
        say(logger, "[plot] Saving tSNE colored by Leiden")
        fig, ax = plt.subplots(figsize=(8, 6))
        sc.pl.embedding(ad_rna, basis="X_tsne_joint", color="leiden_joint",
                        legend_loc="right margin", show=False, ax=ax, frameon=True)
        ax.set_title(f"Joint tSNE | Leiden res={resolution}")
        plt.tight_layout()
        fig.savefig(outdir / "tsne_joint_leiden.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Return the RNA AnnData of the subset, like before
    return ad_rna


def run_pairwise_de_ds(model, m_ct, ad_rna_ct, de_delta, ds_delta, fdr,
                       batch_size_post, norm_splicing_function, logger: logging.Logger):
    say(logger, "[pairwise] Running pairwise DE/DS across subclusters in target population")
    pairs = []
    de_tables = {}
    ds_tables = {}

    clabs = ad_rna_ct.obs["leiden_joint"].astype(str).values
    unique_subs = sorted(pd.unique(clabs), key=lambda x: (len(x), x))
    say(logger, f"[pairwise] Subclusters: {unique_subs}")

    
    for i in range(len(unique_subs)):
        for j in range(i + 1, len(unique_subs)):
            a = unique_subs[i]
            b = unique_subs[j]
            idx1 = (clabs == a)
            idx2 = (clabs == b)
            key = f"{a}_vs_{b}"
            say(logger, f"[pairwise] {key}: DE (RNA) …")
            de_df = model.differential_expression(
                adata=m_ct,
                idx1=idx1,
                idx2=idx2,
                mode="change",
                delta=de_delta,
                fdr_target=fdr,
                batch_size=batch_size_post,
                all_stats=True,
                silent=True,
            )
            say(logger, f"[pairwise] {key}: DS (splicing PSI) …")
            ds_df = model.differential_splicing(
                adata=m_ct,
                idx1=idx1,
                idx2=idx2,
                mode="change",
                delta=ds_delta,
                fdr_target=fdr,
                batch_size=batch_size_post,
                all_stats=True,
                silent=True,
                norm_splicing_function=norm_splicing_function,
            )

            pairs.append(key)
            de_tables[key] = de_df
            ds_tables[key] = ds_df
            say(logger, f"[pairwise] {key}: done")

    say(logger, "[pairwise] All pairwise comparisons complete")
    return pairs, de_tables, ds_tables


def summarize_pair(de_df, ds_df, fdr, mdata, logger: logging.Logger):
    # Identify probability columns
    col_prob_de = "proba_de" if "proba_de" in de_df.columns else ("probability" if "probability" in de_df.columns else None)
    col_prob_ds = "proba_ds" if "proba_ds" in ds_df.columns else ("proba_de" if "proba_de" in ds_df.columns else None)
    if col_prob_de is None or col_prob_ds is None:
        raise RuntimeError("Could not find probability columns for DE or DS output")

    # Significant hits
    sig_de = de_df[de_df[col_prob_de] >= 1 - fdr]
    sig_ds = ds_df[ds_df[col_prob_ds] >= 1 - fdr]

    # Lookups to gene_name
    rna_var = mdata["rna"].var
    spl_var = mdata["splicing"].var

    # RNA: map DE index -> gene_name (fallback to index if missing)
    if "gene_name" in rna_var.columns:
        rna_name_map = rna_var["gene_name"].astype(str)
        rna_name_map.index = rna_var.index.astype(str)

        idx = sig_de.index.astype(str)
        fallback = pd.Series(idx, index=idx)  # <- Series, not Index
        de_gene_names = (
            rna_name_map.reindex(idx)
            .fillna(fallback)
            .astype(str)
        )
    else:
        de_gene_names = pd.Series(sig_de.index.astype(str), index=sig_de.index)

    # Splicing: map junction var_names -> gene_name (fallback to junction id if missing)
    # Splicing: map junction var_names -> gene_name (fallback to junction id if missing)
    if "gene_name" in spl_var.columns:
        ds_index = sig_ds.index.astype(str)

        spl_name_map = spl_var["gene_name"].astype(str)
        spl_name_map.index = spl_var.index.astype(str)

        fallback = pd.Series(ds_index, index=ds_index)  # make it a Series, not an Index
        ds_gene_names = spl_name_map.reindex(ds_index).fillna(fallback).astype(str)
    else:
        ds_gene_names = pd.Series(sig_ds.index.astype(str), index=sig_ds.index)

    de_genes_set = set(de_gene_names.values.astype(str))
    ds_genes_set = set(ds_gene_names.values.astype(str))
    gene_overlap = sorted(de_genes_set.intersection(ds_genes_set))

    summary = {
        "n_sig_de": int(sig_de.shape[0]),
        "n_sig_ds": int(sig_ds.shape[0]),
        "n_overlap_genes": len(gene_overlap),
        "overlap_genes_preview": gene_overlap[:50],
    }
    say(logger, f"[summary] sig DE={summary['n_sig_de']} | sig DS={summary['n_sig_ds']} | gene-overlap={summary['n_overlap_genes']}")
    return summary, sig_de, sig_ds



def demo_overlays(model, mdata, m_ct, ad_rna, ct_key, target_celltype,
                  ds_sig_df, de_sig_df, batch_size_post, outdir: Path, logger: logging.Logger):
    if ds_sig_df.empty or de_sig_df.empty:
        say(logger, "[overlay] Not enough hits for demo overlays")
        return

    junc = ds_sig_df.index.tolist()[0]
    gene = de_sig_df.index.tolist()[0]
    say(logger, f"[overlay] Demo junction={junc} | gene={gene}")

    # Build the same target mask for overlays
    mask_ct = ad_rna.obs[ct_key] == target_celltype
    m_ct = m_ct[ad_rna.obs_names]

    # Get normalized values
    psi = model.get_normalized_splicing(adata=m_ct, batch_size=batch_size_post, return_numpy=True)
    expr = model.get_normalized_expression(adata=m_ct, batch_size=batch_size_post, return_numpy=True)

    # Plot directly on ad_rna view of the same cells
    ad_plot = ad_rna

    if junc in m_ct["splicing"].var_names:
        j_idx = m_ct["splicing"].var_names.get_loc(junc)
        ad_plot.obs[f"PSI::{junc}"] = psi[:, j_idx]
        say(logger, f"[overlay] Saving PSI overlay for {junc}")
        fig, ax = plt.subplots(figsize=(7, 6))
        sc.pl.embedding(ad_plot, basis="X_umap_joint", color=f"PSI::{junc}",
                        color_map="viridis", show=False, ax=ax, frameon=True)
        ax.set_title(f"{target_celltype} | PSI {junc}")
        plt.tight_layout()
        fig.savefig(outdir / f"umap_{target_celltype}_PSI_{junc}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    if gene in m_ct["rna"].var_names:
        g_idx = m_ct["rna"].var_names.get_loc(gene)
        ad_plot.obs[f"EXP::{gene}"] = expr[:, g_idx]
        say(logger, f"[overlay] Saving expression overlay for {gene}")
        fig, ax = plt.subplots(figsize=(7, 6))
        sc.pl.embedding(ad_plot, basis="X_umap_joint", color=f"EXP::{gene}",
                        color_map="viridis", show=False, ax=ax, frameon=True)
        ax.set_title(f"{target_celltype} | Expr {gene}")
        plt.tight_layout()
        fig.savefig(outdir / f"umap_{target_celltype}_Expr_{gene}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main():
    args = parse_args()

    # Create timestamped OUTDIR
    model_name = Path(args.model_dir).name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.base_outdir) / f"{stamp}__{model_name}"
    outdir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(outdir)

    say(logger, f"[env] scvi-tools {scvi.__version__} | scanpy {sc.__version__} | torch {torch.__version__}")
    say(logger, f"[args] model_dir={args.model_dir}")
    say(logger, f"[args] mudata_path={args.mudata_path}")
    say(logger, f"[args] outdir={outdir}")

    # Load MuData in memory so we can write to .obs/.obsm without implicit copies
    if not os.path.exists(args.mudata_path):
        say(logger, f"[error] MuData not found: {args.mudata_path}")
        sys.exit(1)
    say(logger, "[load] Reading MuData …")
    mdata = mu.read_h5mu(args.mudata_path)  # no backed mode
    say(logger, f"[load] Modalities: {list(mdata.mod.keys())}")

    # Choose celltype key but do not subset here
    ct_key_available = pick_celltype_key(mdata["rna"], args.preferred_celltype_col, args.fallback_celltype_col)
    say(logger, f"[subset] Using celltype column: {ct_key_available}")

    # Setup registry on full data
    say(logger, "[setup] Configuring scvi registries for MULTIVISPLICE")
    scvi.model.MULTIVISPLICE.setup_mudata(
        mdata,
        batch_key=None,
        size_factor_key="X_library_size",
        rna_layer="length_norm",
        junc_ratio_layer=args.x_layer,
        atse_counts_layer=args.cluster_counts_layer,
        junc_counts_layer=args.junction_counts_layer,
        psi_mask_layer=args.mask_layer,
        modalities={"rna_layer": "rna", "junc_ratio_layer": "splicing"},
    )

    plot_junction_coverage_vs_variance(
        mdata=mdata,
        x_layer=args.x_layer,
        junc_counts_layer=args.junction_counts_layer,
        mask_layer=args.mask_layer,
        outdir=outdir,
        logger=logger,
    )

    # Load model
    if not os.path.isdir(args.model_dir):
        say(logger, f"[error] Model directory not found: {args.model_dir}")
        sys.exit(1)
    say(logger, "[model] Loading trained MULTIVISPLICE model …")
    model = scvi.model.MULTIVISPLICE.load(args.model_dir, adata=mdata)
    say(logger, "[model] Loaded.")

    # Joint Leiden + UMAP (+ optional tSNE) inside function on the subset
    ad_rna = run_leiden_umap_joint(
        model=model,
        mdata=mdata,
        ct_key=ct_key_available,
        target_celltype=args.target_celltype,
        target_tissues=args.target_tissues,
        resolution=args.leiden_resolution,
        run_tsne=args.run_tsne,
        outdir=outdir,
        logger=logger,
    )
    ad_rna.uns["celltype_key"] = ct_key_available

    # Save cluster labels for the subset
    say(logger, "[save] Writing Leiden labels CSV")
    ad_rna.obs[["leiden_joint"]].to_csv(outdir / "leiden_joint_labels.csv")

    # Build the same subset for DE and DS
    mask_ct = mdata["rna"].obs[ct_key_available] == args.target_celltype
    if args.target_tissues and "tissue" in mdata["rna"].obs:
        tt = [t.lower() for t in args.target_tissues]
        mask_ct &= mdata["rna"].obs["tissue"].astype(str).str.lower().isin(tt)
    m_ct = mdata[mask_ct].copy()

    # Pairwise DE and DS inside the filtered population
    pairs, de_tables, ds_tables = run_pairwise_de_ds(
        model=model,
        m_ct=m_ct,
        ad_rna_ct=ad_rna,
        de_delta=args.de_delta,
        ds_delta=args.ds_delta,
        fdr=args.fdr,
        batch_size_post=args.batch_size_post,
        norm_splicing_function = args.norm_splicing_function,
        logger=logger,
    )

    # Summaries, CSV dumps
    summaries = {}
    sig_de_per_pair = {}
    sig_ds_per_pair = {}

    for key in pairs:
        say(logger, f"[summary] Pair {key}")
        s, de_sig, ds_sig = summarize_pair(de_tables[key], ds_tables[key], args.fdr, m_ct, logger)
        summaries[key] = s
        sig_de_per_pair[key] = de_sig
        sig_ds_per_pair[key] = ds_sig

        say(logger, f"[save] Writing tables for {key}")
        de_tables[key].to_csv(outdir / f"DE_{args.target_celltype}_{key}.csv")
        ds_tables[key].to_csv(outdir / f"DS_{args.target_celltype}_{key}.csv")
        de_sig.to_csv(outdir / f"DEsig_{args.target_celltype}_{key}.csv")
        ds_sig.to_csv(outdir / f"DSsig_{args.target_celltype}_{key}.csv")

    if summaries:
        pd.DataFrame(summaries).T.to_csv(outdir / "pairwise_summary.csv")
        say(logger, f"[save] Summary CSV → {outdir/'pairwise_summary.csv'}")
    else:
        say(logger, "[summary] No subcluster pairs found")

    # Demo overlays from first pair if available
    if pairs:
        say(logger, f"[overlay] Using first pair {pairs[0]} for demo overlays")
        demo_overlays(
            model, mdata, m_ct, ad_rna,
            ct_key=ct_key_available,
            target_celltype=args.target_celltype,
            ds_sig_df=sig_ds_per_pair[pairs[0]],
            de_sig_df=sig_de_per_pair[pairs[0]],
            batch_size_post=args.batch_size_post,
            outdir=outdir,
            logger=logger,
        )

    say(logger, "[done] Analysis complete.")


if __name__ == "__main__":
    main()
