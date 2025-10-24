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
    """Print to stdout (flushed) and log to file."""
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
    return p.parse_args()


def pick_celltype_key(ad_rna, preferred: str, fallback: str | None) -> str:
    if preferred in ad_rna.obs:
        return preferred
    if fallback and fallback in ad_rna.obs:
        return fallback
    raise ValueError("Could not find a cell type column in obs")


def subset_mudata(mdata, ct_key: str, target_celltype: str, target_tissues: list[str],
                  logger: logging.Logger):
    obs_rna = mdata["rna"].obs
    mask = obs_rna[ct_key] == target_celltype
    if target_tissues and "tissue" in obs_rna:
        tt = [t.lower() for t in target_tissues]
        mask &= obs_rna["tissue"].astype(str).str.lower().isin(tt)
    n_keep = int(mask.sum())
    if n_keep == 0:
        raise ValueError(f"No cells matched '{target_celltype}' with tissues={target_tissues}")
    say(logger, f"[subset] Keeping {n_keep} cells for '{target_celltype}', tissues={target_tissues}")
    return mdata[mask].copy()


def run_leiden_umap_joint(model, mdata, resolution: float, run_tsne: bool,
                          outdir: Path, logger: logging.Logger):
    say(logger, "[latent] Computing joint latent, neighbors, Leiden, UMAP" + (" + tSNE" if run_tsne else ""))
    ad_rna = mdata["rna"].copy()

    Z_joint = model.get_latent_representation(adata=mdata, modality="joint")
    ad_rna.obsm["X_latent_joint"] = Z_joint
    say(logger, f"[latent] Z_joint shape: {Z_joint.shape}")

    sc.pp.neighbors(ad_rna, use_rep="X_latent_joint", key_added="neighbors_joint")
    sc.tl.leiden(ad_rna, neighbors_key="neighbors_joint", key_added="leiden_joint", resolution=resolution)

    sc.tl.umap(ad_rna, neighbors_key="neighbors_joint")
    ad_rna.obsm["X_umap_joint"] = ad_rna.obsm["X_umap"].copy()

    if run_tsne:
        sc.tl.tsne(ad_rna, use_rep="X_latent_joint", n_pcs=0, learning_rate=200.0,
                   perplexity=30.0, random_state=0)
        ad_rna.obsm["X_tsne_joint"] = ad_rna.obsm["X_tsne"].copy()

    ncl = int(ad_rna.obs["leiden_joint"].nunique())
    say(logger, f"[leiden] Found {ncl} clusters at res={resolution}")

    # Plots
    say(logger, "[plot] Saving UMAP colored by Leiden")
    fig, ax = plt.subplots(figsize=(8, 6))
    sc.pl.embedding(ad_rna, basis="X_umap_joint", color="leiden_joint",
                    legend_loc="right margin", show=False, ax=ax, frameon=True)
    ax.set_title(f"Joint UMAP — Leiden (res={resolution})")
    plt.tight_layout()
    fig.savefig(outdir / "umap_joint_leiden.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    if run_tsne:
        say(logger, "[plot] Saving tSNE colored by Leiden")
        fig, ax = plt.subplots(figsize=(8, 6))
        sc.pl.embedding(ad_rna, basis="X_tsne_joint", color="leiden_joint",
                        legend_loc="right margin", show=False, ax=ax, frameon=True)
        ax.set_title(f"Joint tSNE — Leiden (res={resolution})")
        plt.tight_layout()
        fig.savefig(outdir / "tsne_joint_leiden.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    return ad_rna


def run_pairwise_de_ds(model, m_ct, ad_rna_ct, de_delta, ds_delta, fdr,
                       batch_size_post, logger: logging.Logger):
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
                adata=m_ct["rna"],
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
                adata=m_ct["splicing"],
                idx1=idx1,
                idx2=idx2,
                mode="change",
                delta=ds_delta,
                fdr_target=fdr,
                batch_size=batch_size_post,
                all_stats=True,
                silent=True,
            )

            pairs.append(key)
            de_tables[key] = de_df
            ds_tables[key] = ds_df
            say(logger, f"[pairwise] {key}: done")

    say(logger, "[pairwise] All pairwise comparisons complete")
    return pairs, de_tables, ds_tables


def summarize_pair(de_df, ds_df, fdr, mdata, logger: logging.Logger,
                   gene_col_candidates=("gene_id", "gene_name")):
    col_prob_de = "proba_de" if "proba_de" in de_df.columns else ("probability" if "probability" in de_df.columns else None)
    col_prob_ds = "proba_ds" if "proba_ds" in ds_df.columns else ("proba_de" if "proba_de" in ds_df.columns else None)
    if col_prob_de is None or col_prob_ds is None:
        raise RuntimeError("Could not find probability columns for DE or DS output")

    sig_de = de_df[de_df[col_prob_de] >= 1 - fdr].copy()
    sig_ds = ds_df[ds_df[col_prob_ds] >= 1 - fdr].copy()

    sp_var = mdata["splicing"].var
    gene_col = next((c for c in gene_col_candidates if c in sp_var.columns), None)
    if gene_col is not None:
        sig_ds_genes = sp_var.loc[sig_ds.index, gene_col].astype(str)
    else:
        sig_ds_genes = pd.Series(index=sig_ds.index, data=["NA"] * len(sig_ds))

    de_genes = set(sig_de.index.astype(str))
    ds_genes_set = set(sig_ds_genes.astype(str))
    gene_overlap = sorted(de_genes.intersection(ds_genes_set))

    summary = {
        "n_sig_de": int(sig_de.shape[0]),
        "n_sig_ds": int(sig_ds.shape[0]),
        "n_overlap_genes": len(gene_overlap),
        "overlap_genes_preview": gene_overlap[:50],
    }
    say(logger, f"[summary] sig DE={summary['n_sig_de']} | sig DS={summary['n_sig_ds']} | gene-overlap={summary['n_overlap_genes']}")
    return summary, sig_de, sig_ds


def demo_overlays(model, mdata, ad_rna, ct_key, target_celltype,
                  ds_sig_df, de_sig_df, batch_size_post, outdir: Path, logger: logging.Logger):
    if ds_sig_df.empty or de_sig_df.empty:
        say(logger, "[overlay] Not enough hits for demo overlays")
        return

    junc = ds_sig_df.index.tolist()[0]
    gene = de_sig_df.index.tolist()[0]
    say(logger, f"[overlay] Demo junction={junc} | gene={gene}")

    m_ct = mdata[ad_rna.obs[ct_key] == target_celltype].copy()
    psi = model.get_normalized_splicing(adata=m_ct, batch_size=batch_size_post, return_numpy=True)
    expr = model.get_normalized_expression(adata=m_ct, batch_size=batch_size_post, return_numpy=True)

    ad_plot = ad_rna[ad_rna.obs[ct_key] == target_celltype].copy()

    j_idx = m_ct["splicing"].var_names.get_loc(junc) if junc in m_ct["splicing"].var_names else None
    g_idx = m_ct["rna"].var_names.get_loc(gene) if gene in m_ct["rna"].var_names else None

    if j_idx is not None:
        ad_plot.obs[f"PSI::{junc}"] = psi[:, j_idx]
        say(logger, f"[overlay] Saving PSI overlay for {junc}")
        fig, ax = plt.subplots(figsize=(7, 6))
        sc.pl.embedding(ad_plot, basis="X_umap_joint", color=f"PSI::{junc}",
                        color_map="viridis", show=False, ax=ax, frameon=True)
        ax.set_title(f"{target_celltype} — PSI {junc}")
        plt.tight_layout()
        fig.savefig(outdir / f"umap_{target_celltype}_PSI_{junc}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    if g_idx is not None:
        ad_plot.obs[f"EXP::{gene}"] = expr[:, g_idx]
        say(logger, f"[overlay] Saving expression overlay for {gene}")
        fig, ax = plt.subplots(figsize=(7, 6))
        sc.pl.embedding(ad_plot, basis="X_umap_joint", color=f"EXP::{gene}",
                        color_map="viridis", show=False, ax=ax, frameon=True)
        ax.set_title(f"{target_celltype} — Expr {gene}")
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

    # Load MuData
    if not os.path.exists(args.mudata_path):
        say(logger, f"[error] MuData not found: {args.mudata_path}")
        sys.exit(1)
    say(logger, "[load] Reading MuData …")
    mdata = mu.read_h5mu(args.mudata_path, backed="r")
    say(logger, f"[load] Modalities: {list(mdata.mod.keys())}")

    # Subset to target type before setup
    ct_key_available = pick_celltype_key(mdata["rna"], args.preferred_celltype_col, args.fallback_celltype_col)
    say(logger, f"[subset] Using celltype column: {ct_key_available}")
    mdata = subset_mudata(mdata, ct_key_available, args.target_celltype, args.target_tissues, logger)

    # Setup registry
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

    # Load model
    if not os.path.isdir(args.model_dir):
        say(logger, f"[error] Model directory not found: {args.model_dir}")
        sys.exit(1)
    say(logger, "[model] Loading trained MULTIVISPLICE model …")
    model = scvi.model.MULTIVISPLICE.load(args.model_dir, mdata=mdata)
    say(logger, "[model] Loaded.")

    # Joint Leiden + UMAP (+ optional tSNE)
    ad_rna = run_leiden_umap_joint(
        model, mdata,
        resolution=args.leiden_resolution,
        run_tsne=args.run_tsne,
        outdir=outdir,
        logger=logger,
    )
    ad_rna.uns["celltype_key"] = ct_key_available

    # Save cluster labels
    say(logger, "[save] Writing Leiden labels CSV")
    ad_rna.obs[["leiden_joint"]].to_csv(outdir / "leiden_joint_labels.csv")

    # Pairwise DE and DS inside the filtered population
    pairs, de_tables, ds_tables = run_pairwise_de_ds(
        model=model,
        m_ct=mdata,
        ad_rna_ct=ad_rna,
        de_delta=args.de_delta,
        ds_delta=args.ds_delta,
        fdr=args.fdr,
        batch_size_post=args.batch_size_post,
        logger=logger,
    )

    # Summaries, CSV dumps
    summaries = {}
    sig_de_per_pair = {}
    sig_ds_per_pair = {}

    for key in pairs:
        say(logger, f"[summary] Pair {key}")
        s, de_sig, ds_sig = summarize_pair(de_tables[key], ds_tables[key], args.fdr, mdata, logger)
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
            model, mdata, ad_rna,
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
