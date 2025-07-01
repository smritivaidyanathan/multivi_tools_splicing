#!/usr/bin/env python3
# weights_multivisplice.py (updated)

import os
import argparse
import logging
import mudata as mu
import scvi
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
from scipy.cluster.hierarchy import linkage, dendrogram


def parse_args():
    p = argparse.ArgumentParser(description="Weight analysis for MULTIVISPLICE")
    p.add_argument("--mudata_path", type=str, required=True,
                   help="Input .h5mu")
    p.add_argument("--model_path", type=str, required=True,
                   help="Directory with trained model")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Base output directory")
    p.add_argument("--umap_labels", nargs="+", default=["broad_cell_type","dataset"],
                   help="obs columns to color UMAP by")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("weights_analysis")

    print("[info] Starting MULTIVISPLICE weight analysis")
    print(f"[info] MuData path: {args.mudata_path}")
    print(f"[info] Model path: {args.model_path}")
    print(f"[info] Output directory: {args.out_dir}")

    fig_dir = os.path.join(args.out_dir, "figures"); os.makedirs(fig_dir, exist_ok=True)
    csv_dir = os.path.join(args.out_dir, "csv_files"); os.makedirs(csv_dir, exist_ok=True)

    print("[info] Loading MuData and model...")
    mdata = mu.read_h5mu(args.mudata_path)
    model = scvi.model.MULTIVISPLICE.load(args.model_path, adata=mdata)
    print("[info] Model loaded successfully")

    # ── Extra: modality‐weight boxplots by obs‐label ──────────────────────
    if model.modality_weights == "cell":
        print("[info] Generating splicing‐weight box plots...")
        raw = model.module.mod_weights.detach()
        mix = F.softmax(raw, dim=1).cpu().numpy()
        w_splice = mix[:, 1]
        df = mdata["rna"].obs[args.umap_labels].copy()
        df["w_splicing"] = w_splice

        for label in args.umap_labels:
            print(f"[info] Plotting boxplot for label: {label}")
            fig, ax = plt.subplots(figsize=(4, 4))
            sns.boxplot(
                x=label,
                y="w_splicing",
                data=df,
                ax=ax,
                showfliers=False,
            )
            ax.set_title(f"Splicing weight by {label}")
            ax.set_ylabel("splicing weight (softmaxed)")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            filename = f"w_splicing_by_{label}.png"
            out_path = os.path.join(fig_dir, filename)
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"[info] Saved splicing‐weight boxplot for '{label}' → {out_path}")
            wandb.log({f"w_splicing_by_{label}": wandb.Image(out_path)})

    # ── If using concatenation, export one heatmap per decoder ──
    if model.module.modality_weights == "concatenate":
        print("[info] Generating decoder heatmaps...")
        out_heat = os.path.join(fig_dir, "decoder_heatmaps")
        os.makedirs(out_heat, exist_ok=True)

        # Expression Decoder weights
        expr_dec = model.module.z_decoder_expression
        lin_expr = expr_dec.factor_regressor.fc_layers[0][0]
        W_expr = lin_expr.weight.detach().cpu().numpy()
        print("[info] Got expression decoder weights (shape: {} )".format(W_expr.shape))

        # Splicing Decoder weights
        spl_dec = model.module.z_decoder_splicing
        lin_spl = spl_dec.linear
        W_spl = lin_spl.weight.detach().cpu().numpy()
        print("[info] Got splicing decoder weights (shape: {} )".format(W_spl.shape))

        def _plot_heatmap(W, xlabel, ylabel, title, fname):
            fig, ax = plt.subplots(figsize=(12, 6))
            im = ax.imshow(np.abs(W).T, aspect="auto")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            fig.colorbar(im, ax=ax, label="|weight|")
            path = os.path.join(out_heat, fname)
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"[info] Saved {title} → {path}")

        _plot_heatmap(
            W_expr,
            "Gene index",
            "Latent dimension",
            "Expression Decoder |weights|",
            "expression_decoder_heatmap.png",
        )
        _plot_heatmap(
            W_spl,
            "Junction index",
            "Latent dimension + covariates",
            "Splicing Decoder |weights|",
            "splicing_decoder_heatmap.png",
        )

        # ── Hierarchical clustering of features ──
        cluster_dir = os.path.join(fig_dir, "hierarchical_clustering")
        os.makedirs(cluster_dir, exist_ok=True)

        # Genes
        print("[info] Performing hierarchical clustering on gene weights...")
        Wg = np.abs(W_expr)
        link_genes = linkage(Wg, method="average", metric="correlation")
        fig, ax = plt.subplots(figsize=(8, 6))
        dendrogram(link_genes, no_labels=True)
        ax.set_title("Hierarchical clustering of genes based on |weights|")
        plt.tight_layout()
        gene_clust_path = os.path.join(cluster_dir, "hierarchical_clustering_genes.png")
        fig.savefig(gene_clust_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[info] Saved gene clustering dendrogram → {gene_clust_path}")

        # Junctions
        print("[info] Performing hierarchical clustering on junction weights...")
        Wj = np.abs(W_spl)
        link_junc = linkage(Wj, method="average", metric="correlation")
        fig, ax = plt.subplots(figsize=(8, 6))
        dendrogram(link_junc, no_labels=True)
        ax.set_title("Hierarchical clustering of junctions based on |weights|")
        plt.tight_layout()
        junc_clust_path = os.path.join(cluster_dir, "hierarchical_clustering_junctions.png")
        fig.savefig(junc_clust_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[info] Saved junction clustering dendrogram → {junc_clust_path}")

if __name__ == "__main__":
    main()
