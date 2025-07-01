#!/usr/bin/env python3
# latentspace_multivisplice.py

import os
import argparse
import logging
import numpy as np
import pandas as pd
import mudata as mu
import scvi
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from scipy import sparse
from scipy.stats import spearmanr
from sklearn.metrics import (
    silhouette_score, precision_score, recall_score, f1_score,
    confusion_matrix, accuracy_score
)
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay

def parse_args():
    p = argparse.ArgumentParser(
        description="Latent‐space evaluation for MULTIVISPLICE"
    )
    p.add_argument("--mudata_path", type=str, required=True,
                   help="Input .h5mu")
    p.add_argument("--model_path", type=str, required=True,
                   help="Directory with trained model")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Base output directory")
    p.add_argument("--cluster_numbers", nargs="+", type=int,
                   default=[3,5,10], help="k's for subcluster eval")
    p.add_argument("--neighbor_k", nargs="+", type=int,
                   default=[30], help="k's for neighborhood overlap")
    p.add_argument("--cell_type_column", type=str, default="broad_cell_type",
                   help="Cell-type column in obs")
    p.add_argument("--top_n_celltypes", type=int, default=5,
                   help="Number of top cell types to highlight")
    p.add_argument("--umap_cell_label", nargs="+",
                   default=["broad_cell_type"], help="obs columns to color UMAP by")
    p.add_argument("--n_pcs", type=int, default=5,
                   help="Number of PCs for modality contribution")
    return p.parse_args()

def setup_dirs(base):
    overall = os.path.join(base, "overall")
    subcluster = os.path.join(base, "subcluster_eval")
    classif = os.path.join(base, "cell_type_classification")
    neighbor = os.path.join(base, "local_neighborhood_analysis")
    for d in [overall, subcluster, classif, neighbor]:
        os.makedirs(os.path.join(d, "figures"), exist_ok=True)
        os.makedirs(os.path.join(d, "csv_files"),  exist_ok=True)
    os.makedirs(os.path.join(subcluster, "figures", "line_plots"), exist_ok=True)
    os.makedirs(os.path.join(subcluster, "figures", "confusion_matrices"), exist_ok=True)
    os.makedirs(os.path.join(subcluster, "figures", "umaps"), exist_ok=True)
    return overall, subcluster, classif, neighbor

def plot_umap(adata, rep_name, color_by, out_dir, top_n=None):
    logger.info(f"UMAP ({rep_name}) colored by {color_by}")
    sc.pp.neighbors(adata, use_rep=rep_name, n_neighbors=15)
    sc.tl.umap(adata, min_dist=0.1)
    if top_n:
        top = adata.obs[color_by].value_counts().head(top_n).index.tolist()
        adata.obs["highlighted"] = np.where(
            adata.obs[color_by].isin(top), adata.obs[color_by], "Other"
        )
        cmap = cm.get_cmap("tab20", len(top))
        pal = {grp: cmap(i) for i, grp in enumerate(top)}
        pal["Other"] = (0.9,0.9,0.9,1)
        sc.pl.umap(adata, color="highlighted", palette=pal, show=False)
    else:
        sc.pl.umap(adata, color=color_by, show=False)
    plt.savefig(os.path.join(out_dir, f"umap_{rep_name}_{color_by}.pdf"))
    plt.close()

def plot_subcluster_umap(Z, true_labels, preds_dict, prefix, out_dir):
    # Save one UMAP per label set
    for name, labels in preds_dict.items():
        ad = sc.AnnData(Z)
        ad.obs["true"] = true_labels.astype(str)
        ad.obsm["X_latent"] = Z
        sc.pp.neighbors(ad, use_rep="X_latent", n_neighbors=15)
        sc.tl.umap(ad, min_dist=0.1)
        fig, ax = plt.subplots(figsize=(6,6))
        sc.pl.umap(ad, color="true", ax=ax, show=False)
        fig.savefig(os.path.join(out_dir, f"{prefix}_true_umap.pdf"))
        plt.close(fig)
        ad.obs[name] = labels.astype(str)
        fig, ax = plt.subplots(figsize=(6,6))
        sc.pl.umap(ad, color=name, ax=ax, show=False)
        fig.savefig(os.path.join(out_dir, f"{prefix}_{name}_umap.pdf"))
        plt.close(fig)

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    global logger
    logger = logging.getLogger("latent_eval")

    # Directories
    overall_dir, subcluster_dir, classif_dir, neighbor_dir = setup_dirs(args.out_dir)

    # Load data & model
    logger.info("Loading MuData and model")
    mdata = mu.read_h5mu(args.mudata_path)
    # ensure mask for splicing
    layer = mdata["splicing"].layers["cell_by_cluster_matrix"]
    mat = sparse.csr_matrix(layer) if not sparse.isspmatrix_csr(layer) else layer
    mask = mat.copy(); mask.data = np.ones_like(mask.data, dtype=np.uint8)
    mdata["splicing"].layers["psi_mask"] = mask
    model = scvi.model.MULTIVISPLICE.load(args.model_path, adata=mdata)

    # Extract latent representations
    logger.info("Computing latent representations")
    Z_joint = model.get_latent_representation()
    Z_ge    = model.get_latent_representation(modality="rna")
    Z_as    = model.get_latent_representation(modality="junc_counts")
    labels_ct = mdata["rna"].obs[args.cell_type_column].astype("category")

    # --- 1) Overall UMAP & silhouette ---
    logger.info("[Overall] UMAP & silhouette")
    sil_records = []
    for rep, Z in [("joint", Z_joint), ("ge", Z_ge), ("as", Z_as)]:
        sil = silhouette_score(Z, labels_ct.cat.codes.values)
        sil_records.append({"rep": rep, "silhouette": sil})
        ad = sc.AnnData(Z)
        ad.obs[args.cell_type_column] = labels_ct.values
        plot_umap(ad, "X", args.cell_type_column, os.path.join(overall_dir, "figures"), args.top_n_celltypes)
    pd.DataFrame(sil_records).to_csv(os.path.join(overall_dir, "csv_files", "silhouette_scores.csv"), index=False)

    # --- 1.5) Modality contributions to PCs ---
    logger.info(f"[Contribution] First {args.n_pcs} PCs")
    d = Z_ge.shape[1]
    pca = PCA(n_components=args.n_pcs, random_state=42)
    pca.fit(Z_joint)
    loadings = pca.components_
    contrib = []
    for i, comp in enumerate(loadings, start=1):
        abs_comp = np.abs(comp)
        expr_sum   = abs_comp[:d].sum()
        splice_sum = abs_comp[d:].sum()
        contrib.append({"PC": i, "expr": expr_sum, "splice": splice_sum})
    pd.DataFrame(contrib).to_csv(os.path.join(overall_dir, "csv_files", "pc_contributions.csv"), index=False)

    # --- 2) Subcluster reproducibility ---
    logger.info("[Subcluster] Starting")
    rec_sub = []
    for ct in labels_ct.cat.categories:
        idx = np.where(labels_ct == ct)[0]
        if len(idx) < 2: continue
        split = len(idx)//2
        idx1, idx2 = idx[:split], idx[split:]
        Z1_joint = Z_joint[idx1]
        Z2_joint = Z_joint[idx2]
        for k in args.cluster_numbers:
            km = KMeans(n_clusters=k, random_state=42)
            lab1 = km.fit_predict(Z1_joint)
            lab2 = km.predict(Z2_joint)
            preds = {}
            for rep, Z in [("joint", Z_joint), ("ge", Z_ge), ("as", Z_as)]:
                clf = LogisticRegression(max_iter=200).fit(Z[idx2], lab2)
                p = clf.predict(Z[idx1])
                preds[rep] = p
                for metric, fn in [
                    ("accuracy", lambda a,p: (p==a).mean()),
                    ("f1", lambda a,p: f1_score(a,p,average="weighted",zero_division=0))
                ]:
                    rec_sub.append({
                        "cell_type": ct, "rep": rep, "k": k,
                        "metric": metric, "value": fn(lab1, p)
                    })
            # random baseline
            rng = np.random.default_rng(42)
            Zr = rng.standard_normal(Z_joint.shape)
            clf = LogisticRegression(max_iter=200).fit(Zr[idx2], lab2)
            pr = clf.predict(Zr[idx1])
            preds["random"] = pr
            for metric, fn in [
                ("accuracy", lambda a,p: (p==a).mean()),
                ("f1", lambda a,p: f1_score(a,p,average="weighted",zero_division=0))
            ]:
                rec_sub.append({
                    "cell_type": ct, "rep": "random", "k": k,
                    "metric": metric, "value": fn(lab1, pr)
                })
            # UMAPs for selected types
            if ct in ["Excitatory Neurons", "MICROGLIA"]:
                plot_subcluster_umap(Z1_joint, lab1, preds, f"{ct}_k{k}", os.path.join(subcluster_dir, "figures", "umaps"))
    pd.DataFrame(rec_sub).to_csv(os.path.join(subcluster_dir, "csv_files", "subcluster_metrics.csv"), index=False)

    # --- 3) Cell‐type classification ---
    logger.info("[Classification] Starting")
    rec_cls = []
    y_all = labels_ct.cat.codes.values
    for rep, Z in [("joint", Z_joint), ("ge", Z_ge), ("as", Z_as)]:
        Xtr, Xte, ytr, yte = train_test_split(Z, y_all, test_size=0.2, stratify=y_all, random_state=42)
        clf = LogisticRegression(max_iter=500).fit(Xtr, ytr)
        yp = clf.predict(Xte)
        for idx, ct in enumerate(labels_ct.cat.categories):
            mask = yte == idx
            if mask.sum():
                rec_cls.append({
                    "rep": rep, "cell_type": ct,
                    "accuracy": (yp[mask]==yte[mask]).mean(),
                    "f1": f1_score(yte[mask], yp[mask], average="macro", zero_division=0)
                })
        rec_cls.append({
            "rep": rep, "cell_type": "ALL",
            "accuracy": accuracy_score(yte, yp),
            "f1": f1_score(yte, yp, average="macro", zero_division=0)
        })
    pd.DataFrame(rec_cls).to_csv(os.path.join(classif_dir, "csv_files", "cell_type_classification.csv"), index=False)

    # --- 4) Local neighborhood consistency ---
    logger.info("[Neighborhood] Starting")
    rec_nb = []
    nbr_joint = NearestNeighbors(n_neighbors=max(args.neighbor_k)+1).fit(Z_joint)
    idxj = nbr_joint.kneighbors(return_distance=False)[:,1:]
    for k in args.neighbor_k:
        nbr_ge = NearestNeighbors(n_neighbors=k+1).fit(Z_ge)
        idxge = nbr_ge.kneighbors(return_distance=False)[:,1:]
        nbr_as = NearestNeighbors(n_neighbors=k+1).fit(Z_as)
        idxas = nbr_as.kneighbors(return_distance=False)[:,1:]
        for i in range(Z_joint.shape[0]):
            s_ge = len(set(idxj[i][:k]) & set(idxge[i])) / k
            s_as = len(set(idxj[i][:k]) & set(idxas[i])) / k
            rec_nb.append({"cell": i, "k": k, "S_GE": s_ge, "S_AS": s_as})
    df_nb = pd.DataFrame(rec_nb)
    df_nb.to_csv(os.path.join(neighbor_dir, "csv_files", "neighborhood_overlap.csv"), index=False)
    rho, pval = spearmanr(df_nb["S_GE"], df_nb["S_AS"])
    logger.info(f"Spearman ρ: {rho:.3f}, p={pval:.3g}")
    for metric in ["S_GE", "S_AS"]:
        fig, ax = plt.subplots()
        sns.histplot(df_nb[metric], kde=True, ax=ax)
        fig.savefig(os.path.join(neighbor_dir, "figures", f"{metric}_hist.png"), dpi=300)
        plt.close(fig)
        ad = sc.AnnData(Z_joint)
        ad.obs[metric] = df_nb[metric].values
        plot_umap(ad, "X", metric, neighbor_dir)
    fig, ax = plt.subplots()
    sns.scatterplot(x="S_GE", y="S_AS", data=df_nb, ax=ax)
    fig.savefig(os.path.join(neighbor_dir, "figures", "scatter_SGE_SAS.png"), dpi=300)
    logger.info("[Neighborhood] Done")

if __name__ == "__main__":
    main()
