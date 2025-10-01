#!/usr/bin/env python
import os
import inspect
import argparse

import scanpy as sc
import scvi
import wandb
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
from scipy import sparse
import torch
from sklearn.neighbors import NearestNeighbors

# ------------------------------
# 0. Default Paths (CLI-overridable)
# ------------------------------
DEFAULT_ANN_DATA = "/gpfs/commons/groups/knowles_lab/Karin/TMS_MODELING/DATA_FILES/SIMULATED/simulated_data_2025-03-12.h5ad"
DEFAULT_MODEL_DIR = "/gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/models"
DEFAULT_FIG_DIR = "/gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/figures"

# ------------------------------
# 1. Grab train() defaults
# ------------------------------
train_sig = inspect.signature(scvi.model.MULTIVISPLICE.train)
train_defaults = {
    name: param.default
    for name, param in train_sig.parameters.items()
    if name != 'self' and param.default is not inspect._empty
}

# ------------------------------
# 2. Grab __init__ defaults for model
# ------------------------------
init_sig = inspect.signature(scvi.model.MULTIVISPLICE.__init__)
init_defaults = {
    name: param.default
    for name, param in init_sig.parameters.items()
    if name not in ('self', 'adata') and param.default is not inspect._empty
}

# ------------------------------
# 3. Build argparse
# ------------------------------
parser = argparse.ArgumentParser("SpliceVI-MultiVI")
# paths
parser.add_argument(
    "--train_mdata_path", type=str, default=DEFAULT_ANN_DATA,
    help=f"Train MuData (.h5ad) input path (default: {DEFAULT_ANN_DATA})"
)
parser.add_argument(
    "--test_mdata_path", type=str, default=DEFAULT_ANN_DATA,
    help=f"Test MuData (.h5ad) input path (default: {DEFAULT_ANN_DATA})"
)
parser.add_argument(
    "--masked_test_mdata_path", type=str, default=DEFAULT_ANN_DATA,
    help=f"Test MuData (.h5ad) input path (default: {DEFAULT_ANN_DATA})"
)
parser.add_argument(
    "--model_dir", type=str, default=DEFAULT_MODEL_DIR,
    help=f"Directory to save trained model (default: {DEFAULT_MODEL_DIR})"
)
parser.add_argument(
    "--fig_dir", type=str, default=DEFAULT_FIG_DIR,
    help=f"Directory to save UMAP figures (default: {DEFAULT_FIG_DIR})"
)


# model init params
for name, default in init_defaults.items():
    arg_type = type(default) if default is not None else float
    parser.add_argument(
        f"--{name}", type=arg_type, default=None,
        help=f"{name} (default = {default!r})"
    )
# training params
for name, default in train_defaults.items():
    arg_type = type(default) if default is not None else float
    parser.add_argument(
        f"--{name}", type=arg_type, default=None,
        help=f"{name} (default = {default!r})"
    )
# UMAP color fields
parser.add_argument(
    "--umap_colors", nargs='+', default=["cell_type_grouped"],
    help="List of obs fields to color UMAP by"
)

args = parser.parse_args()

# ------------------------------
# 4. Prepare directories
# ------------------------------
os.makedirs(args.model_dir, exist_ok=True)
os.makedirs(args.fig_dir, exist_ok=True)

# ------------------------------
# 5. Initialize W&B
# ------------------------------
full_config = {**init_defaults, **train_defaults}
for key in list(full_config):
    val = getattr(args, key)
    if val is not None:
        full_config[key] = val
full_config.update({
    "train_mdata_path": args.train_mdata_path,
    "test_mdata_path": args.test_mdata_path,
    "model_dir": args.model_dir,
    "fig_dir": args.fig_dir,
    "umap_colors": args.umap_colors,
})
wandb.init(project="MLCB_SUBMISSION", config=full_config)
wandb_logger = WandbLogger(project="MLCB_SUBMISSION", config=full_config)

# ------------------------------
# 6. Load Training MuData & Preprocess
# ------------------------------
print(f"Loading Training MuData from {args.train_mdata_path}…")
import mudata as mu
mdata = mu.read_h5mu(args.train_mdata_path, backed = "r")

# Layer names
x_layer = "junc_ratio"
junction_counts_layer = "cell_by_junction_matrix"
cluster_counts_layer = "cell_by_cluster_matrix"
mask_layer = "psi_mask"

print(f"Found layers in training splicing modality AnnData: {list(mdata["splicing"].layers.keys())}")

print("Setting up SpliceVI PartialVAE…")

scvi.model.MULTIVISPLICE.setup_mudata(
    mdata,
    batch_key = None,
    size_factor_key="X_library_size",
    rna_layer="length_norm",
    junc_ratio_layer=x_layer,
    atse_counts_layer=cluster_counts_layer,
    junc_counts_layer=junction_counts_layer,
    psi_mask_layer=mask_layer,
    modalities={"rna_layer": "rna", "junc_ratio_layer": "splicing"},
)

# ------------------------------
# 7. Initialize model
# ------------------------------

if getattr(args, "n_latent", None) is not None:
    args.n_latent = int(args.n_latent)
    
model_kwargs = {
    name: getattr(args, name)
    for name in init_defaults
    if getattr(args, name) is not None
}

print("Initializing model with:", model_kwargs)
model = scvi.model.MULTIVISPLICE(
    mdata,
    n_genes=(mdata["rna"].var["modality"] == "Gene_Expression").sum(),
    n_junctions=(mdata["splicing"].var["modality"] == "Splicing").sum(),
    **model_kwargs,
)
model.view_anndata_setup()

wandb.watch(
    model.module,          # the torch.nn.Module you want to instrument
    log="all",             # you can choose "gradients", "parameters", or "all"
    log_freq=1000,         # how often (in steps) to log
    log_graph=False        # True if you also want to log the computational graph
)

# count & log total parameters
total_params = sum(p.numel() for p in model.module.parameters())
print(f"Total model parameters: {total_params:,}")
wandb.log({"total_parameters": total_params})


# ------------------------------
# 8. Train
# ------------------------------
train_kwargs = {
    name: getattr(args, name)
    for name in train_defaults
    if getattr(args, name) is not None
}
print("Starting training with settings:", train_kwargs)
model.train(logger=wandb_logger, check_val_every_n_epoch=5, **train_kwargs)
model.save(args.model_dir, overwrite=True)
wandb.log({"model_saved_to": args.model_dir})

# ------------------------------
# 9. Compute UMAP
# ------------------------------
print("Computing latent representations and UMAPs…")

umap_color_key = "broad_cell_type"
cell_type_classification_key = "medium_cell_type" if "medium_cell_type" in mdata.obs else "broad_cell_type"

latent_spaces = {
    "joint":      model.get_latent_representation(),
    "expression": model.get_latent_representation(modality="expression"),
    "splicing":   model.get_latent_representation(modality="splicing"),
}

for name, Z in latent_spaces.items():
    key_latent = f"X_latent_{name}"
    key_nn     = f"neighbors_{name}"
    key_umap   = f"X_umap_{name}"   # store in .obsm under this name

    # write latent, build neighbors with a custom key
    mdata["rna"].obsm[key_latent] = Z
    sc.pp.neighbors(mdata["rna"], use_rep=key_latent, key_added=key_nn)

    # run UMAP; no key_added here. Use copy=True and pull out the embedding.
    sc.tl.umap(mdata["rna"], min_dist=0.1, neighbors_key=key_nn)
    mdata["rna"].obsm[key_umap] = mdata["rna"].obsm["X_umap"]

    print(f"Generating UMAP for latent space: {name}")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(13, 6))
    sc.pl.embedding(
        mdata["rna"],
        basis=key_umap,                 # <- custom basis lives in .obsm[key_umap]
        color=umap_color_key,
        legend_loc=None,
        frameon=True,
        legend_fontsize=10,
        show=False,
    )
    plt.title(f"UMAP by {umap_color_key} – {name}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = f"{args.fig_dir}/umap_{umap_color_key}_{name}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    wandb.log({f"umap_{umap_color_key}_{name}": wandb.Image(out_path)})
    plt.close()


    
    plt.figure(figsize=(6, 6))
    sc.pl.embedding(
        mdata["rna"],
        basis=key_umap,                 # <- custom basis lives in .obsm[key_umap]
        color=umap_color_key,
        legend_loc=None,
        frameon=True,
        legend_fontsize=10,
        show=False,
    )
    plt.title(f"UMAP by {umap_color_key} – {name}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = f"{args.fig_dir}/umap_{umap_color_key}_{name}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    wandb.log({f"umap_sqr_{umap_color_key}_{name}": wandb.Image(out_path)})
    plt.close()

    if cell_type_classification_key != umap_color_key:
        plt.figure(figsize=(13, 6))
        sc.pl.embedding(
            mdata["rna"],
            basis=key_umap,                 # <- custom basis lives in .obsm[key_umap]
            color=cell_type_classification_key,
            legend_loc="right margin",
            frameon=True,
            legend_fontsize=10,
            show=False,
        )
        plt.title(f"UMAP by {cell_type_classification_key} – {name}")
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        out_path = f"{args.fig_dir}/umap_{cell_type_classification_key}_{name}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        wandb.log({f"umap_{cell_type_classification_key}_{name}": wandb.Image(out_path)})
        plt.close()



print("All UMAP embeddings complete.")


# ------------------------------
# 9b. Unsupervised clustering and cross-space consistency
# ------------------------------
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_mutual_info_score

LEIDEN_RESOLUTION = 1.0
NN_OVERLAPS_KS = [2, 5, 15, 50, 100]

print("Running Leiden clustering on each latent space and computing consistency…")

# Helper: build neighbors and Leiden using the already-computed latent embedding stored in obsm
def run_leiden_on_basis(ad, basis_key: str, neigh_key: str, leiden_key: str, resolution: float):
    sc.pp.neighbors(ad, use_rep=basis_key, key_added=neigh_key)
    sc.tl.leiden(ad, neighbors_key=neigh_key, key_added=leiden_key, resolution=resolution)

# Compute Leiden per latent space with independent neighbor graphs
leiden_keys = {}
for name in ["joint", "expression", "splicing"]:
    basis_key = f"X_latent_{name}"
    neigh_key = f"neighbors_{name}_leiden"
    leiden_key = f"leiden_{name}"
    run_leiden_on_basis(mdata["rna"], basis_key, neigh_key, leiden_key, LEIDEN_RESOLUTION)
    leiden_keys[name] = leiden_key

    # Log basic clustering stats
    n_cl = int(mdata["rna"].obs[leiden_key].nunique())
    wandb.log({f"clustering/{name}_leiden_n_clusters": n_cl})

# Pairwise same-cluster consistency:
# For each cell i, let Cx(i) be its cluster in space X and Cy(i) in space Y.
# Define Sx(i) = set of indices in Cx(i) \ {i}, Sy(i) = set in Cy(i) \ {i}.
# Score cell-wise overlap = |Sx(i) ∩ Sy(i)| / max(|Sx(i)|, 1).
# Aggregate by cell type to find percent NOT consistent = 100*(1 - mean_overlap).
pairs = [("expression", "joint"), ("splicing", "joint"), ("expression", "splicing")]

cell_type_col = "broad_cell_type"
if "medium_cell_type" in mdata["rna"].obs:
    cell_type_col = "medium_cell_type"

labels_ct = mdata["rna"].obs[cell_type_col].astype(str).values
n_cells = mdata["rna"].n_obs
idx_all = np.arange(n_cells, dtype=np.int32)

# Precompute cluster membership dicts
cluster_members = {}
for name in ["joint", "expression", "splicing"]:
    labs = mdata["rna"].obs[leiden_keys[name]].values
    # map cluster -> numpy array of member indices
    members = {}
    for cid, grp in pd.Series(idx_all).groupby(labs):
        members[cid] = grp.values.astype(np.int32, copy=False)
    cluster_members[name] = (labs, members)

# Compute per-cell overlap for each pair and summarize per cell type
heat_records = []
for a, b in pairs:
    labs_a, mem_a = cluster_members[a]
    labs_b, mem_b = cluster_members[b]

    overlap = np.empty(n_cells, dtype=np.float32)
    for i in range(n_cells):
        ca = labs_a[i]
        cb = labs_b[i]
        Sa = mem_a[ca]
        Sb = mem_b[cb]
        # exclude self quickly if present at the first occurrence
        # Using set operations on small arrays by converting to Python sets lazily
        # Note: sizes are typically moderate under Leiden, so this is fine
        if Sa.size <= 1:
            overlap[i] = np.nan
            continue
        # Remove i from Sa without copying whole array when i is first element is not guaranteed, so do a mask
        # For small sizes this is acceptable
        Sa_no_i = Sa[Sa != i]
        # Compute intersection size
        inter_sz = len(set(Sa_no_i).intersection(Sb))
        overlap[i] = inter_sz / float(Sa_no_i.size)

    # Attach per-cell result to obs for traceability
    key_cell = f"samecluster_overlap_{a}_vs_{b}"
    mdata["rna"].obs[key_cell] = overlap

    # Summary stats overall
    mean_ov = float(np.nanmean(overlap))
    median_ov = float(np.nanmedian(overlap))
    wandb.log({
        f"clustering/{a}_vs_{b}_samecluster_mean": mean_ov,
        f"clustering/{a}_vs_{b}_samecluster_median": median_ov,
    })

    # Per cell type: percent NOT consistent = 100*(1 - mean overlap)
    df_tmp = pd.DataFrame({
        "cell_type": labels_ct,
        "overlap": overlap
    }).groupby("cell_type", as_index=False)["overlap"].mean()
    df_tmp["pct_not_consistent"] = (1.0 - df_tmp["overlap"].fillna(0.0)) * 100.0
    df_tmp["pair"] = f"{a}_vs_{b}"
    heat_records.append(df_tmp[["cell_type", "pair", "pct_not_consistent"]])

# Build heatmap table
heat_df = pd.concat(heat_records, ignore_index=True)
heat_pivot = heat_df.pivot(index="cell_type", columns="pair", values="pct_not_consistent").fillna(0.0)

plt.figure(figsize=(max(6, 0.25*heat_pivot.shape[1] + 4), max(6, 0.3*heat_pivot.shape[0] + 3)))
sns.heatmap(heat_pivot, annot=True, fmt=".1f", cbar=True)
plt.title(f"Percent not consistent by cell type (Leiden, res={LEIDEN_RESOLUTION})")
plt.tight_layout()
out_path = f"{args.fig_dir}/heatmap_pct_not_consistent_leiden_res_{LEIDEN_RESOLUTION}.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
wandb.log({"clustering/heatmap_pct_not_consistent": wandb.Image(out_path)})
plt.close()

# Optional: global alignment sanity via AMI between clusterings
for a, b in pairs:
    ami = adjusted_mutual_info_score(
        mdata["rna"].obs[leiden_keys[a]].values,
        mdata["rna"].obs[leiden_keys[b]].values
    )
    wandb.log({f"clustering/{a}_vs_{b}_AMI": float(ami)})

del heat_records, heat_pivot, heat_df
gc.collect()


import gc
import numpy as np
from sklearn.neighbors import NearestNeighbors

print("Computing expression↔︎splicing k-NN overlap on training data…")
# ------------------------------
# 9c. Extended kNN overlap for multiple k and space pairs
# ------------------------------
print("Computing extended kNN overlap for multiple k and space pairs…")

# Use the latent spaces you already computed
Z_joint = latent_spaces["joint"]
Z_expr  = latent_spaces["expression"]
Z_sp    = latent_spaces["splicing"]

pairs_knn = [
    ("expression", Z_expr, "joint", Z_joint),
    ("splicing",   Z_sp,   "joint", Z_joint),
]

from sklearn.neighbors import NearestNeighbors
import numpy as np
import gc
import matplotlib.pyplot as plt
import scanpy as sc

# Normalize/validate ks
ks = NN_OVERLAPS_KS
kmax = max(ks)
n = Z_joint.shape[0]

def knn_indices(Z, k):
    nn = NearestNeighbors(n_neighbors=min(k+1, n), metric="euclidean", n_jobs=-1)
    nn.fit(Z)
    return nn.kneighbors(Z, return_distance=False)

def strip_self(idx_full, k):
    # drop potential self neighbor (first column) or trim to k
    if idx_full.shape[1] > 0:
        out = idx_full[:, 1:] if idx_full.shape[1] > k else idx_full[:, :k]
    else:
        out = idx_full
    if out.dtype != np.int32:
        out = out.astype(np.int32, copy=False)
    return out

# Precompute neighbor indices once at kmax+1 for each space
idx_cache = {}
for name, Z in [("expression", Z_expr), ("splicing", Z_sp), ("joint", Z_joint)]:
    idx = knn_indices(Z, kmax)  # shape n x (kmax+1)
    idx_cache[name] = idx.astype(np.int32, copy=False)

# For each pair and each k, compute overlap counts, attach to obs, log stats, and plot UMAPs
for a_name, Za, b_name, Zb in pairs_knn:
    idx_a_full = idx_cache[a_name]
    idx_b_full = idx_cache[b_name]

    for k in ks:
        # slice to current k and drop self
        idx_a = strip_self(idx_a_full[:, :min(k+1, idx_a_full.shape[1])], k)
        idx_b = strip_self(idx_b_full[:, :min(k+1, idx_b_full.shape[1])], k)

        # Per-cell overlap size divided by k
        overlap_frac = np.empty(n, dtype=np.float32)
        for i in range(n):
            a = idx_a[i]
            b = idx_b[i]
            if a.size == 0:
                overlap_frac[i] = np.nan
                continue
            inter_sz = np.intersect1d(a, b, assume_unique=False).size
            overlap_frac[i] = inter_sz / float(min(k, a.size))

        # Attach and summarize
        key_obs = f"nn_overlap_{a_name}_vs_{b_name}_k{k}_pct"
        mdata["rna"].obs[key_obs] = (100.0 * overlap_frac).astype(np.float32, copy=False)

        vals = mdata["rna"].obs[key_obs].values
        mean_pct   = float(np.nanmean(vals))
        median_pct = float(np.nanmedian(vals))
        p10        = float(np.nanquantile(vals, 0.10))
        p90        = float(np.nanquantile(vals, 0.90))

        wandb.log({
            f"knn-extended/{a_name}_vs_{b_name}_k{k}_mean_pct": mean_pct,
            f"knn-extended/{a_name}_vs_{b_name}_k{k}_median_pct": median_pct,
            f"knn-extended/{a_name}_vs_{b_name}_k{k}_p10_pct": p10,
            f"knn-extended/{a_name}_vs_{b_name}_k{k}_p90_pct": p90,
        })

        # Joint-UMAP overlays (wide + square), colored by current pairwise overlap metric
        if "X_umap_joint" in mdata["rna"].obsm_keys():
            # Wide
            plt.figure(figsize=(13, 6))
            sc.pl.embedding(
                mdata["rna"],
                basis="X_umap_joint",
                color=key_obs,
                legend_loc=None,
                frameon=True,
                color_map="viridis",
                show=False,
            )
            plt.title(f"Joint UMAP — {a_name}↔{b_name} k={k} overlap (%)")
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            out_path = f"{args.fig_dir}/umap_joint_{a_name}_vs_{b_name}_k{k}.png"
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            wandb.log({f"knn-extended/umap_joint_{a_name}_vs_{b_name}_k{k}": wandb.Image(out_path)})
            plt.close()

            # Square
            plt.figure(figsize=(6, 6))
            sc.pl.embedding(
                mdata["rna"],
                basis="X_umap_joint",
                color=key_obs,
                legend_loc=None,
                frameon=True,
                color_map="viridis",
                show=False,
            )
            plt.title(f"Joint UMAP — {a_name}↔{b_name} k={k} overlap (%)")
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            out_path = f"{args.fig_dir}/umap_sqr_joint_{a_name}_vs_{b_name}_k{k}.png"
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            wandb.log({f"knn-extended/umap_sqr_joint_{a_name}_vs_{b_name}_k{k}": wandb.Image(out_path)})
            plt.close()

        # Clean up per-k temporaries
        del overlap_frac
        gc.collect()

# Free large neighbor caches
del idx_cache
gc.collect()


# ------------------------------
# 10. Unified Evaluation on TRAIN only
# ------------------------------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, silhouette_score
)
from scipy.stats import spearmanr

def evaluate_split(name: str, mdata, mask_coords=None, Z_type = "joint"):
    print(f"\n=== Evaluating {name.upper()}-{Z_type} split ===")
    # latent representation
    Z = model.get_latent_representation(adata=mdata, modality = Z_type)

    # Fit PCA on the latent embedding and count PCs to reach 90% variance
    n_comp_max = min(Z.shape[0], Z.shape[1])
    pca = PCA(n_components=n_comp_max, svd_solver="full")
    pca.fit(Z)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    pcs_90 = int(np.searchsorted(cum_var, 0.90) + 1)

    print(f"[{name}-{Z_type}] PCs for 90% variance: {pcs_90}/{Z.shape[1]}")
    wandb.log({
        f"real-{name}-{Z_type}/pca_n_components_90var": pcs_90,
        f"real-{name}-{Z_type}/pca_total_dim": Z.shape[1],
        f"real-{name}-{Z_type}/pca_var90_ratio": pcs_90 / Z.shape[1]
    })


    labels = mdata.obs[umap_color_key].astype(str).values

    # 1) silhouette score based on broad cell type
    sil = silhouette_score(Z, labels)
    print(f"[{name}-{Z_type}] silhouette score: {sil:.4f}")
    wandb.log({f"real-{name}-{Z_type}/{umap_color_key}-silhouette_score": sil})

    labels = mdata.obs[cell_type_classification_key].astype(str).values

    # 1) silhouette score based on medium cell type
    sil = silhouette_score(Z, labels)
    print(f"[{name}-{Z_type}] silhouette score: {sil:.4f}")
    wandb.log({f"real-{name}-{Z_type}/{cell_type_classification_key}-silhouette_score": sil})

    # 2) train/test logistic regression
    Z_tr, Z_ev, y_tr, y_ev = train_test_split(Z, labels, test_size=0.2, random_state=0)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Z_tr, y_tr)
    y_pred = clf.predict(Z_ev)
    acc = accuracy_score(y_ev, y_pred)
    prec = precision_score(y_ev, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_ev, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_ev, y_pred, average="weighted", zero_division=0)
    print(f"[{name}-{Z_type}] LR — acc: {acc:.4f}, prec: {prec:.4f}, rec: {rec:.4f}, f1: {f1:.4f}")
    wandb.log({
        f"real-{name}-{Z_type}/accuracy": acc,
        f"real-{name}-{Z_type}/precision": prec,
        f"real-{name}-{Z_type}/recall": rec,
        f"real-{name}-{Z_type}/f1_score": f1,
    })

    # 4) age regression (splicing latent)
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    ages = mdata.obs['age_numeric'].astype(float).values  

    # standardize latent factors
    X_latent = StandardScaler().fit_transform(Z)

    # train/test split
    X_tr, X_ev, y_tr, y_ev = train_test_split(
        X_latent, ages, test_size=0.2, random_state=0
    )
    # fit ridge on splice‐only latent space
    ridge_sp = RidgeCV(alphas=np.logspace(-2, 3, 20), cv=5).fit(X_tr, y_tr)
    r2_age = ridge_sp.score(X_ev, y_ev)

    print(f"[{name}-{Z_type}] age regression R²: {r2_age:.4f}")
    wandb.log({f"real-{name}-{Z_type}/age_r2": r2_age})

# run evaluation on training data
evaluate_split("train", mdata, Z_type="joint")
evaluate_split("train", mdata, Z_type="expression")
evaluate_split("train", mdata, Z_type="splicing")

# free training data from memory
print("Cleaning up training data from memory…")
del mdata
torch.cuda.empty_cache()

# ------------------------------
# 11. TEST split + masked‐junction imputation
# ------------------------------
print("\nLoading TEST MuData for evaluation and imputation…")
mdata = mu.read_h5mu(args.test_mdata_path, backed = "r")

print("Setting up SpliceVI PartialVAE…")
scvi.model.MULTIVISPLICE.setup_mudata(
    mdata,
    batch_key = None,
    size_factor_key="X_library_size",
    rna_layer="length_norm",
    junc_ratio_layer=x_layer,
    atse_counts_layer=cluster_counts_layer,
    junc_counts_layer=junction_counts_layer,
    psi_mask_layer=mask_layer,
    modalities={"rna_layer": "rna", "junc_ratio_layer": "splicing"},
)

# real-test evaluation
evaluate_split("test", mdata, Z_type="joint")
evaluate_split("test", mdata, Z_type="expression")
evaluate_split("test", mdata, Z_type="splicing")

# ------------------------------
# 11. TEST split + masked‐ATSE imputation
# ------------------------------
del mdata
torch.cuda.empty_cache()

print(f"\n=== Masked-ATSE imputation on TEST using {args.masked_test_mdata_path} ===")
MASK_FRACTION = 0.2  # fraction of ATSEs to mask
mdata = mu.read_h5mu(args.masked_test_mdata_path, backed = "r")
ad_masked = mdata["splicing"]

print(f"Setting up SpliceVI PartialVAE… ")
scvi.model.MULTIVISPLICE.setup_mudata(
    mdata,
    batch_key = None,
    size_factor_key="X_library_size",
    rna_layer="length_norm",
    junc_ratio_layer=x_layer,
    atse_counts_layer=cluster_counts_layer,
    junc_counts_layer=junction_counts_layer,
    psi_mask_layer=mask_layer,
    modalities={"rna_layer": "rna", "junc_ratio_layer": "splicing"},
)

from scipy import sparse

# 6) run imputation in cell-index batches using args.batch_size
print("Step 6: running batched imputation and computing correlations")

model.module.eval()

# Ensure CSR for fast row slicing
masked_orig = ad_masked.layers["junc_ratio_masked_original"]
if not sparse.isspmatrix_csr(masked_orig):
    masked_orig = sparse.csr_matrix(masked_orig)

bin_mask = ad_masked.layers["junc_ratio_masked_bin_mask"]
if not sparse.isspmatrix_csr(bin_mask):
    bin_mask = sparse.csr_matrix(bin_mask)

n_cells = bin_mask.shape[0]
bs = int(args.batch_size) if getattr(args, "batch_size", None) else 512
assert bs > 0, "batch_size must be positive"

orig_all = []
pred_all = []
pairs_total = 0

for start in range(0, n_cells, bs):
    stop = min(start + bs, n_cells)
    # find masked positions for these rows only
    submask = bin_mask[start:stop]                # CSR
    sub_r, sub_c = submask.nonzero()             # local rows, global cols
    if sub_r.size == 0:
        continue

    # decode only these cells, using your existing batch_size for internal minibatching too
    idx = np.arange(start, stop, dtype=np.int64)
    with torch.inference_mode():
        decoded_batch = model.get_normalized_splicing(
            adata=mdata,
            indices=idx,
            return_numpy=True,
            batch_size=bs,          # use args.batch_size here
        )                           # shape: (stop-start, n_junc)

    # ground truth at masked positions
    orig_vals_b = masked_orig[start:stop][:, sub_c][sub_r, np.arange(sub_r.size)].A1
    # predictions at same positions
    pred_vals_b = decoded_batch[sub_r, sub_c]

    orig_all.append(orig_vals_b.astype(np.float32, copy=False))
    pred_all.append(pred_vals_b.astype(np.float32, copy=False))
    pairs_total += orig_vals_b.size

    # free batch ASAP
    del decoded_batch
    torch.cuda.empty_cache()

if pairs_total == 0:
    print("[impute-test] No masked entries found; skipping correlation.")
else:
    orig_all = np.concatenate(orig_all, dtype=np.float32)
    pred_all = np.concatenate(pred_all, dtype=np.float32)

    pearson_m  = float(np.corrcoef(orig_all, pred_all)[0, 1])
    from scipy.stats import spearmanr
    spearman_m = float(spearmanr(orig_all, pred_all, nan_policy="omit")[0])

    print(f"[impute-test] masked-ATSE PSI corr — Pearson: {pearson_m:.4f}, Spearman: {spearman_m:.4f}  (n={pairs_total})")
    wandb.log({
        "impute-test/psi_pearson_corr_masked_atse": pearson_m,
        "impute-test/psi_spearman_corr_masked_atse": spearman_m,
        "impute-test/n_masked_entries": int(pairs_total),
        "impute-test/batch_size_imputation": bs,
    })


# ------------------------------
# 12. Finish
# ------------------------------
print("Pipeline complete. Finishing W&B run.")
wandb.finish()
