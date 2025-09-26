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
mdata = mu.read_h5mu(args.train_mdata_path, backed = True)

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

import gc
import numpy as np
from sklearn.neighbors import NearestNeighbors

print("Computing expression↔︎splicing k-NN overlap on training data…")

KNN_K = 15
k = int(KNN_K)

Z_expr = latent_spaces["expression"]
Z_sp   = latent_spaces["splicing"]

n = Z_expr.shape[0]
assert Z_sp.shape[0] == n, "Expression and splicing latents must have same #rows (cells)."

# Build kNN (exclude self by asking k+1 then dropping)
nnb_expr = NearestNeighbors(n_neighbors=min(k+1, n), metric="euclidean", n_jobs=-1)
nnb_sp   = NearestNeighbors(n_neighbors=min(k+1, n), metric="euclidean", n_jobs=-1)

nnb_expr.fit(Z_expr)
nnb_sp.fit(Z_sp)

# Request indices only to avoid allocating distances
idx_expr = nnb_expr.kneighbors(Z_expr, return_distance=False)
idx_sp   = nnb_sp.kneighbors(Z_sp,   return_distance=False)

# Drop self without copying full arrays when possible
def drop_self_inplace(idx_full: np.ndarray, k: int) -> np.ndarray:
    """
    Returns a view (slice) with self dropped if present in col 0; otherwise trims to k.
    Ensures dtype int32 to cut memory ~in half vs default int64.
    """
    # slice first, then astype with copy=False (will copy only if needed)
    if idx_full.shape[1] > 0:
        idx_k = idx_full[:, 1:] if idx_full.shape[1] > k else idx_full[:, :k]
    else:
        idx_k = idx_full  # degenerate
    if idx_k.dtype != np.int32:
        idx_k = idx_k.astype(np.int32, copy=False)
    return idx_k

idx_expr_k = drop_self_inplace(idx_expr, k)
idx_sp_k   = drop_self_inplace(idx_sp,   k)

# Free the larger (k+1) neighbor arrays if they aren't just views
if idx_expr_k.base is not idx_expr:
    del idx_expr
else:
    # idx_expr_k is a view; still safe to delete name
    del idx_expr
if idx_sp_k.base is not idx_sp:
    del idx_sp
else:
    del idx_sp

gc.collect()

# Per-cell overlap (float32 to save memory)
overlap_frac = np.empty(n, dtype=np.float32)

# Row-wise intersection keeps temporaries tiny (≤k)
for i in range(n):
    a = idx_expr_k[i]
    b = idx_sp_k[i]
    if a.size == 0:
        overlap_frac[i] = np.nan
        continue
    # np.intersect1d allocates a small temporary (≤k)
    inter_sz = np.intersect1d(a, b, assume_unique=False).size
    denom = float(min(k, a.size))
    overlap_frac[i] = inter_sz / denom

# Not needed anymore
del idx_expr_k, idx_sp_k
gc.collect()

overlap_pct = (100.0 * overlap_frac).astype(np.float32, copy=False)

# attach to AnnData for plotting
overlap_key = f"nn_overlap_expr_splice_k{k}_pct"
mdata["rna"].obs[overlap_key] = overlap_pct  # AnnData will own a copy if needed

# Summary stats (nan-aware); quantiles via nanquantile avoid extra temp copies
mean_pct   = float(np.nanmean(overlap_pct))
median_pct = float(np.nanmedian(overlap_pct))
p10        = float(np.nanquantile(overlap_pct, 0.10))
p90        = float(np.nanquantile(overlap_pct, 0.90))

print(f"[train] NN overlap (k={k}) — mean: {mean_pct:.2f}%, median: {median_pct:.2f}%, p10: {p10:.2f}%, p90: {p90:.2f}%")

wandb.log({
    f"real-train/nn_overlap_k{k}_mean_pct": mean_pct,
    f"real-train/nn_overlap_k{k}_median_pct": median_pct,
    f"real-train/nn_overlap_k{k}_p10_pct": p10,
    f"real-train/nn_overlap_k{k}_p90_pct": p90,
})

# Plot (UMAP) — keep memory usage low and clean up figures
key_umap_joint = "X_umap_joint"
if key_umap_joint in mdata["rna"].obsm_keys():
    import matplotlib.pyplot as plt
    import scanpy as sc

    plt.figure(figsize=(8, 6))
    sc.pl.embedding(
        mdata["rna"],
        basis=key_umap_joint,
        color=overlap_key,
        frameon=True,
        legend_loc=None,
        color_map="viridis",
        show=False,
    )
    plt.title(f"Joint UMAP colored by expr↔︎splice NN-overlap (k={k}, %)")
    plt.tight_layout()

    out_path = f"{args.fig_dir}/umap_joint_nn_overlap_k{k}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    wandb.log({f"umap_joint_nn_overlap_k{k}": wandb.Image(out_path)})
    plt.close()
else:
    print(f"[train] WARNING: {key_umap_joint} not found; skipping overlap-colored UMAP.")

# Final explicit cleanup for peace of mind
del overlap_frac  # overlap_pct is retained in AnnData
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
mdata = mu.read_h5mu(args.test_mdata_path, backed = True)

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
mdata = mu.read_h5mu(args.masked_test_mdata_path, backed = True)
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

# 6) run imputation and evaluate on masked-out entries using sparse indexing
print("Step 6: running imputation and computing correlations")
model.module.eval()
with torch.no_grad():
    decoded = model.get_normalized_splicing(adata=mdata, return_numpy=True)

# ensure CSR for fast row/col lookups
masked_orig = ad_masked.layers["junc_ratio_masked_original"]
if not sparse.isspmatrix_csr(masked_orig):
    masked_orig = sparse.csr_matrix(masked_orig)

bin_mask = ad_masked.layers["junc_ratio_masked_bin_mask"]
if not sparse.isspmatrix_csr(bin_mask):
    bin_mask = sparse.csr_matrix(bin_mask)

# get masked locations (row, col indices)
rows, cols = bin_mask.nonzero()

# ground-truth original PSI values (may be zero)
orig_vals = masked_orig[rows, cols].A1  # .A1 = flatten to 1D

# model predictions (dense output)
pred_vals = decoded[rows, cols]

# safety check
if orig_vals.size == 0:
    print("[impute-test] No masked entries found in bin mask; skipping correlation.")
else:
    import numpy as np
    from scipy.stats import spearmanr

    pearson_m  = np.corrcoef(orig_vals, pred_vals)[0, 1]
    spearman_m = spearmanr(orig_vals, pred_vals, nan_policy="omit")[0]

    print(f"[impute-test] masked‐ATSE PSI corr — Pearson: {pearson_m:.4f}, Spearman: {spearman_m:.4f}")
    wandb.log({
        "impute-test/psi_pearson_corr_masked_atse": pearson_m,
        "impute-test/psi_spearman_corr_masked_atse": spearman_m,
        "impute-test/n_masked_entries": int(orig_vals.size),
    })

# ------------------------------
# 12. Finish
# ------------------------------
print("Pipeline complete. Finishing W&B run.")
wandb.finish()
