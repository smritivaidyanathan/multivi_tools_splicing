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


# ------------------------------
# 0. Default Paths (CLI-overridable)
# ------------------------------
DEFAULT_ANN_DATA = "/gpfs/commons/groups/knowles_lab/Karin/TMS_MODELING/DATA_FILES/SIMULATED/simulated_data_2025-03-12.h5ad"
DEFAULT_MODEL_DIR = "/gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/models"
DEFAULT_FIG_DIR = "/gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/figures"

# ------------------------------
# 1. Grab train() defaults
# ------------------------------
train_sig = inspect.signature(scvi.model.SPLICEVI.train)
train_defaults = {
    name: param.default
    for name, param in train_sig.parameters.items()
    if name != 'self' and param.default is not inspect._empty
}

# ------------------------------
# 2. Grab __init__ defaults for model
# ------------------------------
init_sig = inspect.signature(scvi.model.SPLICEVI.__init__)
init_defaults = {
    name: param.default
    for name, param in init_sig.parameters.items()
    if name not in ('self', 'adata') and param.default is not inspect._empty
}

# ------------------------------
# 3. Build argparse
# ------------------------------
parser = argparse.ArgumentParser("SpliceVI-PartialVAE")
# paths
parser.add_argument(
    "--adata_path", type=str, default=DEFAULT_ANN_DATA,
    help=f"AnnData (.h5ad) input path (default: {DEFAULT_ANN_DATA})"
)
parser.add_argument(
    "--model_dir", type=str, default=DEFAULT_MODEL_DIR,
    help=f"Directory to save trained model (default: {DEFAULT_MODEL_DIR})"
)
parser.add_argument(
    "--fig_dir", type=str, default=DEFAULT_FIG_DIR,
    help=f"Directory to save UMAP figures (default: {DEFAULT_FIG_DIR})"
)

# ------------------------------
# 3b. Add flags for post‐hoc tests
# ------------------------------
parser.add_argument(
    "--simulated",
    action="store_true",
    help="If set, train a logistic regression on the learned latent space and log metrics to W&B."
)
parser.add_argument(
    "--imputedencoder",
    action="store_true",
    help="If set, compute correlation between model.module.impute_net output and original junc_ratio."
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
    "adata_path": args.adata_path,
    "model_dir": args.model_dir,
    "fig_dir": args.fig_dir,
    "umap_colors": args.umap_colors,
})
wandb.init(project="splicevi-partialvae", config=full_config)
wandb_logger = WandbLogger(project="splicevi-partialvae", config=full_config)

# ------------------------------
# 6. Load AnnData & Preprocess
# ------------------------------
print(f"Loading AnnData from {args.adata_path}…")
if args.simulated:
    print("Is Simulated Data!")
    ad = sc.read_h5ad(args.adata_path)
else:
    import mudata as mu
    print("Is Not Simulated Data!")
    mdata = mu.read_h5mu(args.adata_path)
    # grab the splicing modality
    ad = mdata["splicing"]

if args.imputedencoder:
    print("Is Using Impected Decoder!")
else:
    print("Is Not Using Imputed Decoder!")

# # Check if any NaNs in junc_ratio 
# X = ad.layers["junc_ratio"] 

# # Step 1: Compute mean of non-NaN values per column (axis=0)
# col_means = np.nanmean(X, axis=0)

# # Step 2: Subtract column means from non-NaN entries
# X_centered = X - col_means[np.newaxis, :]  # broadcast subtraction

# # Step 3: Replace NaNs (which are now just untouched entries) with 0
# X_centered[np.isnan(X_centered)] = 0.0

# # X should be of shape (num_samples, input_dim)
# CODE_DIM = args.code_dim or init_defaults.get("code_dim", 16)  # fallback if None
# print(f"↪ Using CODE_DIM = {CODE_DIM} for PCA")
# pca = PCA(n_components=CODE_DIM)
# X_pca = pca.fit_transform(X_centered)  # shape: (n_cells, CODE_DIM)
# pca_components = pca.components_.T  # shape: (input_dim, code_dim)

# Layer names
x_layer = "junc_ratio"
junction_counts_layer = "cell_by_junction_matrix"
cluster_counts_layer = "cell_by_cluster_matrix"
mask_layer = "psi_mask"

if args.simulated:
# --- Construct mask from cluster counts ---
    if cluster_counts_layer in ad.layers:
        cc = ad.layers[cluster_counts_layer]
        cc_array = cc.toarray() if sparse.issparse(cc) else np.asarray(cc)
        mask = (cc_array > 0).astype(np.uint8)
        ad.layers[mask_layer] = sparse.csr_matrix(mask)
        print(f"Mask layer `{mask_layer}` created from `{cluster_counts_layer}`.")

    # --- Preprocess junction ratio layer ---
    if x_layer in ad.layers:
        jr = ad.layers[x_layer]
        jr_array = jr.toarray() if sparse.issparse(jr) else np.asarray(jr)
        ad.layers[x_layer] = sparse.csr_matrix(np.nan_to_num(jr_array, nan=0.0))
        print(f"Cleaned NaNs in `{x_layer}` and converted to CSR.")

print(f"Found Layers: {ad.layers}")


print("Setting up SpliceVI PartialVAE…")
scvi.model.SPLICEVI.setup_anndata(
    ad,
    junc_ratio_layer=x_layer,
    junc_counts_layer=junction_counts_layer,
    cluster_counts_layer=cluster_counts_layer,
    psi_mask_layer=mask_layer,
    batch_key=None  
)
# ------------------------------
# 7. Setup model
# ------------------------------

# model init kwargs
model_kwargs = {name: getattr(args, name) for name in init_defaults if getattr(args, name) is not None}
print("Initializing model with:", model_kwargs)
model = scvi.model.SPLICEVI(ad, **model_kwargs)
print(model._model_summary_string)
model.view_anndata_setup()

# ── count & log total parameters ─────────────────────────────────────────────
total_params = sum(p.numel() for p in model.module.parameters())
print(f"Total model parameters: {total_params:,}")
wandb.log({"total_parameters": total_params})

# ── watch parameters & gradients in WandB ────────────────────────────────────
wandb.watch(
    model.module,
    log="all",
    log_freq=1000,
    log_graph=False
)

model.view_anndata_setup()

# ------------------------------
# 8. Train
# ------------------------------
train_kwargs = {name: getattr(args, name) for name in train_defaults if getattr(args, name) is not None}
print("Training with:", train_kwargs)
model.train(logger=wandb_logger, check_val_every_n_epoch=5, **train_kwargs)
model.save(args.model_dir, overwrite=True)
wandb.log({"model_saved_to": args.model_dir})

if args.imputedencoder:
    model.module.encoder.finished_training = True

# ------------------------------
# 9. Compute UMAP
# ------------------------------
import scanpy as sc
print("Computing latent representation and UMAP…")
ad.obsm['X_splicevi'] = model.get_latent_representation()
sc.pp.neighbors(ad, use_rep='X_splicevi')
sc.tl.umap(ad, min_dist=0.1)
print("UMAP embedding done.")

umap_color_key = "broad_cell_type"
if args.simulated:
    umap_color_key = "cell_type"

# instead of just sc.pl.umap(ad, color="cell_type"), do:
fig = sc.pl.umap(
    ad,
    color=umap_color_key,
    show=False,          # don’t pop up the figure
    return_fig=True      # return the matplotlib Figure
)

# log it to wandb
import wandb
wandb.log({"umap_cell_type": wandb.Image(fig)})

# clean up
import matplotlib.pyplot as plt
plt.close(fig)


# ------------------------------
# 12. Additional Tests
# ------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, silhouette_score
from scvi import REGISTRY_KEYS
import torch
import numpy as np

# after wandb.finish() or just before exiting:
if args.simulated:
    print("Running logistic regression on latent space…")
    # Get latent embedding and labels
    Z = ad.obsm["X_splicevi"]
    y = ad.obs[umap_color_key].astype(str).values
    # simple train/test split
    from sklearn.model_selection import train_test_split
    Z_train, Z_test, y_train, y_test = train_test_split(Z, y, test_size=0.2, random_state=0)
    # fit classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Z_train, y_train)
    y_pred = clf.predict(Z_test)
    # compute metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    # log to W&B
    wandb.log({
        "simulated/accuracy": acc,
        "simulated/precision": prec,
        "simulated/recall": rec,
        "simulated/f1_score": f1,
    })
    print(f"LogReg — acc: {acc:.4f}, prec: {prec:.4f}, rec: {rec:.4f}, f1: {f1:.4f}")


# ————————————————————————————————————————————————————————
# Silhouette score on the learned latent space
print("Computing silhouette score…")
labels = ad.obs[umap_color_key].astype(str).values
Z = ad.obsm["X_splicevi"]
sil = silhouette_score(Z, labels)
wandb.log({"silhouette_score": sil})
print(f"Silhouette score ({umap_color_key}): {sil:.4f}")

# ————————————————————————————————————————————————————————
from scipy.stats import spearmanr
# Correlation between decoder-predicted PSI and observed PSI (from psi_mask)
print("Computing PSI prediction vs observed correlation (psi_mask)…")

# 1) pull out decoded splicing probs as a numpy array
decoded = model.get_normalized_splicing(adata=ad, return_numpy=True)

# 2) pull out observed PSI from the AnnData layer
jr = ad.layers["junc_ratio"]
obs = jr.toarray() if sparse.issparse(jr) else jr

# 3) flatten both
flat_decoded = decoded.ravel()
flat_obs = obs.ravel()

# 4) get mask from psi_mask layer
mask_mat = ad.layers["psi_mask"]
mask = mask_mat.toarray().ravel().astype(bool) if sparse.issparse(mask_mat) else mask_mat.ravel().astype(bool)

# 5) filter using psi_mask
filtered_obs = flat_obs[mask]
filtered_decoded = flat_decoded[mask]

# 6) compute Pearson and Spearman
pearson_corr = np.corrcoef(filtered_obs, filtered_decoded)[0, 1]
spearman_corr, _ = spearmanr(filtered_obs, filtered_decoded)

print(f"Predicted vs observed PSI correlations — Pearson: {pearson_corr:.4f}, Spearman: {spearman_corr:.4f}")

# 7) log to W&B
wandb.log({
    "psi_pred_obs_pearson_corr": pearson_corr,
    "psi_pred_obs_spearman_corr": spearman_corr,
})


# 6) scatterplot
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(filtered_obs, filtered_decoded, alpha=0.05, edgecolors="none")
ax.set_xlabel("Observed PSI")
ax.set_ylabel("Predicted PSI")
ax.set_title("Predicted vs Observed PSI (Excluding Zeros)")
# line of best fit
m, b = np.polyfit(filtered_obs, filtered_decoded, 1)
x_line = np.array([filtered_obs.min(), filtered_obs.max()])
ax.plot(x_line, m*x_line + b, linewidth=2)
wandb.log({"psi_scatter_excl_zeros": wandb.Image(fig)})
plt.close(fig)

# 7) hexbin plot
fig_hex, ax_hex = plt.subplots(figsize=(6,6))
hb = ax_hex.hexbin(filtered_obs, filtered_decoded, gridsize=100, mincnt=1)
ax_hex.set_xlabel("Observed PSI")
ax_hex.set_ylabel("Predicted PSI")
ax_hex.set_title("Hexbin: Predicted vs Observed PSI (Excluding Zeros)")
cb = fig_hex.colorbar(hb, ax=ax_hex)
cb.set_label("Count")
wandb.log({"psi_hexbin_excl_zeros": wandb.Image(fig_hex)})
plt.close(fig_hex)



# ------------------------------
# 11. Finish
# ------------------------------
print("Pipeline complete.")
wandb.finish()
