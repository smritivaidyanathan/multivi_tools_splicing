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
    "--train_adata_path", type=str, default=DEFAULT_ANN_DATA,
    help=f"Train AnnData (.h5ad) input path (default: {DEFAULT_ANN_DATA})"
)
parser.add_argument(
    "--test_adata_path", type=str, default=DEFAULT_ANN_DATA,
    help=f"Test AnnData (.h5ad) input path (default: {DEFAULT_ANN_DATA})"
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
    "train_adata_path": args.train_adata_path,
    "test_adata_path": args.test_adata_path,
    "model_dir": args.model_dir,
    "fig_dir": args.fig_dir,
    "umap_colors": args.umap_colors,
})
wandb.init(project="splicevi-partialvae", config=full_config)
wandb_logger = WandbLogger(project="splicevi-partialvae", config=full_config)

# ------------------------------
# 6. Load AnnData & Preprocess
# ------------------------------
print(f"Loading Training MuData from {args.train_adata_path}…")

import mudata as mu
print("Loading Training AnnData")
mdata = mu.read_h5mu(args.train_adata_path)
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
# model.view_anndata_setup()

# # Initialize model
# model = scvi.model.SPLICEVI(
#     adata=ad,
#     code_dim=CODE_DIM,
#     h_hidden_dim=64,
#     encoder_hidden_dim=128,
#     latent_dim=10,
#     dropout_rate=0.01,
#     learn_concentration=False,
#     splice_likelihood="binomial"
# )

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

# # Add PCA embedding for initialization
# # Get feature embedding before PCA init
# pca_tensor = torch.tensor(pca_components, dtype=model.module.encoder.feature_embedding.dtype)
# embedding_tensor = model.module.encoder.feature_embedding.detach()

# diff = torch.norm(embedding_tensor - pca_tensor).item()
# print(f"L2 norm between PCA loadings and random model embedding: {diff:.6f}")

# model.module.initialize_feature_embedding_from_pca(pca_components)
# pca_tensor = torch.tensor(pca_components, dtype=model.module.encoder.feature_embedding.dtype)
# embedding_tensor = model.module.encoder.feature_embedding.detach()
# diff = torch.norm(embedding_tensor - pca_tensor).item()
# print(f"L2 norm between initialized PCA and model embedding: {diff:.6f}")


# ------------------------------
# 8. Train
# ------------------------------
train_kwargs = {name: getattr(args, name) for name in train_defaults if getattr(args, name) is not None}
print("Training with:", train_kwargs)
# model.train(
#     max_epochs=100,
#     lr=1e-2,
#     batch_size=512,
#     early_stopping=True,
#     n_epochs_kl_warmup=10,
#     weight_decay=0,
#     save_best=False,
#     check_val_every_n_epoch=5,
# )
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
cell_type_classification_key = "medium_cell_type"

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


print("Running logistic regression on latent space…")

print("Loading Testing AnnData...")
mdata = None
ad = None

mdata = mu.read_h5mu(args.test_adata_path)
# grab the splicing modality
ad = mdata["splicing"]

# Get latent embedding and labels
Z = ad.obsm["X_splicevi"]
y = ad.obs[cell_type_classification_key].astype(str).values
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
    "real/accuracy": acc,
    "real/precision": prec,
    "real/recall": rec,
    "real/f1_score": f1,
})
print(f"LogReg — acc: {acc:.4f}, prec: {prec:.4f}, rec: {rec:.4f}, f1: {f1:.4f}")


# ————————————————————————————————————————————————————————
# Silhouette score on the learned latent space
print("Computing silhouette score…")
labels = ad.obs[cell_type_classification_key].astype(str).values
Z = ad.obsm["X_splicevi"]
sil = silhouette_score(Z, labels)
wandb.log({"real/silhouette_score": sil})
print(f"Silhouette score ({cell_type_classification_key}): {sil:.4f}")

# ————————————————————————————————————————————————————————
from scipy.stats import spearmanr
# Correlation between decoder-predicted PSI and observed PSI (excluding zeros)
print("Computing PSI prediction vs observed correlation (excluding zeros)…")
# 1) pull out decoded splicing probs as a numpy array
decoded = model.get_normalized_splicing(adata=ad, return_numpy=True)
# 2) pull out observed PSI from the AnnData layer
jr = ad.layers["junc_ratio"]
obs = jr.toarray() if sparse.issparse(jr) else jr
# 3) flatten both
flat_decoded = decoded.ravel()
flat_obs = obs.ravel()
# 4) filter for non-zero observed PSI
mask = flat_obs != 0
filtered_obs = flat_obs[mask]
filtered_decoded = flat_decoded[mask]
# 5) compute Pearson and Spearman
pearson_corr = np.corrcoef(filtered_obs, filtered_decoded)[0, 1]
spearman_corr, _ = spearmanr(filtered_obs, filtered_decoded)
print(f"Predicted vs observed PSI correlations — Pearson: {pearson_corr:.4f}, Spearman: {spearman_corr:.4f}")
# log to W&B
wandb.log({
    "real/psi_pred_obs_pearson_corr": pearson_corr,
    "real/psi_pred_obs_spearman_corr": spearman_corr,
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
wandb.log({"real/psi_hexbin_excl_zeros": wandb.Image(fig_hex)})
plt.close(fig_hex)



# ------------------------------
# 11. Finish
# ------------------------------
print("Pipeline complete.")
wandb.finish()
