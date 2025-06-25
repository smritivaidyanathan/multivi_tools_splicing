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
ad = sc.read_h5ad(args.adata_path)

# Check if any NaNs in junc_ratio 
X = ad.layers["junc_ratio"] 

# Step 1: Compute mean of non-NaN values per column (axis=0)
col_means = np.nanmean(X, axis=0)

# Step 2: Subtract column means from non-NaN entries
X_centered = X - col_means[np.newaxis, :]  # broadcast subtraction

# Step 3: Replace NaNs (which are now just untouched entries) with 0
X_centered[np.isnan(X_centered)] = 0.0

# X should be of shape (num_samples, input_dim)
CODE_DIM = args.code_dim or init_defaults.get("code_dim", 16)  # fallback if None
print(f"↪ Using CODE_DIM = {CODE_DIM} for PCA")
pca = PCA(n_components=CODE_DIM)
X_pca = pca.fit_transform(X_centered)  # shape: (n_cells, CODE_DIM)
pca_components = pca.components_.T  # shape: (input_dim, code_dim)

# Layer names
x_layer = "junc_ratio"
junction_counts_layer = "cell_by_junction_matrix"
cluster_counts_layer = "cell_by_cluster_matrix"
mask_layer = "mask"

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

# ------------------------------
# 9. Compute UMAP
# ------------------------------
import scanpy as sc
print("Computing latent representation and UMAP…")
ad.obsm['X_splicevi'] = model.get_latent_representation()
sc.pp.neighbors(ad, use_rep='X_splicevi')
sc.tl.umap(ad, min_dist=0.1)
print("UMAP embedding done.")

# instead of just sc.pl.umap(ad, color="cell_type"), do:
fig = sc.pl.umap(
    ad,
    color="cell_type",
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
# 11. Finish
# ------------------------------
print("Pipeline complete.")
wandb.finish()
