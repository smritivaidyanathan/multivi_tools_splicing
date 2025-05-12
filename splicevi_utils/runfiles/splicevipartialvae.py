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
import numpy as np
from scipy import sparse

# ------------------------------
# 0. Default Paths (CLI-overridable)
# ------------------------------
DEFAULT_ANN_DATA = "/gpfs/commons/groups/knowles_lab/Karin/TMS_MODELING/DATA_FILES/SIMULATED/simulated_data_2025-03-27.h5ad"
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
# mask layer
mask = (ad.layers.get('cell_by_cluster_matrix', ad.X).toarray() > 0).astype(np.uint8)
ad.layers['mask'] = sparse.csr_matrix(mask)
# clean junction ratios if present
if 'junc_ratio' in ad.layers:
    jr = ad.layers['junc_ratio']
    jr_arr = jr.toarray() if sparse.issparse(jr) else jr
    ad.layers['junc_ratio'] = sparse.csr_matrix(np.nan_to_num(jr_arr, nan=0.0))

# ------------------------------
# 7. Setup model
# ------------------------------
print("Setting up SpliceVI PartialVAE…")
scvi.model.SPLICEVI.setup_anndata(
    ad,
    junc_ratio_layer="junc_ratio",
    junc_counts_layer="cell_by_junction_matrix",
    cluster_counts_layer="cell_by_cluster_matrix",
    psi_mask_layer="mask",
    batch_key="mouse.id",
)
# model init kwargs
model_kwargs = {name: getattr(args, name) for name in init_defaults if getattr(args, name) is not None}
print("Initializing model with:", model_kwargs)
model = scvi.model.SPLICEVI(ad, **model_kwargs)

# ------------------------------
# 8. Train
# ------------------------------
train_kwargs = {name: getattr(args, name) for name in train_defaults if getattr(args, name) is not None}
print("Training with:", train_kwargs)
model.train(logger=wandb_logger, **train_kwargs)
model.save(args.model_dir, overwrite=True)
wandb.log({"model_saved_to": args.model_dir})

# ------------------------------
# 9. Compute UMAP
# ------------------------------
print("Computing latent representation and UMAP…")
ad.obsm['X_splicevi'] = model.get_latent_representation()
sc.pp.neighbors(ad, use_rep='X_splicevi')
sc.tl.umap(ad, min_dist=0.2)
print("UMAP embedding done.")

# ------------------------------
# 10. Plot & log UMAP for each color
# ------------------------------
for color in args.umap_colors:
    if color not in ad.obs:
        print(f"Warning: '{color}' not in ad.obs — skipping.")
        continue
    ad.obs[color] = ad.obs[color].astype('category')
    fig = sc.pl.umap(ad, color=color, show=False, return_fig=True)
    out_path = os.path.join(args.fig_dir, f"umap_splicevi_{color}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    wandb.log({f"umap_splicevi_{color}": wandb.Image(fig)})
    print(f"Saved & logged UMAP for '{color}' → {out_path}")

# ------------------------------
# 11. Finish
# ------------------------------
print("Pipeline complete.")
wandb.finish()
