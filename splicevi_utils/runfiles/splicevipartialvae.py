# splicevi_pipeline.py
#!/usr/bin/env python
import os
import inspect
import argparse

import scanpy as sc
import scvi
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
import wandb

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
    if name != "self" and param.default is not inspect._empty
}

# ------------------------------
# 2. Grab __init__ defaults
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
parser = argparse.ArgumentParser("SpliceVI-Test")
# paths
parser.add_argument("--adata_path", type=str, default=DEFAULT_ANN_DATA,
                    help=f"AnnData (.h5ad) path (default: {DEFAULT_ANN_DATA})")
parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR,
                    help=f"Where to save model (default: {DEFAULT_MODEL_DIR})")
parser.add_argument("--fig_dir", type=str, default=DEFAULT_FIG_DIR,
                    help=f"Where to save figures (default: {DEFAULT_FIG_DIR})")
# model init params
for name, default in init_defaults.items():
    arg_type = type(default) if default is not None else float
    parser.add_argument(f"--{name}", type=arg_type, default=None,
                        help=f"{name} (default: {default!r})")
# training params
for name, default in train_defaults.items():
    arg_type = type(default) if default is not None else float
    parser.add_argument(f"--{name}", type=arg_type, default=None,
                        help=f"{name} (default: {default!r})")
# UMAP color fields
parser.add_argument(
    "--umap_colors",
    nargs='+',
    default=["cell_type_grouped"],
    help="List of obs fields to color UMAP by"
)
# wandb project/run name
parser.add_argument("--wandb_project", type=str, default="multivi-splice", help="W&B project name")
parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name (optional)")
args = parser.parse_args()

# ------------------------------
# 4. Prepare directories
# ------------------------------
os.makedirs(args.model_dir, exist_ok=True)
os.makedirs(args.fig_dir, exist_ok=True)

# ------------------------------
# 5. Initialize W&B
# ------------------------------
# merge config
config = {**init_defaults, **train_defaults}
for key in list(config):
    val = getattr(args, key)
    if val is not None:
        config[key] = val
config.update({
    "adata_path": args.adata_path,
    "model_dir": args.model_dir,
    "fig_dir": args.fig_dir,
    "umap_colors": args.umap_colors
})

wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=config)

# ------------------------------
# 6. Run pipeline
# ------------------------------
print("scvi-tools:", scvi.__version__)
adata = sc.read_h5ad(args.adata_path)
# preprocess mask
mask = (adata.layers['cell_by_cluster_matrix'].toarray() > 0).astype(np.uint8)
adata.layers['mask'] = mask
# clean junc_ratio
jr = adata.layers['junc_ratio']
jr_arr = jr.toarray() if sparse.issparse(jr) else jr
adata.layers['junc_ratio'] = sparse.csr_matrix(np.nan_to_num(jr_arr, nan=0.0))

# setup
scvi.model.SPLICEVI.setup_anndata(
    adata,
    junc_ratio_layer="junc_ratio",
    junc_counts_layer="cell_by_junction_matrix",
    cluster_counts_layer="cell_by_cluster_matrix",
    psi_mask_layer="mask",
    batch_key="mouse.id",
)
# init model
model_kwargs = {k: v for k, v in init_defaults.items() if getattr(args, k) is not None}
model = scvi.model.SPLICEVI(adata, **model_kwargs)
wandb.log({"model_summary": model._model_summary_string})
# train
train_kwargs = {k: v for k, v in train_defaults.items() if getattr(args, k) is not None}
model.train(**train_kwargs)
model.save(args.model_dir, overwrite=True)
wandb.log({"model_saved_to": args.model_dir})

# ------------------------------
# 7. Latent representation
# ------------------------------
adata.obsm['X_splicevi'] = model.get_latent_representation()
sc.pp.neighbors(adata, use_rep='X_splicevi')
sc.tl.umap(adata)

# ------------------------------
# 8. Plot & log UMAP for each color
# ------------------------------
for color in args.umap_colors:
    if color not in adata.obs:
        print(f"Warning: '{color}' not in adata.obs—skipping.")
        continue
    adata.obs[color] = adata.obs[color].astype('category')
    fig = sc.pl.umap(
        adata,
        color=color,
        show=False,
        return_fig=True,
    )
    out_path = os.path.join(args.fig_dir, f"umap_splicevi_{color}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    wandb.log({f"umap_splicevi_{color}": wandb.Image(fig)})
    print(f"Saved & logged UMAP for '{color}' → {out_path}")

wandb.finish()