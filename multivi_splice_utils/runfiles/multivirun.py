#!/usr/bin/env python
import os
import inspect
import argparse

import scvi
import mudata as mu
import wandb
from pytorch_lightning.loggers import WandbLogger
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# 0. Default Paths (can be overridden via CLI)
# ------------------------------
DEFAULT_MUDATA_PATH       = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/052025/mouse_foundation_data_20250502_155802_ge_splice_combined.h5mu"
DEFAULT_MODEL_SAVE_DIR    = "/gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/models"
DEFAULT_FIGURE_OUTPUT_DIR = "/gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/figures"

# ------------------------------
# 1. Grab train() defaults
# ------------------------------
train_sig = inspect.signature(scvi.model.MULTIVISPLICE.train)
train_defaults = {
    name: param.default
    for name, param in train_sig.parameters.items()
    if name not in ("self", "logger") and param.default is not inspect._empty
}

# ------------------------------
# 2. Grab __init__ default for n_latent
# ------------------------------
init_sig         = inspect.signature(scvi.model.MULTIVISPLICE.__init__)
default_n_latent = init_sig.parameters["n_latent"].default  # None

# ------------------------------
# 3. Build argparse
# ------------------------------
parser = argparse.ArgumentParser("MultiVI-Splice")
parser.add_argument(
    "--mudata_path",
    type=str,
    default=DEFAULT_MUDATA_PATH,
    help=f"MuData input (.h5mu) (default: {DEFAULT_MUDATA_PATH})"
)
parser.add_argument(
    "--model_save_dir",
    type=str,
    default=DEFAULT_MODEL_SAVE_DIR,
    help=f"Directory to save the trained model (default: {DEFAULT_MODEL_SAVE_DIR})"
)
parser.add_argument(
    "--figure_output_dir",
    type=str,
    default=DEFAULT_FIGURE_OUTPUT_DIR,
    help=f"Directory to save UMAP figures (default: {DEFAULT_FIGURE_OUTPUT_DIR})"
)
for name, default in train_defaults.items():
    arg_type = type(default) if default is not None else str
    parser.add_argument(
        f"--{name}",
        type=arg_type,
        default=None,
        help=f"{name} (default = {default!r})"
    )
parser.add_argument(
    "--latent_dim",
    type=int,
    default=None,
    help=f"n_latent override (default = {default_n_latent!r})"
)
parser.add_argument(
    "--umap_cell_label",
    nargs="+",
    default=["broad_cell_type"],
    help="List of obs columns to color UMAP by"
)
args = parser.parse_args()

# ------------------------------
# 4. Merge CLI args with defaults + set paths
# ------------------------------
MUDATA_PATH       = args.mudata_path
MODEL_SAVE_DIR    = args.model_save_dir
FIGURE_OUTPUT_DIR = args.figure_output_dir
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)

train_kwargs = {
    name: getattr(args, name) if getattr(args, name) is not None else default
    for name, default in train_defaults.items()
}
latent_dim  = args.latent_dim
umap_labels = args.umap_cell_label

# ------------------------------
# 5. Initialize W&B
# ------------------------------
full_config = {
    "mudata_path":       MUDATA_PATH,
    "model_save_dir":    MODEL_SAVE_DIR,
    "figure_output_dir": FIGURE_OUTPUT_DIR,
    **train_kwargs,
    "n_latent":          latent_dim,
    "umap_cell_label":   umap_labels,
}
wandb.init(project="multivi-splice", config=full_config)
wandb_logger = WandbLogger(project="multivi-splice", config=full_config)

# ------------------------------
# 6. Load data & set up model
# ------------------------------
print(f"Reading MuData from {MUDATA_PATH}…")
mdata = mu.read_h5mu(MUDATA_PATH)
scvi.model.MULTIVISPLICE.setup_mudata(
    mdata,
    batch_key="mouse.id",
    size_factor_key="X_library_size",
    rna_layer="raw_counts",
    junc_ratio_layer="junc_ratio",
    atse_counts_layer="cell_by_cluster_matrix",
    junc_counts_layer="cell_by_junction_matrix",
    modalities={"rna_layer": "rna", "junc_ratio_layer": "splicing"},
)
model = scvi.model.MULTIVISPLICE(
    mdata,
    n_genes=(mdata["rna"].var["modality"] == "Gene_Expression").sum(),
    n_junctions=(mdata["splicing"].var["modality"] == "Splicing").sum(),
    n_latent=latent_dim,
)

# ------------------------------
# 7. Train
# ------------------------------
print("Starting model training…")
model.train(logger=wandb_logger, **train_kwargs)

# ------------------------------
# 8. Compute UMAP
# ------------------------------
print("Computing latent representation and UMAP…")
latent_key = "X_multivi"
mdata["rna"].obsm[latent_key] = model.get_latent_representation()
sc.pp.neighbors(mdata["rna"], use_rep=latent_key)
sc.tl.umap(mdata["rna"], min_dist=0.2)
print("UMAP embedding done.")

# ------------------------------
# 9. Plot & save one UMAP per label
# ------------------------------
for label in umap_labels:
    if label not in mdata["rna"].obs:
        print(f"Warning: '{label}' not in mdata.obs—skipping.")
        continue
    mdata["rna"].obs[label] = mdata["rna"].obs[label].astype("category")
    cats = mdata["rna"].obs[label].cat.categories
    palette = sns.color_palette("hsv", len(cats))

    fig = sc.pl.umap(
        mdata["rna"],
        color=label,
        palette=palette,
        legend_loc="right margin",
        show=False,
        return_fig=True,
    )
    out_path = os.path.join(
        FIGURE_OUTPUT_DIR, f"umap_{label}_latent{latent_dim}.png"
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    wandb.log({f"umap_{label}": wandb.Image(fig)})
    print(f"Saved UMAP for '{label}' → {out_path}")

# ------------------------------
# 10. Finish
# ------------------------------ #
print("Pipeline complete.")
wandb.finish()
