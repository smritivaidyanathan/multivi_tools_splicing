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
from scipy import sparse
import scipy.sparse as ss
import numpy as np

import torch
torch.autograd.set_detect_anomaly(True)

# ------------------------------
# 0. Default Paths (can be overridden via CLI)
# ------------------------------
DEFAULT_MUDATA_PATH       = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/052025/SUBSETTOP5CELLSTYPES_aligned__ge_splice_combined_20250513_035938.h5mu"
DEFAULT_MODEL_SAVE_DIR    = "/gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/models/multivisplice"
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
# 2. Grab __init__ defaults for model
# ------------------------------
init_sig = inspect.signature(scvi.model.MULTIVISPLICE.__init__)
init_defaults = {
    name: param.default
    for name, param in init_sig.parameters.items()
    if name not in ("self", "adata") and param.default is not inspect._empty
}

# ------------------------------
# 3. Build argparse
# ------------------------------
parser = argparse.ArgumentParser("MultiVI-Splice")
# paths
parser.add_argument(
    "--mudata_path", type=str, default=DEFAULT_MUDATA_PATH,
    help=f"MuData input (.h5mu) (default: {DEFAULT_MUDATA_PATH})"
)
parser.add_argument(
    "--model_save_dir", type=str, default=DEFAULT_MODEL_SAVE_DIR,
    help=f"Directory to save the trained model (default: {DEFAULT_MODEL_SAVE_DIR})"
)
parser.add_argument(
    "--figure_output_dir", type=str, default=DEFAULT_FIGURE_OUTPUT_DIR,
    help=f"Directory to save UMAP figures (default: {DEFAULT_FIGURE_OUTPUT_DIR})"
)
# model init params
for name, default in init_defaults.items():
    if name == "n_latent":
        # register it directly as an int
        parser.add_argument(
            "--n_latent",
            type=int,
            default=None,
            help=f"Dimensionality of the latent space (default = {default!r})",
        )
    else:
        arg_type = type(default) if not isinstance(default, bool) else lambda x: x.lower() in ("true", "1")
        parser.add_argument(
            f"--{name}",
            type=arg_type,
            default=None,
            help=f"{name} (default = {default!r})",
        )

# training params
for name, default in train_defaults.items():
    arg_type = type(default) if not isinstance(default, bool) else lambda x: x.lower() in ("true", "1")
    parser.add_argument(
        f"--{name}", type=arg_type, default=None,
        help=f"{name} (default = {default!r})"
    )

# umap labels
parser.add_argument(
    "--umap_cell_label", nargs='+', default=["broad_cell_type"],
    help="List of obs columns to color UMAP by"
)
args = parser.parse_args()

# ------------------------------
# 4. Merge CLI args with defaults + set paths
# ------------------------------
MUDATA_PATH       = args.mudata_path
MODEL_SAVE_DIR    = args.model_save_dir
FIGURE_OUTPUT_DIR = args.figure_output_dir

# build train_kwargs
train_kwargs = {
    name: getattr(args, name) if getattr(args, name) is not None else default
    for name, default in train_defaults.items()
}
# build model init kwargs
model_kwargs = {}
# pick up --n_latent directly
if args.n_latent is not None:
    model_kwargs['n_latent'] = args.n_latent

for name, default in init_defaults.items():
    # skip n_latent since we handled it
    if name == 'n_latent':
        continue
    val = getattr(args, name)
    if val is not None:
        model_kwargs[name] = val

umap_labels = args.umap_cell_label

# ------------------------------
# 5. Initialize W&B
# ------------------------------
full_config = {
    "mudata_path": MUDATA_PATH,
    "model_save_dir": MODEL_SAVE_DIR,
    "figure_output_dir": FIGURE_OUTPUT_DIR,
    **train_kwargs,
    **model_kwargs,
    "umap_cell_label": umap_labels,
}
wandb.init(project="multivi-splice", config=full_config)
wandb_logger = WandbLogger(project="multivi-splice", config=full_config)

# ------------------------------
# 6. Load data & set up model
# ------------------------------




from scipy import sparse
import numpy as np

mdata = mu.read_h5mu(MUDATA_PATH)

# # grab the splicing modality
splicing = mdata["splicing"]

# # 1) load cluster‐ and junction‐counts
# cluster = splicing.layers["cell_by_cluster_matrix"]
# junction = splicing.layers["cell_by_junction_matrix"]

# # 2) ensure CSR
# if not sparse.isspmatrix_csr(cluster):
#     cluster = sparse.csr_matrix(cluster)
# if not sparse.isspmatrix_csr(junction):
#     junction = sparse.csr_matrix(junction)

# # 3) build psi_mask (1 wherever cluster>0)
# mask = cluster.copy()
# mask.data = np.ones_like(mask.data, dtype=np.uint8)
# splicing.layers["psi_mask"] = mask


# # 4) compute junc_ratio = junction / cluster, nan→0
# #    convert to dense for element‐wise division
# cluster_arr = cluster.toarray()
# junction_arr = junction.toarray()
# junc_ratio = np.divide(
#     junction_arr,
#     cluster_arr,
#     out=np.zeros_like(junction_arr, dtype=float),
#     where=(cluster_arr != 0)
# )
# # 5) assign back (dense array is fine here)
# splicing.layers["junc_ratio"] = junc_ratio

print(f"Junction Ratio: {splicing.layers["junc_ratio"]}")

print("Now splicing layers:", splicing.layers.keys())
print(f"Mask: {splicing.layers["psi_mask"]}")
print(f"junc_ratio shape: {splicing.layers["junc_ratio"].shape}")
import gc
# del cluster, junction, cluster_arr, junction_arr, mask
# gc.collect()  # give Python a nudge to free the memory


print("MuData modalities loaded:", list(mdata.mod.keys()))
print(mdata)

scvi.model.MULTIVISPLICE.setup_mudata(
    mdata,
    batch_key = "dataset",
    size_factor_key="X_library_size",
    rna_layer="raw_counts",
    junc_ratio_layer="junc_ratio",
    atse_counts_layer="cell_by_cluster_matrix",
    junc_counts_layer="cell_by_junction_matrix",
    psi_mask_layer="psi_mask",            # ← here
    modalities={"rna_layer": "rna", "junc_ratio_layer": "splicing"},
)

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

# 1) Get the embedding module
emb_mod = model.module.z_encoder_splicing.feature_embedding

# 2) If it’s an nn.Embedding layer, grab .weight; otherwise emb_mod is already the weight matrix
weight_tensor = getattr(emb_mod, "weight", emb_mod)

# 3) Detach, move to CPU, convert to NumPy
emb = weight_tensor.detach().cpu().numpy()

# 4) Print shape and contents (or slice if it's huge)
print("Embedding shape:", emb.shape)
print(emb)           # prints the full matrix
# or, for a quick peek:
print(emb[:5, :5])   # first 5 rows × 5 cols

jr_info = model.adata_manager.data_registry["junc_ratio"]
jr_key, mod_key = jr_info.attr_key, jr_info.mod_key

ac_info = model.adata_manager.data_registry["atse_counts_key"]
ac_key = ac_info.attr_key

# 2) Grab as CSR matrices
X = model.adata["splicing"].layers["junc_ratio"]

print(X)
C = model.adata["splicing"].layers[ac_key]


# ------------------------------
# 7. Train
# ------------------------------
print("Starting model training with:", train_kwargs)
model.train(logger=wandb_logger, **train_kwargs)
try:
    model.save(MODEL_SAVE_DIR, overwrite=True)
    print(f"Model saved successfully to: {MODEL_SAVE_DIR}")
except Exception as e:
    print(f"Error saving model: {e}")

print(f"model.save tried to write files to: {MODEL_SAVE_DIR}")

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
    palette = sns.color_palette("hsv", len(mdata["rna"].obs[label].cat.categories))
    fig = sc.pl.umap(
        mdata["rna"],
        color=label,
        palette=palette,
        legend_loc="right margin",
        show=False,
        return_fig=True,
    )
    out_path = os.path.join(
        FIGURE_OUTPUT_DIR, f"umap_{label}_latent{model_kwargs.get('n_latent', init_defaults.get('n_latent'))}.png"
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    wandb.log({f"umap_{label}": wandb.Image(fig)})
    print(f"Saved UMAP for '{label}' → {out_path}")

# ------------------------------
# 8. Compute UMAPs for multiple latent spaces
# ------------------------------
print("Computing latent representations and UMAPs...")
latent_spaces = {
    "joint":      model.get_latent_representation(),
    "expression": model.get_latent_representation(modality="expression"),
    "splicing":   model.get_latent_representation(modality="splicing"),
}

for space_name, Z in latent_spaces.items():
    key = f"X_{space_name}"
    mdata["rna"].obsm[key] = Z
    sc.pp.neighbors(mdata["rna"], use_rep=key)
    sc.tl.umap(mdata["rna"], min_dist=0.2)
    print(f"UMAP computed for: {space_name}")

    for label in umap_labels:
        if label not in mdata["rna"].obs:
            print(f"Warning: '{label}' not in mdata.obs—skipping.")
            continue

        mdata["rna"].obs[label] = mdata["rna"].obs[label].astype("category")
        palette = sns.color_palette("hsv", len(mdata["rna"].obs[label].cat.categories))
        fig = sc.pl.umap(
            mdata["rna"],
            color=label,
            palette=palette,
            legend_loc="right margin",
            show=False,
            return_fig=True,
        )

        filename = f"umap_{label}_{space_name}_latent{model_kwargs.get('n_latent', init_defaults.get('n_latent'))}.png"
        out_path = os.path.join(FIGURE_OUTPUT_DIR, filename)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        wandb.log({f"{space_name}_umap_{label}": wandb.Image(out_path)})
        print(f"Saved UMAP for '{label}' in space '{space_name}' → {out_path}")


# ------------------------------
# 10. Finish
# ------------------------------
print("Pipeline complete.")
wandb.finish()
