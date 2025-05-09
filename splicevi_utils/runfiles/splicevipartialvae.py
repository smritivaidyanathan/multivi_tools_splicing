# %% [markdown]
# # SpliceVI Test with W&B Logging

# %% 1. Environment & Imports
import os
import scanpy as sc
import scvi
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
import wandb

# Replace with your SpliceVI import if custom
# from my_splicevi_package import SpliceVI

print("scvi-tools:", scvi.__version__)

# Initialize Weights & Biases
wandb.init(project="multivi-splice", name="SpliceVI_Run", config={
    "model": "SpliceVI",
    "adata_path": "/gpfs/commons/groups/knowles_lab/Karin/TMS_MODELING/DATA_FILES/SIMULATED/simulated_data_2025-03-27.h5ad"
})

# %% 2. Paths & I/O
ATSE_ANN_DATA = wandb.config.adata_path
MODEL_DIR     = "/gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/models"
FIG_DIR       = "/gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# %% 3. Load AnnData
wandb.log({"status": "Loading AnnData..."})
print("Loading AnnData of ATSEs…")
adata = sc.read_h5ad(ATSE_ANN_DATA)
print(adata)

# %% 4. Preprocess mask
print("Creating mask from 'cell_by_cluster_matrix'")
atse_counts = adata.layers["cell_by_cluster_matrix"]
atse_arr = atse_counts.toarray() if sparse.issparse(atse_counts) else atse_counts
mask = (atse_arr > 0).astype(np.uint8)
adata.layers["mask"] = mask
print("Added 'mask' layer:", mask.shape)
wandb.log({"mask_unique_values": np.unique(mask).tolist()})

# %% 5. Clean junction ratio
print("Cleaning 'junc_ratio' layer...")
jr = adata.layers["junc_ratio"]
if sparse.issparse(jr):
    jr_arr = jr.toarray()
    jr_clean = np.nan_to_num(jr_arr, nan=0.0)
    adata.layers["junc_ratio"] = sparse.csr_matrix(jr_clean)
else:
    adata.layers["junc_ratio"] = np.nan_to_num(jr, nan=0.0)
print("Cleaned 'junc_ratio' layer.")

# %% 6. Setup scvi-tools AnnData
print("Setting up scvi-tools AnnData...")
scvi.model.SPLICEVI.setup_anndata(
    adata,
    junc_ratio_layer="junc_ratio",
    junc_counts_layer="cell_by_junction_matrix",
    cluster_counts_layer="cell_by_cluster_matrix",
    psi_mask_layer="mask",
    batch_key="mouse.id",
)

# %% 7. Initialize the model
print("Initializing SpliceVI model...")
model = scvi.model.SPLICEVI(adata)
try:
    wandb.log({"model_summary": model._model_summary_string})
except Exception as e:
    print("Could not log model summary:", e)

# %% 8. Train
print("Training model...")
model.train()
model.save(MODEL_DIR, overwrite=True)
wandb.log({"model_saved_to": MODEL_DIR})
print("Model saved to", MODEL_DIR)

# %% 9. Latent representation and UMAP
print("Computing latent representation and UMAP...")
adata.obsm["X_splicevi"] = model.get_latent_representation()
sc.pp.neighbors(adata, use_rep="X_splicevi")
sc.tl.umap(adata)

fig = sc.pl.umap(
    adata,
    color=["cell_type_grouped"],
    show=False,
    return_fig=True,
)
fig_path = os.path.join(FIG_DIR, "umap_splicevi_batch.png")
fig.savefig(fig_path, dpi=150)
wandb.log({"umap_splicevi": wandb.Image(fig)})
print("UMAP saved and logged.")

wandb.finish()
