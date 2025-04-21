# MultiVI-Splice Pipeline for MuData (sbatch-ready version)

print("Starting MultiVI-Splice pipeline...")

# ------------------------------ #
# 0. Configure Paths
# ------------------------------ #
MUDATA_PATH = "/gpfs/commons/groups/knowles_lab/Karin/TMS_MODELING/DATA_FILES/ALL_CELLS/022025/TMS_MUData_GE_ATSE_20250209_165655.h5mu"
MODEL_SAVE_DIR = "/gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/models"
IMPUTED_DFS_PATH = "/gpfs/commons/home/svaidyanathan/dfs/imputed_dfs.h5"
OUTPUT_UPDATED_MUDATA = "/gpfs/commons/groups/knowles_lab/Karin/TMS_MODELING/DATA_FILES/ALL_CELLS/022025/UPDATED_TMS_MUData_GE_ATSE_20250209_165655.h5mu"
FIGURE_OUTPUT_DIR = "/gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/figures"
import os
os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)

print("Configured all paths.")
print("MuData path:", MUDATA_PATH)
print("Model save directory:", MODEL_SAVE_DIR)
print("Imputed DataFrames path:", IMPUTED_DFS_PATH)
print("Updated MuData output path:", OUTPUT_UPDATED_MUDATA)
print("Figure output directory:", FIGURE_OUTPUT_DIR)

# ------------------------------ #
# 1. Imports
# ------------------------------ #
import scvi
import mudata as mu
import wandb
from pytorch_lightning.loggers import WandbLogger

import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd

print(f"scvi-tools version: {getattr(scvi, '__version__', 'unknown')}")

# ------------------------------ #
# 2. Initialize W&B
# ------------------------------ #
wandb.init(project="multivi-splice", config={
    "mudata_path": MUDATA_PATH,
    "model_save_dir": MODEL_SAVE_DIR,
    "imputed_dfs_path": IMPUTED_DFS_PATH
})
wandb_logger = WandbLogger(project="multivi-splice")
print("Initialized Weights & Biases logging.")

# ------------------------------ #
# 3. Load MuData
# ------------------------------ #
print("Loading MuData...")
mdata = mu.read_h5mu(MUDATA_PATH)
print("Loaded MuData with modalities:", list(mdata.mod.keys()))

# ------------------------------ #
# 4. Setup Model
# ------------------------------ #
print("Setting up MultiVI-Splice...")
scvi.model.MULTIVISPLICE.setup_mudata(
    mdata,
    batch_key="mouse.id",
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
)
model.view_anndata_setup()
print("Model initialized.")

# ------------------------------ #
# 5. Train and Save
# ------------------------------ #
print("Beginning model training...")
model.train(logger=wandb_logger)
print("Training complete.")

print(f"Saving model to: {MODEL_SAVE_DIR}")
model.save(MODEL_SAVE_DIR, overwrite=True)
print("Model saved.")

# ------------------------------ #
# 6. Reload Model
# ------------------------------ #
print("Reloading MuData and model...")
mdata = mu.read_h5mu(MUDATA_PATH)
model = scvi.model.MULTIVISPLICE.load(MODEL_SAVE_DIR, adata=mdata)
print("Reload complete.")

# ------------------------------ #
# 7. Impute Expression & Splicing
# ------------------------------ #
# print("Generating imputed estimates...")
# model.get_splicing_estimates().to_hdf(IMPUTED_DFS_PATH, key="imputed_splicing_estimates", mode="w")
# model.get_normalized_expression().to_hdf(IMPUTED_DFS_PATH, key="imputed_expression_estimates", mode="a")
# wandb.save(IMPUTED_DFS_PATH)
# print(f"Imputed estimates saved to {IMPUTED_DFS_PATH}")

# ------------------------------ #
# 8. Latent Representation + UMAP
# ------------------------------ #
print("Computing latent representation and UMAP...")
latent_key = "X_multivi"
mdata["rna"].obsm[latent_key] = model.get_latent_representation()

sc.pp.neighbors(mdata["rna"], use_rep=latent_key)
sc.tl.umap(mdata["rna"], min_dist=0.2)
print("UMAP complete.")

# ------------------------------ #
# 9. Plot and Save UMAP (only cell_type_grouped)
# ------------------------------ #
group = "cell_type_grouped"
if group in mdata["rna"].obs.columns:
    print(f"Creating UMAP plot for: {group}")
    fig = sc.pl.umap(mdata["rna"], color=group, show=False)
    fig_path = os.path.join(FIGURE_OUTPUT_DIR, f"umap_{group}.png")
    fig.figure.savefig(fig_path, dpi=300, bbox_inches="tight")
    wandb.log({f"umap_{group}": wandb.Image(fig_path)})
    print(f"UMAP figure saved and logged: {fig_path}")
    plt.close(fig.figure)
else:
    print(f"Column '{group}' not found in .obs. Skipping plot.")

# ------------------------------ #
# 10. Write Updated MuData
# ------------------------------ #
# print("Writing updated MuData with latent space...")
# mdata.write(OUTPUT_UPDATED_MUDATA)
# print(f"Updated MuData written to: {OUTPUT_UPDATED_MUDATA}")

# ------------------------------ #
# 11. Done
# ------------------------------ #
print("Pipeline complete.")
wandb.finish()
