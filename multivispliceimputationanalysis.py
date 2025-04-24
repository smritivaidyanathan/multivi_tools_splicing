# %% [markdown]
# # Notebook for Testing Imputation of Missing Modalities (MuData Version)
# 
# This notebook assumes that a combined MuData object has already been created (using the other notebook).
# 
# This notebook will:
# 1. Load combined MuData.
# 2. Corrupt the MuData by masking out a fraction of gene expression and/or splicing data.
# 3. Record original values for masked entries.
# 4. Set up, train, and save the MULTIVISPLICE model.
# 5. Impute missing modalities with `get_normalized_expression` and `get_normalized_splicing`.
# 6. Evaluate imputation accuracy (MSE, median L1, Spearman) on masked entries.
# 7. Log metrics and parameters to W&B.
# 8. Visualize latent space via UMAP

# %% [markdown]
# ## 0. Configure Paths and Parameters

# %%
#MUDATA_PATH = "/gpfs/commons/groups/knowles_lab/Karin/TMS_MODELING/DATA_FILES/BRAIN_ONLY/02112025/TMS_BRAINONLY_MUDATA_GE_ATSE.h5mu"
MUDATA_PATH = "/gpfs/commons/groups/knowles_lab/Karin/TMS_MODELING/DATA_FILES/ALL_CELLS/022025/TMS_MUData_GE_ATSE_20250209_165655.h5mu"
MODEL_SAVE_DIR = "/gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/models"
IMPUTED_DFS_PATH = "/gpfs/commons/home/svaidyanathan/dfs/imputed_dfs.h5"
#OUTPUT_UPDATED_MUDATA = "/gpfs/commons/groups/knowles_lab/Karin/TMS_MODELING/DATA_FILES/BRAIN_ONLY/02112025/MULTVI_TMS_BRAINONLY_MUDATA_GE_ATSE.h5mu"
FIGURE_OUTPUT_DIR = "/gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/figures"
import os
os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)
print("Figure output directory:", FIGURE_OUTPUT_DIR)

# Percent missing parameters
PCT_MISSING_RNA = 0.2       # fraction of cells×genes to mask in gene expression
PCT_MISSING_SPLICE = 0.2    # fraction of cells×junctions to mask in splicing
SEED = 42

# %% [markdown]
# ## 1. Imports

# %%
# Core packages
import numpy as np
import pandas as pd
import anndata as ad
import mudata as mu
import scvi
import scanpy as sc
import scipy.sparse as sp
import torch

# Logging and metrics
import wandb
from scipy.stats import spearmanr

# Visualization
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import WandbLogger

# %% [markdown]
# ## 2. Initialize W&B Logger (Optional)Loading in our stacked annData (created using the ann_data_maker notebook).

# %%
wandb_logger = WandbLogger(project="multivi-splice")  

print("Beginning Script")

print("MuData path:", MUDATA_PATH)
print("Model save directory:", MODEL_SAVE_DIR)
print("Percent missing RNA:", PCT_MISSING_RNA)
print("Percent missing splicing:", PCT_MISSING_SPLICE)

# %% [markdown]
# ## 2. Utility Functions: Corrupt and Evaluate

# %%
def corrupt_mudata(
    mdata: mu.MuData,
    pct_rna: float = 0.0,
    pct_splice: float = 0.0,
    seed: int | None = None,
) -> tuple[mu.MuData, dict]:
    """
    Return a corrupted copy of `mdata` with a fraction of RNA and/or splicing data masked.

    For RNA (gene expression): zero out entries in layer 'raw_counts'.
    For splicing: zero ATSE counts, zero junction counts, set junc_ratio to NaN in layers:
      - 'cell_by_cluster_matrix'
      - 'cell_by_junction_matrix'
      - ratio layer 'junc_ratio'

    Returns:
      - corrupted_mdata: MuData copy with masked entries
      - orig_values: dict of original values for masked entries { 'rna': (indices, values), 'splice': (indices, values) }
      - masks: dict of boolean masks for where data was removed
    """
    rng = np.random.default_rng(seed)
    corrupted = mdata.copy()
    orig = {'rna': None, 'splice': None}

    #Gene Expression corruption
    if pct_rna > 0:
        X = corrupted['rna'].layers['raw_counts']
        # dense matrix
        arr = X.toarray() if sp.issparse(X) else X.copy()
        nonzero = np.argwhere(arr != 0) #find parts of gene expression where its not already 0 for counts
        n_remove = int(len(nonzero) * pct_rna) #how many we should remove
        sel = rng.choice(len(nonzero), size=n_remove, replace=False) #randomly (according to the seed) choose indices
        coords = nonzero[sel] #get the coordinates from the non zero matrix
        values = arr[coords[:,0], coords[:,1]].copy() #storing the original values
        # mask
        arr[coords[:,0], coords[:,1]] = 0
        # assign back
        corrupted['rna'].layers['raw_counts'] = sp.csr_matrix(arr) if sp.issparse(X) else arr #put it back in the the mudata
        orig['rna'] = (coords, values) #storing the og values into the original data dictionary

    #Splicing corruption
    if pct_splice > 0:
        sp_mod = 'splicing'  #adjust if needed
        atse = corrupted[sp_mod].layers['cell_by_cluster_matrix'] 
        junc = corrupted[sp_mod].layers['cell_by_junction_matrix']
        ratio = corrupted[sp_mod].layers['junc_ratio']

        # to dense numpy
        atse_arr = atse.toarray() if sp.issparse(atse) else atse.copy()
        junc_arr = junc.toarray() if sp.issparse(junc) else junc.copy()
        # ensure ratio is a float numpy array
        if sp.issparse(ratio):
            ratio_arr = ratio.toarray().astype(float)
        else:
            ratio_arr = np.array(ratio, copy=True, dtype=float)

        # now this will work without TypeError
        valid = np.argwhere(
            (atse_arr > 0) &
            (junc_arr >= 0) &
            (~np.isnan(ratio_arr))
        ) #we have valid coordinates as the condition above
        n_remove = int(len(valid) * pct_splice)
        sel = rng.choice(len(valid), size=n_remove, replace=False)
        coords = valid[sel]
        orig_vals = np.vstack([
            atse_arr[coords[:,0], coords[:,1]],
            junc_arr[coords[:,0], coords[:,1]],
            ratio_arr[coords[:,0], coords[:,1]]
        ]).T
        # mask, meaning set those vlues to 0
        atse_arr[coords[:,0], coords[:,1]] = 0
        junc_arr[coords[:,0], coords[:,1]] = 0
        ratio_arr[coords[:,0], coords[:,1]] = 0
        # assign back
        corrupted[sp_mod].layers['cell_by_cluster_matrix'] = sp.csr_matrix(atse_arr) if sp.issparse(atse) else atse_arr
        corrupted[sp_mod].layers['cell_by_junction_matrix'] = sp.csr_matrix(junc_arr) if sp.issparse(junc) else junc_arr
        corrupted[sp_mod].layers['junc_ratio'] = ratio_arr
        orig['splice'] = (coords, orig_vals)

    return corrupted, orig

from scipy.stats import spearmanr
import numpy as np

def evaluate_imputation(
    original: tuple[np.ndarray, np.ndarray],
    imputed: np.ndarray
) -> dict[str, float]:
    """
    Compute MSE, median L1, and Spearman correlation between `imputed` and `original` at all stored coordinates in original (which stores data according to the below info).

    For GE (Gene Expression):
      - original = (coords, true_counts)
      - imputed is also counts

    For Splicing:
      - original = (coords, array of shape (N,3): [atse, true_junc_counts, true_ratio])
      - imputed is assumed to be the raw ratio p
      - reconstruct imputed counts = p * atse, then compare to true_junc_counts
    """
    coords, orig_vals = original
    # extract the imputed vals at exactly those coords
    imp_vals = imputed[coords[:, 0], coords[:, 1]]

    # detect splice vs GE by shape of orig_vals
    if orig_vals.ndim == 2 and orig_vals.shape[1] == 3:
        # splice case
        atse            = orig_vals[:, 0]
        true_junc       = orig_vals[:, 1]
        # we ignore orig_vals[:,2] (the original ratio) here
        # reconstruct counts from ratio
        imp_junc_counts = imp_vals * atse
        diff = imp_junc_counts - true_junc
        x1, x2 = true_junc, imp_junc_counts
    else:
        # GE case: orig_vals is a 1D array of counts
        true_counts = orig_vals
        diff = imp_vals - true_counts
        x1, x2 = true_counts, imp_vals

    mse     = np.mean(diff**2)
    med_l1  = np.median(np.abs(diff))
    rho, _  = spearmanr(x1, x2)

    return {'mse': mse, 'median_l1': med_l1, 'spearman': rho}


# %% [markdown]
# ## 3. Load  MuData
# 
# We assume this MuData has all necessary fields.

# %%
mdata = mu.read_h5mu(MUDATA_PATH)
print("MuData modalities loaded:", list(mdata.mod.keys()))
print(mdata)

# %% [markdown]
# ## 4. Corrupt MuData
print("Corrupting MData")
# %%
mdata_corr, orig_vals = corrupt_mudata(
    mdata,
    pct_rna=PCT_MISSING_RNA,
    pct_splice=PCT_MISSING_SPLICE,
    seed=SEED,
)
print("Original RNA masked entries:", orig_vals['rna'][0].shape[0] if orig_vals['rna'] else 0)
print("Original splice masked entries:", orig_vals['splice'][0].shape[0] if orig_vals['splice'] else 0)

# %% [markdown]
# ## 3. Set Up MultiVI‑Splice Model Using the Corrupted MuData Object
# 
# Use `setup_mudata` to register modalities. Here, adjust the keys to match those in your MuData.
# 
# For this example, we assume:
# - The GE AnnData (for gene expression) is under modality key `"rna"` with its raw counts in `"raw_counts"`.
# - The ATSE AnnData (for splicing) is used for raw junction counts and the two additional splicing layers.
#   

# %%
scvi.model.MULTIVISPLICE.setup_mudata(
    mdata_corr,
    batch_key="mouse.id",
    rna_layer="raw_counts",
    junc_ratio_layer="junc_ratio",
    atse_counts_layer="cell_by_cluster_matrix",
    junc_counts_layer="cell_by_junction_matrix",
    modalities={"rna_layer": "rna", "junc_ratio_layer": "splicing"},
)
model = scvi.model.MULTIVISPLICE(
    mdata_corr,
    n_genes=(mdata_corr['rna'].var['modality']=="Gene_Expression").sum(),
    n_junctions=(mdata_corr['splicing'].var['modality']=="Splicing").sum(),
)

model.view_anndata_setup()

# %% [markdown]
# ## 5. Train Model (with W&B Logging) and Save
# 
# If you don't want to log to W&B, remove `logger=wandb_logger`.
print("Beginning Training")
# %%
model.train(logger=wandb_logger)

model.save(MODEL_SAVE_DIR, overwrite=True)
print(f"Model saved to: {MODEL_SAVE_DIR}")

# %% [markdown]
# ## 6. Impute and Evaluate
# 
# Evaluate the trained model on imputation accuracy. 
print("Evaluating Imputation")
# %%
# get imputed expression and splicing
# get normalized expression (cells × genes array)
expr_norm = model.get_normalized_expression(return_numpy=True)
# get library sizes
lib = model.get_library_size_factors()['expression']  # NumPy array length n_obs
# reconstruct counts
imp_expr_counts = expr_norm * lib[:, None]

imp_spl = model.get_normalized_splicing(return_numpy=True, junction_list=None) #should be the raw ratios, ie, "p" from generative outputs

# evaluate only masked entries
metrics_rna = evaluate_imputation(orig_vals['rna'], imp_expr_counts.values)
metrics_spl = evaluate_imputation(orig_vals['splice'], imp_spl.values)


# log to W&B
wandb.log({
    'rna_mse': metrics_rna['mse'],
    'rna_med_l1': metrics_rna['median_l1'],
    'rna_spearman': metrics_rna['spearman'],
    'spl_mse': metrics_spl['mse'],
    'spl_med_l1': metrics_spl['median_l1'],
    'spl_spearman': metrics_spl['spearman'],
})

print(
    f"rna_mse: {metrics_rna['mse']}, "
    f"rna_med_l1: {metrics_rna['median_l1']}, "
    f"rna_spearman: {metrics_rna['spearman']}, "
    f"spl_mse: {metrics_spl['mse']}, "
    f"spl_med_l1: {metrics_spl['median_l1']}, "
    f"spl_spearman: {metrics_spl['spearman']}"
)

# %% [markdown]
# ## 7. Visualize Latent Representation (UMAP)
# 
# 1. Add the model's latent representation to `.obsm`.
# 2. Calculate neighbors/UMAP.
# 3. Plot with different color labels.

print("Plotting UMAPs")
# %%
MULTIVI_LATENT_KEY = "X_multivi"
mdata_corr.obsm[MULTIVI_LATENT_KEY] = model.get_latent_representation()

sc.pp.neighbors(mdata_corr, use_rep=MULTIVI_LATENT_KEY)
sc.tl.umap(mdata_corr, min_dist=0.2)

# Plot by different obs fields (save + log to W&B)
fields_to_plot = ["cell_type_grouped", "modality", "age", "sex"]
for field in fields_to_plot:
    if field in mdata_corr.obs.columns:
        print(f"Plotting UMAP for: {field}")
        ax = sc.pl.umap(mdata_corr, color=field, show=False)
        fig = ax.get_figure()
        fig_path = os.path.join(FIGURE_OUTPUT_DIR, f"umap_{field}.png")
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        wandb.log({f"umap_{field}": wandb.Image(fig_path)})
        plt.close(fig)
        print(f"Saved and logged: {fig_path}")
    else:
        print(f"Field '{field}' not found in .obs — skipping.")


# %% [markdown]
# Getting the latent representation and adding it as an obsm field called "X_multivi"
print("Script execution finished.")


