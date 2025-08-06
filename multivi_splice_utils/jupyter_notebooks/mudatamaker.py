# %% [markdown]
# # Notebook for MuData Creation from GE and ATSE AnnData
# 
# This notebook:
# 1. Reads and inspects ATSE and gene expression AnnData files.
# 2. Fixes NaNs in the splicing data.
# 3. Creates modality-specific `.obs`, `.var`, and `.layers` for each AnnData.
# 4. Creates a MuData object with modalities “rna”, “junc_counts”, “cell_by_junction_matrix”, 
#     and “cell_by_cluster_matrix”.
# 5. Writes out the final MuData object for use with MULTIVISPLICE.

# %% [markdown]
# ## 0. Set Paths and Configuration

# %%
ROOT_PATH = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/072025/"

ATSE_DATA_PATH = ROOT_PATH + "aligned_splicing_data_20250730_164104.h5ad"
GE_DATA_PATH = ROOT_PATH + "aligned_gene_expression_data_20250730_164104.h5ad"
OUTPUT_MUDATA_PATH = ROOT_PATH + "SUBSETTOP5CELLSTYPES_aligned__ge_splice_combined_20250513_035938.h5mu"
REDO_JUNC_RATIO = False

print("ATSE data path:", ATSE_DATA_PATH)
print("GE data path:  ", GE_DATA_PATH)
print("Output MuData path:", OUTPUT_MUDATA_PATH)

# %% [markdown]
# ## 1. Imports

# %%
import anndata as ad
import pandas as pd
import scipy.sparse as sp
import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack
import h5py
import anndata as ad
import mudata as mu

# %% [markdown]
# ## 2. Load ATSE and Gene Expression AnnData

# %%
atse_anndata = ad.read_h5ad(ATSE_DATA_PATH)
print("ATSE AnnData:", atse_anndata)

# %%
ge_anndata = ad.read_h5ad(GE_DATA_PATH)
print("GE AnnData:", ge_anndata)
print(ge_anndata.layers["length_norm"])

# %%
# rescale by overall median transcript length (didn't do this in preprocessing of GE AnnData)
ge_anndata.layers["length_norm"] = ge_anndata.layers["length_norm"] * np.median(ge_anndata.var["mean_transcript_length"])
# make sure to round down to get integer counts (this is CSR)
ge_anndata.layers["length_norm"].data = np.floor(ge_anndata.layers["length_norm"].data)
print(ge_anndata.layers["length_norm"])

# %%
# Recalculate library size using length normalized counts
ge_anndata.obsm["X_library_size"] = ge_anndata.layers["length_norm"].sum(axis=1)
print(ge_anndata.obsm["X_library_size"])

# %% [markdown]
# ## 3. Create `.var` DataFrames for Each Modality
# 
# Here we create modality-specific `.var` metadata. You might later use these to update the
# corresponding AnnData objects inside the MuData container.

# %%
ge_anndata.var["ID"] = ge_anndata.var["gene_id"]
ge_anndata.var["modality"] = "Gene_Expression"

atse_anndata.var["ID"] = atse_anndata.var["junction_id"]
atse_anndata.var["modality"] = "Splicing"

# %% [markdown]
# ## 4. Create a Common `.obs` DataFrame
# 
# You can decide which AnnData’s `.obs` to use (or merge them) if both contain the same information.
# Here we assume ATSE and GE have matching `obs` indices; we take the ATSE `obs`.

# %%
common_obs = atse_anndata.obs.copy()
common_obs["modality"] = "paired"  # if needed; adjust as required
print("Common obs shape:", common_obs.shape)

# Update both AnnData objects:
ge_anndata.obs = common_obs.copy()
atse_anndata.obs = common_obs.copy()

# %% [markdown]
# ## 5. Compute or Fix Splicing `junc_ratio` Layer
# 
# Here we check if `junc_ratio` needs to be recomputed. It is computed as:
# `junc_ratio = cell_by_junction_matrix / cell_by_cluster_matrix`
# and any NaNs/Inf values are replaced by zeros.
# 

# %%
# %% [markdown]
# ### 5.1 Build junc_ratio + psi_mask on the filtered data

# %%
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, issparse
import gc

# grab the splicing modality
splicing = atse_anndata  # if you later rename it to 'splicing', otherwise: atse_anndata

cell_by_junc    = splicing.layers["cell_by_junction_matrix"]
cell_by_cluster = splicing.layers["cell_by_cluster_matrix"]

# 1) ensure CSR format
if not issparse(cell_by_junc):
    cell_by_junc = csr_matrix(cell_by_junc)
if not issparse(cell_by_cluster):
    cell_by_cluster = csr_matrix(cell_by_cluster)

# 2) build psi_mask (1 wherever cluster>0)
mask = cell_by_cluster.copy()
mask.data = np.ones_like(mask.data, dtype=np.uint8)
splicing.layers["psi_mask"] = mask

# 3) compute junc_ratio = junction / cluster, nan→0
cj = cell_by_junc.toarray()
cc = cell_by_cluster.toarray()

junc_ratio = np.divide(
    cj,
    cc,
    out=np.zeros_like(cj, dtype=float),
    where=(cc != 0),
)
# 4) assign back as dense or sparse (dense is fine)
splicing.layers["junc_ratio"] = junc_ratio

print("New splicing layers:", list(splicing.layers.keys()))
print(f"  junc_ratio shape: {junc_ratio.shape}, psi_mask nnz: {mask.nnz}")

# 5) cleanup
del cell_by_junc, cell_by_cluster, cj, cc, mask
gc.collect()


# %% [markdown]
# ## 6. Create a MuData Object
# 
# Instead of stacking into one AnnData, we create a MuData container.
# 
# For MULTIVISPLICE, the new setup expects modalities with the following keys:
# - `rna` : gene expression counts,
# - `junc_ratio` : raw splicing/junction count data,
# - `cell_by_junction_matrix` and `cell_by_cluster_matrix` as additional layers.
# 
# We can use the GE AnnData for gene expression and the ATSE AnnData for all splicing-related data.
# (If needed, make copies so that modalities are independent.)
# 
# 
# Option 1: Use the GE AnnData for RNA and the ATSE AnnData for splicing modalities.
# (You can also combine or pre-process further if desired.)

# %%
mdata = mu.MuData({
    "rna": ge_anndata,
    "splicing": atse_anndata
})

# assert "library_size" in ge_anndata.obs, "'library_size' not found in ge_anndata.obs"
mdata.obsm["X_library_size"] = ge_anndata.obsm["X_library_size"]

# # Confirm it's stored correctly
# print("Library size moved to mdata.obsm['library_size'] with shape:", mdata.obsm["library_size"].shape)


# List of shared obs fields to pull up
shared_obs_keys = [
    'cell_id', 'age', 'cell_ontology_class', 'mouse.id', 'sex', 'tissue', 'dataset', 'broad_cell_type', 'cell_id_index', 'cell_name', 'modality'
]

# We'll assume 'rna' modality has them all and they match 'splicing'
for key in shared_obs_keys:
    assert key in mdata["rna"].obs, f"{key} not found in 'rna' obs"
    assert key in mdata["splicing"].obs, f"{key} not found in 'splicing' obs"
    assert (mdata["rna"].obs[key] == mdata["splicing"].obs[key]).all(), f"{key} values differ between modalities"
    mdata.obs[key] = mdata["rna"].obs[key]
    
print("MuData object created with modalities:", list(mdata.mod.keys()))
print(mdata)

# %%
# %% [markdown]
# ## 8. Stratified train/test split

# %%
from sklearn.model_selection import train_test_split

# 1) Grab all cell IDs and their labels
cells      = mdata.obs_names.to_list()
cell_types = mdata.obs["broad_cell_type"].values

# 2) Split into train (70%) / test (30%) stratified by broad_cell_type
train_cells, test_cells = train_test_split(
    cells,
    test_size=0.30,
    random_state=42,
    stratify=cell_types,
)

# 3) Subset the MuData object
mdata_train = mdata[train_cells, :].copy()
mdata_test  = mdata[test_cells,  :].copy()

# 4) (Optional) If you only need AnnData for the RNA modality:
rna_train = mdata_train["rna"]
rna_test  = mdata_test["rna"]

# 5) Write out both splits
mdata_train.write(ROOT_PATH + "train_70_30_ge_splice_combined_20250730_164104.h5mu")
mdata_test.write( ROOT_PATH + "test_30_70_ge_splice_combined_20250730_164104.h5mu")

print(
    f"Training cells: {mdata_train.n_obs} ({len(train_cells)})\n"
    f"Testing  cells: {mdata_test.n_obs} ({len(test_cells)})"
)


# %% [markdown]
# ## 7. Write Out the Final MuData Object
# 
# The combined MuData object is now ready for use with `MULTIVISPLICE`. Save it as an H5MU file.

# %% [markdown]
# ## 8. Verify the Output
# 
# Read the MuData object back in to ensure everything is correct.


