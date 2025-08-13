# %% [markdown]
# # Notebook for MuData Creation from GE and ATSE AnnData
#
# This notebook:
# 1. Reads and inspects ATSE and gene expression AnnData files.
# 2. Fixes NaNs in the splicing data and adds a numeric age column.
# 3. Creates modality-specific `.var` metadata.
# 4. Builds a MuData object containing only the shared `.obs` columns.
# 5. Splits into stratified train/test by (cell type, age) and writes out the results.

# %% [markdown]
# ## 0. Set Paths and Configuration

# %%
ROOT_PATH          = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/072025/"
ATSE_DATA_PATH     = ROOT_PATH + "aligned_splicing_data_20250730_164104.h5ad"
GE_DATA_PATH       = ROOT_PATH + "aligned_gene_expression_data_20250730_164104.h5ad"
OUTPUT_MUDATA_PATH = ROOT_PATH + "combined_ge_splice_20250730_164104.h5mu"

print("ATSE data path:", ATSE_DATA_PATH)
print("GE data path:  ", GE_DATA_PATH)
print("Output MuData path:", OUTPUT_MUDATA_PATH)

# %% [markdown]
# ## 1. Imports

# %%
import anndata as ad
import numpy as np
from scipy.sparse import csr_matrix, issparse
import mudata as mu
from sklearn.model_selection import train_test_split

# %% [markdown]
# ## 2. Load and Preprocess AnnData

# %%
# Load modalities
atse = ad.read_h5ad(ATSE_DATA_PATH)
ge   = ad.read_h5ad(GE_DATA_PATH)

# Rescale GE length‐norm and recompute library size
ge.layers["length_norm"] = (
    ge.layers["length_norm"] * np.median(ge.var["mean_transcript_length"])
)
ge.layers["length_norm"].data = np.floor(ge.layers["length_norm"].data)
ge.obsm["X_library_size"] = ge.layers["length_norm"].sum(axis=1)

# %% [markdown]
# ## 3. Parse Numeric Age

# %%
for adata in (atse, ge):
    adata.obs["age_numeric"] = (
        adata.obs["age"]
        .str.rstrip("m")      # drop trailing 'm'
        .astype(int)          # convert to integer months
    )

# %% [markdown]
# ## 4. Create `.var` Metadata

# %%
ge.var["ID"]       = ge.var["gene_id"]
ge.var["modality"] = "Gene_Expression"

atse.var["ID"]       = atse.var["junction_id"]
atse.var["modality"] = "Splicing"

# %% [markdown]
# ## 5. Compute Splicing `junc_ratio` and `psi_mask`

# %%
# Ensure sparse then compute ratio
junc = atse.layers["cell_by_junction_matrix"]
clus = atse.layers["cell_by_cluster_matrix"]

if not issparse(junc):
    junc = csr_matrix(junc)
if not issparse(clus):
    clus = csr_matrix(clus)

# Build mask (1 where cluster > 0)
mask = clus.copy()
mask.data = np.ones_like(mask.data, dtype=np.uint8)
atse.layers["psi_mask"] = mask

# Compute junc_ratio = junc / clus (NaN→0)
cj = junc.toarray()
cc = clus.toarray()
ratio = np.divide(cj, cc, out=np.zeros_like(cj, float), where=(cc != 0))
atse.layers["junc_ratio"] = ratio

# %% [markdown]
# ## 6. Assemble MuData with Shared `.obs` Columns

import pandas as pd

# 1) Compute the union of obs‐columns
all_cols = ge.obs.columns.union(atse.obs.columns)
print(all_cols)

# 2) Build a new DataFrame, preferring the 'rna' values when a column exists in both
df = pd.DataFrame(index=ge.obs_names, columns=all_cols)

for col in all_cols:
    if col in ge.obs:
        df[col] = ge.obs[col]
    else:
        df[col] = atse.obs[col]

# 3) Assign it back to each AnnData and to mdata
ge.obs = df
atse.obs = df

mdata = mu.MuData({"rna": ge, "splicing": atse})
mdata.obsm["X_library_size"] = ge.obsm["X_library_size"]
mdata.obs = df

print(mdata)

# %% [markdown]
# ## 7. Stratified Train/Test Split by (Cell Type, Age)

# %%
cells       = mdata.obs_names.to_list()
cell_types  = mdata.obs["broad_cell_type"].values
ages        = mdata.obs["age_numeric"].values

# Build combined labels for stratification
labels = [f"{ct}_{age}" for ct, age in zip(cell_types, ages)]

train_cells, test_cells = train_test_split(
    cells,
    test_size=0.30,
    random_state=42,
    stratify=labels,
)

mdata_train = mdata[train_cells, :].copy()
mdata_test  = mdata[test_cells,  :].copy()

# %% [markdown]
# ## 8. Write Out Results

# %%
mdata_train.write(ROOT_PATH + "train_70_30_20250730.h5mu")
mdata_test.write( ROOT_PATH + "test_30_70_20250730.h5mu")
mdata.write(OUTPUT_MUDATA_PATH)

print(
    f"Training cells: {mdata_train.n_obs} | "
    f"Testing cells: {mdata_test.n_obs}"
)
