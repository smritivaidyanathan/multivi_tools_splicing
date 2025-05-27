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
ROOT_PATH = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/052025/"

ATSE_DATA_PATH = ROOT_PATH + "aligned_splicing_data_20250513_035938.h5ad"
GE_DATA_PATH = ROOT_PATH + "aligned_gene_expression_data_20250513_035938.h5ad"

# Configure if want to use HVGs only or full GE
HVG_only=False
OUTPUT_MUDATA_PATH = ROOT_PATH + "SUBSETTOP5CELLSTYPES_aligned__ge_splice_combined_20250513_035938_full_genes.h5mu"
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
import scanpy as sc

# %% [markdown]
# ## 2. Load ATSE and Gene Expression AnnData

# %%
atse_anndata = ad.read_h5ad(ATSE_DATA_PATH)
print("ATSE AnnData:", atse_anndata)

# %%
ge_anndata = ad.read_h5ad(GE_DATA_PATH)
print("GE AnnData:", ge_anndata)

# %%
# assert that cell_id is in the exact same order in both ge_anndata.obs and atse_anndata.obs
assert (ge_anndata.obs["cell_id"].values == atse_anndata.obs["cell_id"].values).all()
# assert that cell_id is in the exact same order in both ge_anndata.obs and atse_anndata.obs
assert (atse_anndata.obs["cell_id"].values == ge_anndata.obs["cell_id"].values).all()

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

# %%
# Ensure ge_anndata.var_names are gene names
ge_anndata.var_names = ge_anndata.var["gene_name"]
ge_anndata.var_names

# %%
# Do processing required to calculate most highly variable genes
# mitochondrial genes, "MT-" for human, "Mt-" for mouse
ge_anndata.var["mt"] = ge_anndata.var_names.str.startswith("mt-")
# ribosomal genes
ge_anndata.var["ribo"] = ge_anndata.var_names.str.startswith(("Rps", "Rpl"))
# hemoglobin genes
ge_anndata.var["hb"] = ge_anndata.var_names.str.contains("^Hb[^(P)]")
sc.pp.calculate_qc_metrics(
    ge_anndata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
)

# %%
# count number of cells with pct_counts_ribo > 40%
print(f"Number of cells with pct_counts_ribo > 40%: {(ge_anndata.obs['pct_counts_ribo'] > 40).sum()}")
# count number of cells with pct_counts_hb > 40%
print(f"Number of cells with pct_counts_hb > 40%: {(ge_anndata.obs['pct_counts_hb'] > 40).sum()}")
# count number of cells with pct_counts_mt > 40%
print(f"Number of cells with pct_counts_mt > 40%: {(ge_anndata.obs['pct_counts_mt'] > 40).sum()}")

# %%
# Step 1: Create a working copy of length-normalized data
ge_anndata.layers["working_norm"] = ge_anndata.layers["length_norm"].copy()

# Step 2: Normalize and log-transform the working layer
sc.pp.normalize_total(ge_anndata, layer="working_norm", inplace=True)
sc.pp.log1p(ge_anndata, layer="working_norm")

# Step 3: Compute highly variable genes on working layer
sc.pp.highly_variable_genes(
    ge_anndata, n_top_genes=5000, layer="working_norm", batch_key="dataset"
)

# Step 4: Subset to HVGs
if HVG_only:
    ge_anndata = ge_anndata[:, ge_anndata.var["highly_variable"]]

# Step 5: Assign unmodified length-normalized data to .X
ge_anndata.X = ge_anndata.layers["length_norm"]

print(f"The .X of ge_anndata is layer: {ge_anndata.X} corresponding to {ge_anndata.layers['length_norm']}")

# %%
# reset atse_anndata.obs
atse_anndata.obs.reset_index(drop=True, inplace=True)
ge_anndata.obs.reset_index(drop=True, inplace=True)

# assert that cell_id is in the exact same order in both ge_anndata.obs and atse_anndata.obs
assert (ge_anndata.obs["cell_id"].values == atse_anndata.obs["cell_id"].values).all()
assert (atse_anndata.obs["cell_id"].values == ge_anndata.obs["cell_id"].values).all()

from collections import Counter

# count how many cells per broad_cell_type
counts = Counter(atse_anndata.obs['broad_cell_type'])

# pick the top 5 most common
top5 = [ct for ct, _ in counts.most_common(5)]

# build a boolean mask for those top 5
mask = atse_anndata.obs['broad_cell_type'].isin(top5)

# subset both ATSE and GE to exactly those cells
atse_anndata = atse_anndata[mask].copy()
ge_anndata   = ge_anndata[mask].copy()

print(f"Keeping top 5 cell types: {top5}")
print(f"  • ATSE now has {atse_anndata.n_obs} cells")
print(f"  • GE   now has {ge_anndata.n_obs} cells")

# assert that cell_id is in the exact same order in both ge_anndata.obs and atse_anndata.obs
assert (ge_anndata.obs["cell_id"].values == atse_anndata.obs["cell_id"].values).all()
assert (atse_anndata.obs["cell_id"].values == ge_anndata.obs["cell_id"].values).all()

# %% [markdown]
# ## 3. Create `.var` DataFrames for Each Modality
# 
# Here we create modality-specific `.var` metadata. You might later use these to update the
# corresponding AnnData objects inside the MuData container.

# %%
gene_expr_var = pd.DataFrame(
    {
        "ID": ge_anndata.var["gene_id"],  # from the GE AnnData
        "modality": "Gene_Expression",
    },
    index=ge_anndata.var.index
)

splicing_var = pd.DataFrame(
    {
        "ID": atse_anndata.var["junction_id"],  # from the ATSE AnnData
        "modality": "Splicing",
    },
    index=atse_anndata.var.index
)

ge_anndata.var = gene_expr_var.copy()
atse_anndata.var = splicing_var.copy()

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


# %%
print(atse_anndata.layers['junc_ratio'])

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

# Set obs index to cell_id and ensure string dtype
ge_anndata.obs.index = ge_anndata.obs["cell_id"].astype(str)
atse_anndata.obs.index = atse_anndata.obs["cell_id"].astype(str)

# Also make sure the .obs["cell_id"] column itself is str, for safety
ge_anndata.obs["cell_id"] = ge_anndata.obs["cell_id"].astype(str)
atse_anndata.obs["cell_id"] = atse_anndata.obs["cell_id"].astype(str)
print(f"Final number of genes: {len(ge_anndata.var)}")
print(f"Final number of junctions: {len(atse_anndata.var)}")

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

# %% [markdown]
# ## 7. Write Out the Final MuData Object
# 
# The combined MuData object is now ready for use with `MULTIVISPLICE`. Save it as an H5MU file.

# %%
mdata.write(OUTPUT_MUDATA_PATH)
print(f"MuData object written to {OUTPUT_MUDATA_PATH}")

# %% [markdown]
# ## 8. Verify the Output
# 
# Read the MuData object back in to ensure everything is correct.

# %%
mdata_loaded = mu.read_h5mu(OUTPUT_MUDATA_PATH)
print("Loaded MuData modalities:", list(mdata_loaded.mod.keys()))
print(mdata_loaded)

# to submit 
#conda activate LeafletSC
#cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/052025/
#script=/gpfs/commons/home/kisaev/multivi_tools_splicing/multivi_splice_utils/jupyter_notebooks/ann_data_subset_maker.py
#sbatch --mem=300G --partition=bigmem,cpu --wrap "python $script"
