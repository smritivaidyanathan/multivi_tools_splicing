# %% [markdown]
# #Notebook for Setting Up, Training, and Visualization of MultiVI-Splice

# %%
import jax
import jaxlib

print("jax version:", jax.__version__)
print("jaxlib version:", jaxlib.__version__)

import scvi #note that this should be the local copy of scvi, not the API
import h5py
import scanpy as sc
import anndata as ad
import pandas as pd
import scipy.sparse as sp
import numpy as np
import wandb
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(
    project="multivi-splice"  # your wandb project name
    # you can also add "save_dir", "tags", "notes", etc.
)

# %% [markdown]
# Loading in our stacked annData (created using the ann_data_maker notebook).

# %%
combined_adata = ad.read_h5ad('/home/sv2785/research_folder/ann_data/TMS_BRAINONLY_Combined_GE_ATSE.h5ad')


# %% [markdown]
# Loading in another annData that we used to create the stacked annData so we can add mouse.id as a obs.

# %%
reference_adata = sc.read("/home/sv2785/research_folder/ann_data/GE_Anndata_Object_BRAIN_only_20241105.h5ad")
print(reference_adata)
cell_id_to_cell_type = dict(zip(reference_adata.obs.index, reference_adata.obs["mouse.id"]))
combined_adata.obs["mouse.id"] = combined_adata.obs.index.map(cell_id_to_cell_type)
reference_adata = None
combined_adata.obs

# %%
combined_adata.obs
cell_id_to_cell_type = None

# %% [markdown]
# Calling setup_annData and then creating our MultiVISplice model with the batch key as mouse.id and adding our padded layers to be used in the loss function. 

# %%
scvi.model.MULTIVISPLICE.setup_anndata(combined_adata, batch_key="mouse.id", atse_counts='cell_by_cluster_matrix', junc_counts='cell_by_junction_matrix')

# %%

model = scvi.model.MULTIVISPLICE(
    combined_adata,
    n_genes=(combined_adata.var["modality"] == "Gene_Expression").sum(),
    n_junctions=(combined_adata.var["modality"] == "Splicing").sum(),
)

model.view_anndata_setup()

# %% [markdown]
# Training our model, saving, and loading back in so I don't have to retrain this every time. 

# %%
model.train(logger=wandb_logger)

# %%
model.save("/home/sv2785/research_folder/ann_data/", overwrite=True)

# %%
model = scvi.model.MULTIVISPLICE.load("/home/sv2785/research_folder/ann_data/", adata=combined_adata)

# %%
combined_adata

# %% [markdown]
# Getting our imputed input values for both splicing and gene expression, creating a df to hold them for later analysis in data_analysis.py

# %%

imputed_splicing_estimates = model.get_splicing_estimates()

# %%
imputed_splicing_estimates.to_hdf("imputed_dfs.h5", key="imputed_splicing_estimates", mode="w")

# %%
imputed_splicing_estimates = None
imputed_expression_estimates = model.get_normalized_expression()

# %%
imputed_expression_estimates.to_hdf("imputed_dfs.h5", key="imputed_expression_estimates", mode="a")

# %%
imputed_expression_estimates = None

# %% [markdown]
# Getting back our reference data, and adding some more important headers to our combined annData object that are useful for plotting. 

# %%
reference_adata = sc.read("/home/sv2785/research_folder/ann_data/GE_Anndata_Object_BRAIN_only_20241105.h5ad")
print(reference_adata)

cell_id_to_cell_type = dict(zip(reference_adata.obs.index, reference_adata.obs["mouse.id"]))
combined_adata.obs["mouse.id"] = combined_adata.obs.index.map(cell_id_to_cell_type)
cell_id_to_cell_type = dict(zip(reference_adata.obs.index, reference_adata.obs["age"]))
combined_adata.obs["age"] = combined_adata.obs.index.map(cell_id_to_cell_type)
cell_id_to_cell_type = dict(zip(reference_adata.obs.index, reference_adata.obs["sex"]))
combined_adata.obs["sex"] = combined_adata.obs.index.map(cell_id_to_cell_type)
cell_id_to_cell_type = dict(zip(reference_adata.obs.index, reference_adata.obs["cell_type_grouped"]))
combined_adata.obs["cell_type_grouped"] = combined_adata.obs.index.map(cell_id_to_cell_type)
cell_id_to_cell_type = dict(zip(reference_adata.obs.index, reference_adata.obs["cell_ontology_class"]))
combined_adata.obs["cell_ontology_class"] = combined_adata.obs.index.map(cell_id_to_cell_type)

# %%
combined_adata

# %% [markdown]
# Getting the latent representation and adding it as an obsm field called "X_multivi"

# %%
import scanpy as sc
MULTIVI_LATENT_KEY = "X_multivi"

combined_adata.obsm[MULTIVI_LATENT_KEY] = model.get_latent_representation()

# %% [markdown]
# Plotting latent spaces with different options. 

# %%
sc.pp.neighbors(combined_adata, use_rep=MULTIVI_LATENT_KEY)
sc.tl.umap(combined_adata, min_dist=0.2)
sc.pl.umap(combined_adata, color="modality")

# %%
reference_adata = sc.read("/home/sv2785/research_folder/ann_data/GE_Anndata_Object_BRAIN_only_20241105.h5ad")
print(reference_adata)
group = "cell_type_grouped"
cell_id_to_cell_type = dict(zip(reference_adata.obs.index, reference_adata.obs[group]))
combined_adata.obs[group] = combined_adata.obs.index.map(cell_id_to_cell_type)
sc.pl.umap(combined_adata, color=group)

# %% [markdown]
# This is an old latent space from an older model I trained where I didn't set the batch key to mouse.id. Here we can see there is a lot less distinct clustering for each cell type and instead we're seeing some more stratification. 

# %%
sc.pl.umap(combined_adata, color="cell_type_grouped")

# %% [markdown]
# Getting the GE Input Data UMAPs and putting them in our combined annData

# %%
obsm = reference_adata.obsm
umaps = obsm['X_umap']
combined_adata.obsm['GE_X_umap'] = reference_adata.obsm['X_umap']
print(combined_adata)

# %% [markdown]
# Writing out our combined annData so we can use it later (with our latent reps)

# %%
print(combined_adata)
combined_adata.write("/home/sv2785/research_folder/ann_data/MULTIVI_TMS_BRAINONLY_Combined_GE_ATSE.h5ad")

# %% [markdown]
# Plotting the GE UMap with some interesting groupings. 

# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
group = "sex"
cell_types = reference_adata.obs[group]

scatter = plt.scatter(umaps[:, 0], umaps[:, 1], 
                      c=cell_types.astype('category').cat.codes, 
                      cmap='tab10', alpha=0.2, s = 2)
legend_labels = cell_types.astype('category').cat.categories
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.tab10(i / len(legend_labels)), markersize=10) for i in range(len(legend_labels))]
plt.legend(legend_handles, legend_labels, title=f"Cell Type Group:{group}", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("Gene Expression UMAPs")
plt.show()

# %% [markdown]
# Again, this is the old model. We can see that there is less cohesity here.

# %%
reference_adata = sc.read("/home/sv2785/research_folder/ann_data/GE_Anndata_Object_BRAIN_only_20241105.h5ad")
print(reference_adata)
cell_id_to_cell_type = dict(zip(reference_adata.obs.index, reference_adata.obs["sex"]))
combined_adata.obs["sex"] = combined_adata.obs.index.map(cell_id_to_cell_type)
sc.pl.umap(combined_adata, color="cell_ontology_class")

# %%
sc.pl.umap(combined_adata, color="age")
sc.pl.umap(combined_adata, color="sex")


