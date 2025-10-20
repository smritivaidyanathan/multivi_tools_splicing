#!/usr/bin/env python
import os
import inspect
import argparse

import scanpy as sc
import scvi
import wandb
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
from scipy import sparse
import torch
from sklearn.neighbors import NearestNeighbors

# ------------------------------
# 0. Default Paths (CLI-overridable)
# ------------------------------
DEFAULT_ANN_DATA = "/gpfs/commons/groups/knowles_lab/Karin/TMS_MODELING/DATA_FILES/SIMULATED/simulated_data_2025-03-12.h5ad"
DEFAULT_MODEL_DIR = "/gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/models"
DEFAULT_FIG_DIR = "/gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/figures"

# ------------------------------
# 1. Grab train() defaults
# ------------------------------
train_sig = inspect.signature(scvi.model.MULTIVISPLICE.train)
train_defaults = {
    name: param.default
    for name, param in train_sig.parameters.items()
    if name != 'self' and param.default is not inspect._empty
}

# ------------------------------
# 2. Grab __init__ defaults for model
# ------------------------------
init_sig = inspect.signature(scvi.model.MULTIVISPLICE.__init__)
init_defaults = {
    name: param.default
    for name, param in init_sig.parameters.items()
    if name not in ('self', 'adata') and param.default is not inspect._empty
}

# ------------------------------
# 3. Build argparse
# ------------------------------
parser = argparse.ArgumentParser("SpliceVI-MultiVI")
# paths
parser.add_argument(
    "--train_mdata_path", type=str, default=DEFAULT_ANN_DATA,
    help=f"Train MuData (.h5ad) input path (default: {DEFAULT_ANN_DATA})"
)
parser.add_argument(
    "--test_mdata_path", type=str, default=DEFAULT_ANN_DATA,
    help=f"Test MuData (.h5ad) input path (default: {DEFAULT_ANN_DATA})"
)
parser.add_argument(
    "--masked_test_mdata_path", type=str, default=DEFAULT_ANN_DATA,
    help=f"Test MuData (.h5ad) input path (default: {DEFAULT_ANN_DATA})"
)
parser.add_argument(
    "--model_dir", type=str, default=DEFAULT_MODEL_DIR,
    help=f"Directory to save trained model (default: {DEFAULT_MODEL_DIR})"
)
parser.add_argument(
    "--fig_dir", type=str, default=DEFAULT_FIG_DIR,
    help=f"Directory to save UMAP figures (default: {DEFAULT_FIG_DIR})"
)


# model init params
for name, default in init_defaults.items():
    arg_type = type(default) if default is not None else float
    parser.add_argument(
        f"--{name}", type=arg_type, default=None,
        help=f"{name} (default = {default!r})"
    )
# training params
for name, default in train_defaults.items():
    arg_type = type(default) if default is not None else float
    parser.add_argument(
        f"--{name}", type=arg_type, default=None,
        help=f"{name} (default = {default!r})"
    )
# UMAP color fields
parser.add_argument(
    "--umap_colors", nargs='+', default=["cell_type_grouped"],
    help="List of obs fields to color UMAP by"
)

args = parser.parse_args()

# ------------------------------
# 4. Prepare directories
# ------------------------------
os.makedirs(args.model_dir, exist_ok=True)
os.makedirs(args.fig_dir, exist_ok=True)

# ------------------------------
# 5. Initialize W&B
# ------------------------------
full_config = {**init_defaults, **train_defaults}
for key in list(full_config):
    val = getattr(args, key)
    if val is not None:
        full_config[key] = val
full_config.update({
    "train_mdata_path": args.train_mdata_path,
    "test_mdata_path": args.test_mdata_path,
    "model_dir": args.model_dir,
    "fig_dir": args.fig_dir,
    "umap_colors": args.umap_colors,
})
wandb.init(project="MLCB_SUBMISSION", config=full_config)
wandb_logger = WandbLogger(project="MLCB_SUBMISSION", config=full_config)

# ------------------------------
# 6. Load Training MuData & Preprocess
# ------------------------------
print(f"Loading Training MuData from {args.train_mdata_path}…")
import mudata as mu
mdata = mu.read_h5mu(args.train_mdata_path, backed = "r")

# Layer names
x_layer = "junc_ratio"
junction_counts_layer = "cell_by_junction_matrix"
cluster_counts_layer = "cell_by_cluster_matrix"
mask_layer = "psi_mask"

print(f"Found layers in training splicing modality AnnData: {list(mdata['splicing'].layers.keys())}")

print("Setting up SpliceVI PartialVAE…")

scvi.model.MULTIVISPLICE.setup_mudata(
    mdata,
    batch_key = None,
    size_factor_key="X_library_size",
    rna_layer="length_norm",
    junc_ratio_layer=x_layer,
    atse_counts_layer=cluster_counts_layer,
    junc_counts_layer=junction_counts_layer,
    psi_mask_layer=mask_layer,
    modalities={"rna_layer": "rna", "junc_ratio_layer": "splicing"},
)

# ------------------------------
# 7. Initialize model
# ------------------------------

if getattr(args, "n_latent", None) is not None:
    args.n_latent = int(args.n_latent)
    
model_kwargs = {
    name: getattr(args, name)
    for name in init_defaults
    if getattr(args, name) is not None
}

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

# count & log total parameters
total_params = sum(p.numel() for p in model.module.parameters())
print(f"Total model parameters: {total_params:,}")
wandb.log({"total_parameters": total_params})


# ------------------------------
# 8. Train
# ------------------------------
train_kwargs = {
    name: getattr(args, name)
    for name in train_defaults
    if getattr(args, name) is not None
}
print("Starting training with settings:", train_kwargs)
model.train(logger=wandb_logger, check_val_every_n_epoch=5, **train_kwargs)
model.save(args.model_dir, overwrite=True)
wandb.log({"model_saved_to": args.model_dir})

# ------------------------------
# 9. Compute UMAP
# ------------------------------
print("Computing latent representations and UMAPs…")

umap_color_key = "broad_cell_type"
cell_type_classification_key = "medium_cell_type" if "medium_cell_type" in mdata.obs else "broad_cell_type"

latent_spaces = {
    "joint":      model.get_latent_representation(),
    "expression": model.get_latent_representation(modality="expression"),
    "splicing":   model.get_latent_representation(modality="splicing"),
}

for name, Z in latent_spaces.items():
    key_latent = f"X_latent_{name}"
    key_nn     = f"neighbors_{name}"
    key_umap   = f"X_umap_{name}"   # store in .obsm under this name

    # write latent, build neighbors with a custom key
    mdata["rna"].obsm[key_latent] = Z
    sc.pp.neighbors(mdata["rna"], use_rep=key_latent, key_added=key_nn)

    # run UMAP; no key_added here. Use copy=True and pull out the embedding.
    sc.tl.umap(mdata["rna"], min_dist=0.1, neighbors_key=key_nn)
    mdata["rna"].obsm[key_umap] = mdata["rna"].obsm["X_umap"]

    print(f"Generating UMAP for latent space: {name}")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(13, 6))
    sc.pl.embedding(
        mdata["rna"],
        basis=key_umap,                 # <- custom basis lives in .obsm[key_umap]
        color=umap_color_key,
        legend_loc=None,
        frameon=True,
        legend_fontsize=10,
        show=False,
    )
    plt.title(f"UMAP by {umap_color_key} – {name}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = f"{args.fig_dir}/umap_{umap_color_key}_{name}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    wandb.log({f"umap_{umap_color_key}_{name}": wandb.Image(out_path)})
    plt.close()


    
    plt.figure(figsize=(6, 6))
    sc.pl.embedding(
        mdata["rna"],
        basis=key_umap,                 # <- custom basis lives in .obsm[key_umap]
        color=umap_color_key,
        legend_loc=None,
        frameon=True,
        legend_fontsize=10,
        show=False,
    )
    plt.title(f"UMAP by {umap_color_key} – {name}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = f"{args.fig_dir}/umap_{umap_color_key}_{name}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    wandb.log({f"umap_sqr_{umap_color_key}_{name}": wandb.Image(out_path)})
    plt.close()

    if cell_type_classification_key != umap_color_key:
        plt.figure(figsize=(13, 6))
        sc.pl.embedding(
            mdata["rna"],
            basis=key_umap,                 # <- custom basis lives in .obsm[key_umap]
            color=cell_type_classification_key,
            legend_loc="right margin",
            frameon=True,
            legend_fontsize=10,
            show=False,
        )
        plt.title(f"UMAP by {cell_type_classification_key} – {name}")
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        out_path = f"{args.fig_dir}/umap_{cell_type_classification_key}_{name}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        wandb.log({f"umap_{cell_type_classification_key}_{name}": wandb.Image(out_path)})
        plt.close()



print("All UMAP embeddings complete.")




# ------------------------------
# 9b. Unsupervised clustering and cross-space consistency
# ------------------------------
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_mutual_info_score

LEIDEN_RESOLUTION = 1.0
NN_OVERLAPS_KS = [2, 5, 15, 50, 100]

print("Running Leiden clustering on each latent space and computing consistency…")

cell_type_col = "broad_cell_type"
if "medium_cell_type" in mdata["rna"].obs:
    cell_type_col = "medium_cell_type"

# Helper: build neighbors and Leiden using the already-computed latent embedding stored in obsm
def run_leiden_on_basis(ad, basis_key: str, neigh_key: str, leiden_key: str, resolution: float):
    sc.pp.neighbors(ad, use_rep=basis_key, key_added=neigh_key)
    sc.tl.leiden(ad, neighbors_key=neigh_key, key_added=leiden_key, resolution=resolution)

# Compute Leiden per latent space with independent neighbor graphs
excl_multi_records = []  # collects {"space","category","count"}
spaces_order = ["expression", "splicing", "joint"]  # control plot order
leiden_keys = {}
for name in ["joint", "expression", "splicing"]:
    basis_key = f"X_latent_{name}"
    neigh_key = f"neighbors_{name}_leiden"
    leiden_key = f"leiden_{name}"
    run_leiden_on_basis(mdata["rna"], basis_key, neigh_key, leiden_key, LEIDEN_RESOLUTION)
    leiden_keys[name] = leiden_key

    # Log basic clustering stats
    n_cl = int(mdata["rna"].obs[leiden_key].nunique())
    wandb.log({f"clustering/{name}_leiden_n_clusters": n_cl})

    # Count clusters that are exclusive to one cell type vs spanning multiple
    lk = leiden_key
    cts_per_cluster = (
        mdata["rna"].obs.groupby(lk)[cell_type_col]  # or cell_type_col if you prefer
        .apply(lambda s: set(s.astype(str).values))
    )

    n_unique = sum(1 for s in cts_per_cluster.values if len(s) == 1)
    n_multi  = sum(1 for s in cts_per_cluster.values if len(s) > 1)

    # Log and stash for plotting later
    wandb.log({
        f"clusters/{name}_n_unique_one_celltype": int(n_unique),
        f"clusters/{name}_n_multi_celltypes": int(n_multi),
    })

    excl_multi_records.append({"space": name, "category": "Unique to one cell type", "count": int(n_unique)})
    excl_multi_records.append({"space": name, "category": "Multiple cell types", "count": int(n_multi)})


    plt.figure(figsize=(13, 6))
    sc.pl.embedding(
        mdata["rna"],
        basis="X_umap_joint",
        color=leiden_key,
        legend_loc="right margin",
        frameon=True,
        show=False,
    )
    plt.title(f"Joint UMAP colored by Leiden ({name})")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = f"{args.fig_dir}/umap_joint_colored_by_{name}_leiden.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    wandb.log({f"clustering/umap_joint_colored_by_{name}_leiden": wandb.Image(out_path)})
    plt.close()

# ---- ADD: bar plot of number of sub-clusters per cell type (top 20 by size) ----
cell_type_for_bars = cell_type_col  # from earlier logic selecting medium or broad
obs = mdata["rna"].obs

# Determine top-20 cell types by population
ct_counts = obs[cell_type_for_bars].value_counts()
top20_cts = ct_counts.head(20).index.tolist()

# Build dataframe of n_subclusters per cell type per space
records_sub = []
for space_name, leiden_key in leiden_keys.items():  # expression, splicing, joint
    sub_df = (
        obs.loc[obs[cell_type_for_bars].isin(top20_cts), [cell_type_for_bars, leiden_key]]
        .groupby(cell_type_for_bars)[leiden_key]
        .nunique()
        .rename("n_subclusters")
        .reset_index()
    )
    sub_df["space"] = space_name
    records_sub.append(sub_df)

sub_all = pd.concat(records_sub, ignore_index=True)

# Enforce space order and cell type order
space_order = ["expression", "splicing", "joint"]
sub_all["space"] = pd.Categorical(sub_all["space"], categories=space_order, ordered=True)
sub_all[cell_type_for_bars] = pd.Categorical(sub_all[cell_type_for_bars], categories=top20_cts, ordered=True)

# Plot grouped bars
plt.figure(figsize=(max(12, 0.6*len(top20_cts)), 6))
sns.barplot(
    data=sub_all.sort_values([cell_type_for_bars, "space"]),
    x=cell_type_for_bars,
    y="n_subclusters",
    hue="space"
)
plt.xticks(rotation=45, ha="right")
plt.xlabel(cell_type_for_bars)
plt.ylabel("Number of Leiden sub-clusters")
plt.title(f"Sub-clusters per cell type (top 20 by size, res={LEIDEN_RESOLUTION})")
plt.tight_layout()
out_path = f"{args.fig_dir}/bar_subclusters_top20_{cell_type_for_bars}_leiden_res_{LEIDEN_RESOLUTION}.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
wandb.log({"clustering/bar_subclusters_top20": wandb.Image(out_path)})
plt.close()

#PLOT PAR CHART FOR NUM UNIQUE + SHARED LEIDEN CLUSTERS
# One bar chart with two x-groups and three bars per group
ex_df = pd.DataFrame(excl_multi_records)
ex_df["space"] = pd.Categorical(ex_df["space"], categories=spaces_order, ordered=True)
ex_df["category"] = pd.Categorical(ex_df["category"], categories=["Unique to one cell type", "Multiple cell types"], ordered=True)

plt.figure(figsize=(8, 5))
sns.barplot(data=ex_df, x="category", y="count", hue="space")
plt.xlabel("")
plt.ylabel("Number of Leiden clusters")
plt.title("Cluster exclusivity across spaces")
plt.tight_layout()
out_path = f"{args.fig_dir}/clusters_exclusive_vs_multi_by_space.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
wandb.log({"clustering/clusters_exclusive_vs_multi_by_space": wandb.Image(out_path)})
plt.close()


# Pairwise same-cluster consistency:
# For each cell i, let Cx(i) be its cluster in space X and Cy(i) in space Y.
# Define Sx(i) = set of indices in Cx(i) \ {i}, Sy(i) = set in Cy(i) \ {i}.
# Score cell-wise overlap = |Sx(i) ∩ Sy(i)| / max(|Sx(i)|, 1).
# Aggregate by cell type to find Percent consistent = 100*(mean_overlap).
pairs = [("expression", "joint"), ("splicing", "joint"), ("expression", "splicing")]

labels_ct = mdata["rna"].obs[cell_type_col].astype(str).values
n_cells = mdata["rna"].n_obs
idx_all = np.arange(n_cells, dtype=np.int32)

# Precompute cluster membership dicts
cluster_members = {}
for name in ["joint", "expression", "splicing"]:
    labs = mdata["rna"].obs[leiden_keys[name]].values
    # map cluster -> numpy array of member indices
    members = {}
    for cid, grp in pd.Series(idx_all).groupby(labs):
        members[cid] = grp.values.astype(np.int32, copy=False)
    cluster_members[name] = (labs, members)

# Compute per-cell overlap for each pair and summarize per cell type
heat_records = []
for a, b in pairs:
    labs_a, mem_a = cluster_members[a]
    labs_b, mem_b = cluster_members[b]

    overlap = np.empty(n_cells, dtype=np.float32)
    for i in range(n_cells):
        ca = labs_a[i]
        cb = labs_b[i]
        Sa = mem_a[ca]
        Sb = mem_b[cb]
        # exclude self quickly if present at the first occurrence
        # Using set operations on small arrays by converting to Python sets lazily
        # Note: sizes are typically moderate under Leiden, so this is fine
        if Sa.size <= 1:
            overlap[i] = np.nan
            continue
        # Remove i from Sa without copying whole array when i is first element is not guaranteed, so do a mask
        # For small sizes this is acceptable
        Sa_no_i = Sa[Sa != i]
        # Compute intersection size
        inter_sz = len(set(Sa_no_i).intersection(Sb))
        overlap[i] = inter_sz / float(Sa_no_i.size)

    # Attach per-cell result to obs for traceability
    key_cell = f"samecluster_overlap_{a}_vs_{b}"
    mdata["rna"].obs[key_cell] = overlap

    # Summary stats overall
    mean_ov = float(np.nanmean(overlap))
    median_ov = float(np.nanmedian(overlap))
    wandb.log({
        f"clustering/{a}_vs_{b}_samecluster_mean": mean_ov,
        f"clustering/{a}_vs_{b}_samecluster_median": median_ov,
    })

    # Aggregate by "tissue | cell_type" if tissue exists, else by cell_type
    if "tissue" in mdata["rna"].obs:
        labels_tissue = mdata["rna"].obs["tissue"].astype(str).values
        pair_label = np.char.add(np.char.add(labels_tissue, " | "), labels_ct)
    else:
        pair_label = labels_ct.copy()

    df_tmp = (
        pd.DataFrame({"pair_label": pair_label, "overlap": overlap})
        .groupby("pair_label", as_index=False)["overlap"].mean()
    )
    df_tmp["pct_consistent"] = (df_tmp["overlap"].fillna(0.0)) * 100.0
    df_tmp["pair"] = f"{a}_vs_{b}"
    heat_records.append(df_tmp[["pair_label", "pair", "pct_consistent"]])


# Single clustermap indexed by "tissue | cell_type" (or just cell_type if no tissue)
heat_df = pd.concat(heat_records, ignore_index=True)
heat_pivot = heat_df.pivot(index="pair_label", columns="pair", values="pct_consistent").fillna(0.0)

plt.close('all')
g = sns.clustermap(
    heat_pivot,
    cmap="viridis",
    vmin=0.0, vmax=100.0,
    metric="euclidean",
    method="average",
    figsize=(max(6, 0.25*heat_pivot.shape[1] + 4), max(6, 0.30*heat_pivot.shape[0] + 3)),
    row_cluster=True,
    col_cluster=False,
    annot=False
)
g.figure.suptitle(f"Percent consistent by tissue | cell type (Leiden, res={LEIDEN_RESOLUTION})", y=1.02, fontsize=12)
out_path = f"{args.fig_dir}/clustermap_pct_consistent_leiden_res_{LEIDEN_RESOLUTION}.png"
g.figure.savefig(out_path, dpi=300, bbox_inches="tight")
wandb.log({"clustering/clustermap_pct_consistent": wandb.Image(out_path)})
plt.close(g.figure)


# Optional: global alignment sanity via AMI between clusterings
for a, b in pairs:
    ami = adjusted_mutual_info_score(
        mdata["rna"].obs[leiden_keys[a]].values,
        mdata["rna"].obs[leiden_keys[b]].values
    )
    wandb.log({f"clustering/{a}_vs_{b}_AMI": float(ami)})

del heat_records, heat_pivot, heat_df
gc.collect()


import gc
import numpy as np
from sklearn.neighbors import NearestNeighbors

print("Computing expression↔︎splicing k-NN overlap on training data…")
# ------------------------------
# 9c. Extended kNN overlap for multiple k and space pairs
# ------------------------------
print("Computing extended kNN overlap for multiple k and space pairs…")

# Use the latent spaces you already computed
Z_joint = latent_spaces["joint"]
Z_expr  = latent_spaces["expression"]
Z_sp    = latent_spaces["splicing"]

pairs_knn = [
    ("expression", Z_expr, "joint", Z_joint),
    ("splicing",   Z_sp,   "joint", Z_joint),
]

from sklearn.neighbors import NearestNeighbors
import numpy as np
import gc
import matplotlib.pyplot as plt
import scanpy as sc

# Normalize/validate ks
ks = NN_OVERLAPS_KS
kmax = max(ks)
n = Z_joint.shape[0]

def knn_indices(Z, k):
    nn = NearestNeighbors(n_neighbors=min(k+1, n), metric="euclidean", n_jobs=-1)
    nn.fit(Z)
    return nn.kneighbors(Z, return_distance=False)

def strip_self(idx_full, k):
    # drop potential self neighbor (first column) or trim to k
    if idx_full.shape[1] > 0:
        out = idx_full[:, 1:] if idx_full.shape[1] > k else idx_full[:, :k]
    else:
        out = idx_full
    if out.dtype != np.int32:
        out = out.astype(np.int32, copy=False)
    return out

# Precompute neighbor indices once at kmax+1 for each space
idx_cache = {}
for name, Z in [("expression", Z_expr), ("splicing", Z_sp), ("joint", Z_joint)]:
    idx = knn_indices(Z, kmax)  # shape n x (kmax+1)
    idx_cache[name] = idx.astype(np.int32, copy=False)

# For each pair and each k, compute overlap counts, attach to obs, log stats, and plot UMAPs
for a_name, Za, b_name, Zb in pairs_knn:
    idx_a_full = idx_cache[a_name]
    idx_b_full = idx_cache[b_name]

    for k in ks:
        # slice to current k and drop self
        idx_a = strip_self(idx_a_full[:, :min(k+1, idx_a_full.shape[1])], k)
        idx_b = strip_self(idx_b_full[:, :min(k+1, idx_b_full.shape[1])], k)

        # Per-cell overlap size divided by k
        overlap_frac = np.empty(n, dtype=np.float32)
        for i in range(n):
            a = idx_a[i]
            b = idx_b[i]
            if a.size == 0:
                overlap_frac[i] = np.nan
                continue
            inter_sz = np.intersect1d(a, b, assume_unique=False).size
            overlap_frac[i] = inter_sz / float(min(k, a.size))

        # Attach and summarize
        key_obs = f"nn_overlap_{a_name}_vs_{b_name}_k{k}_pct"
        mdata["rna"].obs[key_obs] = (100.0 * overlap_frac).astype(np.float32, copy=False)

        vals = mdata["rna"].obs[key_obs].values
        mean_pct   = float(np.nanmean(vals))
        median_pct = float(np.nanmedian(vals))
        p10        = float(np.nanquantile(vals, 0.10))
        p90        = float(np.nanquantile(vals, 0.90))

        wandb.log({
            f"knn-extended/{a_name}_vs_{b_name}_k{k}_mean_pct": mean_pct,
            f"knn-extended/{a_name}_vs_{b_name}_k{k}_median_pct": median_pct,
            f"knn-extended/{a_name}_vs_{b_name}_k{k}_p10_pct": p10,
            f"knn-extended/{a_name}_vs_{b_name}_k{k}_p90_pct": p90,
        })

        # Joint-UMAP overlays (wide + square), colored by current pairwise overlap metric
        if "X_umap_joint" in mdata["rna"].obsm_keys():
            # Wide
            plt.figure(figsize=(13, 6))
            sc.pl.embedding(
                mdata["rna"],
                basis="X_umap_joint",
                color=key_obs,
                legend_loc=None,
                frameon=True,
                color_map="viridis",
                show=False,
            )
            plt.title(f"Joint UMAP — {a_name}↔{b_name} k={k} overlap (%)")
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            out_path = f"{args.fig_dir}/umap_joint_{a_name}_vs_{b_name}_k{k}.png"
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            wandb.log({f"knn-extended/umap_joint_{a_name}_vs_{b_name}_k{k}": wandb.Image(out_path)})
            plt.close()

            # Square
            plt.figure(figsize=(6, 6))
            sc.pl.embedding(
                mdata["rna"],
                basis="X_umap_joint",
                color=key_obs,
                legend_loc=None,
                frameon=True,
                color_map="viridis",
                show=False,
            )
            plt.title(f"Joint UMAP — {a_name}↔{b_name} k={k} overlap (%)")
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            out_path = f"{args.fig_dir}/umap_sqr_joint_{a_name}_vs_{b_name}_k{k}.png"
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            wandb.log({f"knn-extended/umap_sqr_joint_{a_name}_vs_{b_name}_k{k}": wandb.Image(out_path)})
            plt.close()

        # Clean up per-k temporaries
        del overlap_frac
        gc.collect()

# Free large neighbor caches
del idx_cache
gc.collect()


# ------------------------------
# 10. Unified Evaluation on TRAIN only
# ------------------------------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, silhouette_score
)
from scipy.stats import spearmanr

# ---- ADD BEFORE evaluate_split ----
AGE_R2_RECORDS = []  # collects dicts for all datasets and spaces
MIN_GROUP_N = 25     # skip very small groups for stable estimates


def evaluate_split(name: str, mdata, mask_coords=None, Z_type="joint"):
    print(f"\n=== Evaluating {name.upper()}-{Z_type} split ===")
    Z = model.get_latent_representation(adata=mdata, modality=Z_type)

    # PCA 90% var
    n_comp_max = min(Z.shape[0], Z.shape[1])
    pca = PCA(n_components=n_comp_max, svd_solver="full").fit(Z)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    pcs_90 = int(np.searchsorted(cum_var, 0.90) + 1)
    print(f"[{name}-{Z_type}] PCs for 90% variance: {pcs_90}/{Z.shape[1]}")
    wandb.log({
        f"real-{name}-{Z_type}/pca_n_components_90var": pcs_90,
        f"real-{name}-{Z_type}/pca_total_dim": Z.shape[1],
        f"real-{name}-{Z_type}/pca_var90_ratio": pcs_90 / Z.shape[1]
    })

    # Silhouette on broad and medium
    labels_broad = mdata.obs[umap_color_key].astype(str).values
    sil_broad = silhouette_score(Z, labels_broad)
    wandb.log({f"real-{name}-{Z_type}/{umap_color_key}-silhouette_score": sil_broad})

    labels_med = mdata.obs[cell_type_classification_key].astype(str).values
    sil_med = silhouette_score(Z, labels_med)
    wandb.log({f"real-{name}-{Z_type}/{cell_type_classification_key}-silhouette_score": sil_med})

    # LR classification on medium cell type
    Z_tr, Z_ev, y_tr, y_ev = train_test_split(Z, labels_med, test_size=0.2, random_state=0)
    clf = LogisticRegression(max_iter=1000).fit(Z_tr, y_tr)
    y_pred = clf.predict(Z_ev)
    wandb.log({
        f"real-{name}-{Z_type}/accuracy": accuracy_score(y_ev, y_pred),
        f"real-{name}-{Z_type}/precision": precision_score(y_ev, y_pred, average="weighted", zero_division=0),
        f"real-{name}-{Z_type}/recall": recall_score(y_ev, y_pred, average="weighted", zero_division=0),
        f"real-{name}-{Z_type}/f1_score": f1_score(y_ev, y_pred, average="weighted", zero_division=0),
    })

    # Age regression overall
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    if "age_numeric" in mdata.obs:
        ages = mdata.obs["age_numeric"].astype(float).values
        X_latent = StandardScaler().fit_transform(Z)
        X_tr, X_ev, y_tr, y_ev = train_test_split(X_latent, ages, test_size=0.2, random_state=0)
        ridge = RidgeCV(alphas=np.logspace(-2, 3, 20), cv=5).fit(X_tr, y_tr)
        r2_age = ridge.score(X_ev, y_ev)
        wandb.log({f"real-{name}-{Z_type}/age_r2": r2_age})

        # Age regression per (tissue, cell_type) pairing
        if "tissue" in mdata.obs:
            tissue_series = mdata.obs["tissue"].astype(str)
            ct_series = mdata.obs[cell_type_classification_key].astype(str)
            pair = tissue_series + " | " + ct_series
            pair_unique = pair.unique()

            for p in pair_unique:
                idx = np.where(pair.values == p)[0]
                if idx.size < MIN_GROUP_N:
                    continue
                # prepare group data
                Zg = X_latent[idx]
                yg = ages[idx]
                # skip degenerate variance
                if np.std(yg) == 0.0:
                    continue
                Ztr, Zev, ytr, yev = train_test_split(Zg, yg, test_size=0.2, random_state=0)
                try:
                    rg = RidgeCV(alphas=np.logspace(-2, 3, 20), cv=5).fit(Ztr, ytr)
                    r2g = rg.score(Zev, yev)
                except Exception:
                    continue
                AGE_R2_RECORDS.append({
                    "dataset": name,          # "train" or "test"
                    "space": Z_type,          # "joint","expression","splicing"
                    "pair": p,                # "tissue | celltype"
                    "tissue": p.split(" | ", 1)[0],
                    "cell_type": p.split(" | ", 1)[1],
                    "r2": float(r2g),
                    "n": int(idx.size),
                })


# run evaluation on training data
evaluate_split("train", mdata, Z_type="joint")
evaluate_split("train", mdata, Z_type="expression")
evaluate_split("train", mdata, Z_type="splicing")

# free training data from memory
print("Cleaning up training data from memory…")
del mdata
torch.cuda.empty_cache()

# ------------------------------
# 11. TEST split + masked‐junction imputation
# ------------------------------
print("\nLoading TEST MuData for evaluation and imputation…")
mdata = mu.read_h5mu(args.test_mdata_path, backed = "r")

print("Setting up SpliceVI PartialVAE…")
scvi.model.MULTIVISPLICE.setup_mudata(
    mdata,
    batch_key = None,
    size_factor_key="X_library_size",
    rna_layer="length_norm",
    junc_ratio_layer=x_layer,
    atse_counts_layer=cluster_counts_layer,
    junc_counts_layer=junction_counts_layer,
    psi_mask_layer=mask_layer,
    modalities={"rna_layer": "rna", "junc_ratio_layer": "splicing"},
)

# real-test evaluation
evaluate_split("test", mdata, Z_type="joint")
evaluate_split("test", mdata, Z_type="expression")
evaluate_split("test", mdata, Z_type="splicing")

# ---- ADD: materialize age R2 dataframe and plot train/test heatmaps ----
import itertools

if len(AGE_R2_RECORDS) > 0:
    age_df = pd.DataFrame(AGE_R2_RECORDS)
    # Save a CSV for downstream reuse
    csv_path = f"{args.fig_dir}/age_r2_by_tissue_celltype_train_test.csv"
    age_df.to_csv(csv_path, index=False)
    wandb.log({"age_r2/records_csv_path": csv_path})

    # For each dataset, build a pivot with rows=pairing, cols=space, values=r2
    for dset in ["train", "test"]:
        sub = age_df[age_df["dataset"] == dset].copy()
        if sub.empty:
            continue
        # Optional: keep only pairs with at least MIN_GROUP_N to match what we computed
        # Already enforced in collection, but keep for clarity

        # Sort pairs by max R2 to place informative ones on top
        
else:
    print("No age R² pairing records collected. Skipping heatmap.")


# ------------------------------
# 11. TEST split + masked‐ATSE imputation
# ------------------------------
del mdata
torch.cuda.empty_cache()

print(f"\n=== Masked-ATSE imputation on TEST using {args.masked_test_mdata_path} ===")
MASK_FRACTION = 0.2  # fraction of ATSEs to mask
mdata = mu.read_h5mu(args.masked_test_mdata_path, backed = "r")
ad_masked = mdata["splicing"]

print(f"Setting up SpliceVI PartialVAE… ")
scvi.model.MULTIVISPLICE.setup_mudata(
    mdata,
    batch_key = None,
    size_factor_key="X_library_size",
    rna_layer="length_norm",
    junc_ratio_layer=x_layer,
    atse_counts_layer=cluster_counts_layer,
    junc_counts_layer=junction_counts_layer,
    psi_mask_layer=mask_layer,
    modalities={"rna_layer": "rna", "junc_ratio_layer": "splicing"},
)

from scipy import sparse

# 6) run imputation in cell-index batches using args.batch_size
print("Step 6: running batched imputation and computing correlations")

model.module.eval()

# Ensure CSR for fast row slicing
masked_orig = ad_masked.layers["junc_ratio_masked_original"]
if not sparse.isspmatrix_csr(masked_orig):
    masked_orig = sparse.csr_matrix(masked_orig)

bin_mask = ad_masked.layers["junc_ratio_masked_bin_mask"]
if not sparse.isspmatrix_csr(bin_mask):
    bin_mask = sparse.csr_matrix(bin_mask)

n_cells = bin_mask.shape[0]
bs = int(args.batch_size) if getattr(args, "batch_size", None) else 512
assert bs > 0, "batch_size must be positive"

orig_all = []
pred_all = []
pairs_total = 0

for start in range(0, n_cells, bs):
    stop = min(start + bs, n_cells)
    # find masked positions for these rows only
    submask = bin_mask[start:stop]                # CSR
    sub_r, sub_c = submask.nonzero()             # local rows, global cols
    if sub_r.size == 0:
        continue

    # decode only these cells, using your existing batch_size for internal minibatching too
    idx = np.arange(start, stop, dtype=np.int64)
    with torch.inference_mode():
        decoded_batch = model.get_normalized_splicing(
            adata=mdata,
            indices=idx,
            return_numpy=True,
            batch_size=bs,          # use args.batch_size here
        )                           # shape: (stop-start, n_junc)

    # ground truth at masked positions
    orig_vals_b = masked_orig[start:stop][:, sub_c][sub_r, np.arange(sub_r.size)].A1
    # predictions at same positions
    pred_vals_b = decoded_batch[sub_r, sub_c]

    orig_all.append(orig_vals_b.astype(np.float32, copy=False))
    pred_all.append(pred_vals_b.astype(np.float32, copy=False))
    pairs_total += orig_vals_b.size

    # free batch ASAP
    del decoded_batch
    torch.cuda.empty_cache()

if pairs_total == 0:
    print("[impute-test] No masked entries found; skipping correlation.")
else:
    orig_all = np.concatenate(orig_all, dtype=np.float32)
    pred_all = np.concatenate(pred_all, dtype=np.float32)

    pearson_m  = float(np.corrcoef(orig_all, pred_all)[0, 1])
    from scipy.stats import spearmanr
    spearman_m = float(spearmanr(orig_all, pred_all, nan_policy="omit")[0])

    print(f"[impute-test] masked-ATSE PSI corr — Pearson: {pearson_m:.4f}, Spearman: {spearman_m:.4f}  (n={pairs_total})")
    wandb.log({
        "impute-test/psi_pearson_corr_masked_atse": pearson_m,
        "impute-test/psi_spearman_corr_masked_atse": spearman_m,
        "impute-test/n_masked_entries": int(pairs_total),
        "impute-test/batch_size_imputation": bs,
    })





# ------------------------------
# 12. Finish
# ------------------------------
print("Pipeline complete. Finishing W&B run.")
wandb.finish()
