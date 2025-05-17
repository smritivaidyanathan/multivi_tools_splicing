#!/usr/bin/env python

import os
import random
import mudata as mu
import scvi
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # for color palettes
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from scipy import sparse

# ---------------------------
# Config from environment
# ---------------------------
LATENT_EVAL_OUTDIR = os.environ.get("LATENT_EVAL_OUTDIR", "./latent_eval_output")
FIG_DIR = os.path.join(LATENT_EVAL_OUTDIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------
# User inputs
# ---------------------------
MUDATA_PATH = (
    "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/"
    "MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/052025/"
    "aligned__ge_splice_combined_20250513_035938.h5mu"
)
model_paths = {
    "dataset_batch_key": "/gpfs/commons/home/svaidyanathan/"
                         "multi_vi_splice_runs/"
                         "MultiVISpliceTraining_20250515_224315_job4683389/models",
    "mouse_id_batch_key":  "/gpfs/commons/home/svaidyanathan/"
                          "multi_vi_splice_runs/"
                          "MultiVISpliceTraining_20250515_225017_job4683680/models",
}
UMAP_GROUP       = "broad_cell_type"
CELL_TYPE_COLUMN = "broad_cell_type"
TARGET_CELL_TYPE = "Excitatory Neurons"
CLUSTER_NUMBERS  = [3, 5, 10]
RANDOM_SEED      = 42

# ---------------------------
# Load data & compute latents
# ---------------------------
print("Loading MuData from", MUDATA_PATH)
mdata = mu.read_h5mu(MUDATA_PATH)

# Build psi_mask layer on the splicing modality
splicing = mdata["splicing"]
cluster = splicing.layers["cell_by_cluster_matrix"]
if not sparse.isspmatrix_csr(cluster):
    cluster = sparse.csr_matrix(cluster)
mask = cluster.copy()
mask.data = np.ones_like(mask.data, dtype=np.uint8)
splicing.layers["psi_mask"] = mask
print("Now splicing layers:", splicing.layers.keys())

# Compute joint latent for each model
latents = {}
for name, path in model_paths.items():
    print(f"...loading model '{name}' from {path}")
    model = scvi.model.MULTIVISPLICE.load(path, adata=mdata)
    lat = model.get_latent_representation()
    latents[name] = lat
    print(f"   latent shape: {lat.shape}")
# ---------------------------
# Figure 1: UMAP side-by-side
# ---------------------------
print("\nGenerating UMAP side-by-side…")

# pull obs as plain strings
orig = mdata["rna"].obs[UMAP_GROUP].astype(str)
# pick the top 10
top10 = orig.value_counts().iloc[:10].index.tolist()
# build new column: keep top10, collapse everything else into 'Other'
new_vals = np.where(orig.isin(top10), orig, "Other")
# make it categorical (so Scanpy plots nicely)
categories = top10 + ["Other"]
mdata["rna"].obs["top10_or_other"] = pd.Categorical(new_vals, categories=categories)

# prepare palette: 10 distinct + grey for Other
base_colors = sns.color_palette(n_colors=10)
palette = {cat: col for cat, col in zip(top10, base_colors)}
palette["Other"] = "lightgrey"

# layout: one plot per model
import matplotlib.gridspec as gridspec
fig = plt.figure(constrained_layout=True, figsize=(8, 4))
gs  = gridspec.GridSpec(1, len(model_paths), figure=fig, width_ratios=[4,1])
axes = [fig.add_subplot(gs[0, i]) for i in range(len(model_paths))]

for ax, (name, lat) in zip(axes, latents.items()):
    ad = sc.AnnData(lat)
    ad.obs["top10_or_other"] = mdata["rna"].obs["top10_or_other"].values
    ad.obsm["X_input"] = lat

    sc.pp.neighbors(ad, use_rep="X_input")
    sc.tl.umap(ad, min_dist=0.1)

    show_legend = (ax is axes[0])
    sc.pl.umap(
        ad,
        color="top10_or_other",
        ax=ax,
        palette=palette,
        legend_loc="right margin" if show_legend else None,
        show=False,
    )
    if not show_legend:
        leg = ax.get_legend()
        if leg:
            leg.remove()

    ax.set_title(name)

fig.suptitle(f"UMAPs colored by top-10 '{UMAP_GROUP}'", y=1.02)
fig.savefig(
    os.path.join(FIG_DIR, f"umap_side_by_side_{UMAP_GROUP}.png"),
    dpi=300,
    bbox_inches="tight",
)

# right after your UMAP block, before computing silhouettes:
# ensure you have a categorical version of the original cell‐type labels
orig_cat = mdata["rna"].obs[UMAP_GROUP].astype("category")

# ---------------------------
# Figure 2: Silhouette scores
# ---------------------------
print("\nComputing silhouette scores…")
labels = orig_cat.cat.codes.values   # ← use orig_cat here
silhouette_scores = []
for name, lat in latents.items():
    score = silhouette_score(lat, labels)
    silhouette_scores.append({"model": name, "silhouette_score": score})
    print(f"  {name}: {score:.3f}")
df_sil = pd.DataFrame(silhouette_scores)
df_sil.to_csv(os.path.join(LATENT_EVAL_OUTDIR, "silhouette_scores.csv"), index=False)

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(df_sil["model"], df_sil["silhouette_score"])
ax.set_ylabel("Silhouette score")
ax.set_title(f"Silhouette scores for '{UMAP_GROUP}'")
plt.xticks(rotation=45, ha="right")
fig.tight_layout()
fig.savefig(
    os.path.join(FIG_DIR, f"silhouette_{UMAP_GROUP}.png"),
    dpi=300,
)

# ---------------------------
# Figure 3: Subcluster reproducibility across cell types
# ---------------------------
print("\nRunning subcluster reproducibility across cell types…")
records = []

# only consider cell types with at least 1000 cells
counts = mdata["rna"].obs[UMAP_GROUP].value_counts()
valid_ct = counts[counts >= 1000].index.tolist()
print(f"Cell types with ≥1000 cells: {valid_ct}")

def get_latents(model):
    return {
        "joint":     model.get_latent_representation(),
        "expression":model.get_latent_representation(modality="expression"),
        "splicing":  model.get_latent_representation(modality="splicing"),
    }

for ct in valid_ct:
    print(f"\n--- Processing cell type: {ct} (n={counts[ct]}) ---")
    # which obs indices belong to this type
    obs_ct = mdata["rna"].obs[UMAP_GROUP] == ct
    cells = mdata["rna"].obs_names[obs_ct].tolist()

    # random split
    random.seed(RANDOM_SEED)
    random.shuffle(cells)
    half = len(cells) // 2
    idx1 = np.isin(mdata["rna"].obs_names, cells[:half])
    idx2 = np.isin(mdata["rna"].obs_names, cells[half:])

    for name, path in model_paths.items():
        print(f"Model '{name}':")
        model = scvi.model.MULTIVISPLICE.load(path, adata=mdata)
        Z = get_latents(model)

        x1 = {m: Z[m][idx1] for m in Z}
        x2 = {m: Z[m][idx2] for m in Z}

        for k in CLUSTER_NUMBERS:
            km = KMeans(n_clusters=k, random_state=RANDOM_SEED)
            labels1 = km.fit_predict(x1["joint"])
            labels2 = km.predict(x2["joint"])
            for mod in ["joint", "expression", "splicing"]:
                clf = LogisticRegression(max_iter=200, random_state=RANDOM_SEED)
                clf.fit(x2[mod], labels2)
                acc = (clf.predict(x1[mod]) == labels1).mean()
                records.append({
                    "model":     name,
                    "cell_type": ct,
                    "modality":  mod,
                    "k":         k,
                    "accuracy":  acc,
                })
                print(f"  k={k:2d}, {mod:12s} → acc={acc:.3f}")

# assemble DataFrame and save
df_acc = pd.DataFrame.from_records(records)
df_acc.to_csv(
    os.path.join(LATENT_EVAL_OUTDIR, "subcluster_accuracies.csv"),
    index=False,
)
print("Saved subcluster results → subcluster_accuracies.csv")

# ---- plot mean±SD across cell types for each model ----
print("\nPlotting accuracy means ± SD across cell types…")
import matplotlib.gridspec as gridspec

n = len(model_paths)
fig = plt.figure(constrained_layout=True, figsize=(5*n, 4))
gs  = gridspec.GridSpec(1, n, figure=fig)
for i, name in enumerate(model_paths):
    ax = fig.add_subplot(gs[0, i])
    sub = df_acc[df_acc.model == name]
    # compute mean and std across cell types
    pivot = (
        sub
        .pivot_table(index="k", columns="modality", values="accuracy",
                     aggfunc=["mean", "std"])
    )
    means = pivot["mean"][["joint","expression","splicing"]]
    stds  = pivot["std"][["joint","expression","splicing"]]
    means.plot.bar(
        yerr=stds,
        ax=ax,
        capsize=4,
    )
    ax.set_title(name)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Accuracy")
    ax.tick_params(rotation=0)
    if i > 0:
        ax.get_legend().remove()

fig.suptitle("Subcluster reproducibility (mean ± SD over cell types)", y=1.02)
out3 = os.path.join(FIG_DIR, "subcluster_reproducibility.png")
fig.savefig(out3, dpi=300, bbox_inches="tight")
print("Saved plot →", out3)
