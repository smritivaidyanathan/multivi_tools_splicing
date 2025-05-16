#!/usr/bin/env python

import os
import random
import mudata as mu
import scvi
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
MUDATA_PATH = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/052025/aligned__ge_splice_combined_20250513_035938.h5mu"
model_paths = {
    "dataset_batch_key": "/gpfs/commons/home/svaidyanathan/multi_vi_splice_runs/MultiVISpliceTraining_20250515_224315_job4683389/models",
    "mouse_id_batch_key": "/gpfs/commons/home/svaidyanathan/multi_vi_splice_runs/MultiVISpliceTraining_20250515_224315_job4683389/models",
}
UMAP_GROUP = "broad_cell_type"
CELL_TYPE_COLUMN = "broad_cell_type"
TARGET_CELL_TYPE = "Excitatory Neurons"
CLUSTER_NUMBERS = [3, 5, 10]
RANDOM_SEED = 42

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
# UMAP side-by-side
# ---------------------------
print("\nGenerating UMAP side-by-side…")
mdata["rna"].obs[UMAP_GROUP] = mdata["rna"].obs[UMAP_GROUP].astype("category")
fig, axes = plt.subplots(1, len(model_paths), figsize=(5 * len(model_paths), 4), squeeze=False)
for i, (name, lat) in enumerate(latents.items()):
    ad = sc.AnnData(lat)
    ad.obs = mdata["rna"].obs.copy()
    ad.obsm["X_input"] = lat
    sc.pp.neighbors(ad, use_rep="X_input")
    sc.tl.umap(ad, min_dist=0.1)
    sc.pl.umap(
        ad, color=UMAP_GROUP, ax=axes[0, i], title=name,
        legend_loc="right margin",
    )
fig.suptitle(f"UMAPs colored by '{UMAP_GROUP}'", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, f"umap_side_by_side_{UMAP_GROUP}.png"),
            dpi=300, bbox_inches="tight")

# ---------------------------
# Silhouette scores
# ---------------------------
print("\nComputing silhouette scores…")
labels = mdata["rna"].obs[UMAP_GROUP].cat.codes.values
silhouette_scores = []
for name, lat in latents.items():
    score = silhouette_score(lat, labels)
    silhouette_scores.append({"model": name, "silhouette_score": score})
    print(f"  {name}: {score:.3f}")

# Save silhouette scores
df_sil = pd.DataFrame(silhouette_scores)
df_sil.to_csv(os.path.join(LATENT_EVAL_OUTDIR, "silhouette_scores.csv"), index=False)

# Plot silhouette
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(df_sil["model"], df_sil["silhouette_score"])
ax.set_ylabel("Silhouette score")
ax.set_title(f"Silhouette scores for '{UMAP_GROUP}'")
plt.xticks(rotation=45, ha="right")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, f"silhouette_{UMAP_GROUP}.png"), dpi=300)

# ---------------------------
# Subcluster reproducibility
# ---------------------------
def get_latents_by_modality(model, modality="joint"):
    return model.get_latent_representation(modality=modality)

print("\nRunning subcluster reproducibility…")
records = []
for name, path in model_paths.items():
    print(f"\nModel '{name}':")
    model = scvi.model.MULTIVISPLICE.load(path, adata=mdata)
    Z = {
        "joint": get_latents_by_modality(model, "joint"),
        "expression": get_latents_by_modality(model, "expression"),
        "splicing": get_latents_by_modality(model, "splicing"),
    }

    obs = mdata["rna"].obs
    # subset to TARGET_CELL_TYPE if specified
    if TARGET_CELL_TYPE:
        cells = obs.index[obs[CELL_TYPE_COLUMN] == TARGET_CELL_TYPE].tolist()
    else:
        cells = obs.index.tolist()
    random.seed(RANDOM_SEED)
    random.shuffle(cells)
    half = len(cells) // 2
    idx1 = np.isin(mdata["rna"].obs_names, cells[:half])
    idx2 = np.isin(mdata["rna"].obs_names, cells[half:])

    x1 = {m: Z[m][idx1] for m in Z}
    x2 = {m: Z[m][idx2] for m in Z}

    for k in CLUSTER_NUMBERS:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED)
        labels1 = km.fit_predict(x1["joint"])
        labels2 = km.predict(x2["joint"])
        for mod in ["joint", "expression", "splicing"]:
            clf = LogisticRegression(max_iter=200, random_state=RANDOM_SEED)
            clf.fit(x2[mod], labels2)
            pred1 = clf.predict(x1[mod])
            acc = (pred1 == labels1).mean()
            records.append({
                "model": name,
                "modality": mod,
                "k": k,
                "accuracy": acc,
            })
            print(f"  k={k:3d}, {mod:10s} → acc={acc:.3f}")

# Save subcluster accuracies
df_acc = pd.DataFrame.from_records(records)
df_acc.to_csv(os.path.join(LATENT_EVAL_OUTDIR, "subcluster_accuracies.csv"), index=False)

# Plot accuracy by modality
print("\nPlotting accuracy by modality…")
fig, axes = plt.subplots(1, len(model_paths), figsize=(5 * len(model_paths), 4), squeeze=False)
for i, name in enumerate(model_paths):
    sub = df_acc[df_acc.model == name]
    pivot = sub.pivot(index="k", columns="modality", values="accuracy")[["joint", "expression", "splicing"]]
    pivot.plot(kind="bar", ax=axes[0, i])
    axes[0, i].set_title(name)
    axes[0, i].set_xlabel("Number of clusters (k)")
    axes[0, i].set_ylabel("Accuracy")
    axes[0, i].legend(title="Modality")
    axes[0, i].tick_params(rotation=0)

fig.suptitle("Subcluster reproducibility", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "subcluster_reproducibility.png"), dpi=300,
            bbox_inches="tight")
