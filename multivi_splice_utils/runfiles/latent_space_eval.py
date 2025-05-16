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

# ---------------------------
# User inputs
# ---------------------------

# 1. Your MuData file (shared by all models)
MUDATA_PATH = "/path/to/your/data/mouse_foundation_data.h5mu"

# 2. Models to evaluate
model_paths = {
    "dataset_batch_key": "/gpfs/…/jobXXXX/models",
    "mouse_id_batch_key": "/gpfs/…/jobYYYY/models",
    # add more as needed
}

# 3. Which obs field to color UMAPs by and compute silhouette on
UMAP_GROUP = "broad_cell_type"

# 4. Subcluster-reproducibility settings
CELL_TYPE_COLUMN = "broad_cell_type"   # major cell types column
TARGET_CELL_TYPE = None                  # e.g. "Microglia", or None for all cells
CLUSTER_NUMBERS = [2, 3, 5, 10, 20, 30, 40]

# 5. Random seed for reproducibility
RANDOM_SEED = 42

# 6. Where to save figures
FIG_DIR = "/gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/multivi_splice_utils/jupyter_notebooks/figures"
os.makedirs(FIG_DIR, exist_ok=True)


# ---------------------------
# 1. Load data & compute latents
# ---------------------------
print("Loading MuData from", MUDATA_PATH)
mdata = mu.read_h5mu(MUDATA_PATH)

latents = {}
for name, path in model_paths.items():
    print(f"...loading model '{name}' from {path}")
    model = scvi.model.MULTIVISPLICE.load(path, adata=mdata["rna"].to_adata())
    # compute joint latent
    lat = model.get_latent_representation()
    latents[name] = lat
    print(f"   latent shape: {lat.shape}")


# ---------------------------
# 2. Figure 1: UMAPs side-by-side
# ---------------------------
print("\nGenerating UMAP side-by-side…")
# ensure categorical
mdata["rna"].obs[UMAP_GROUP] = mdata["rna"].obs[UMAP_GROUP].astype("category")

n_models = len(model_paths)
fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), squeeze=False)

for i, (name, lat) in enumerate(latents.items()):
    ad = sc.AnnData(lat)
    ad.obs = mdata["rna"].obs.copy()
    ad.obsm["X_input"] = lat

    sc.pp.neighbors(ad, use_rep="X_input", show=False)
    sc.tl.umap(ad, min_dist=0.2, show=False)

    sc.pl.umap(
        ad,
        color=UMAP_GROUP,
        ax=axes[0, i],
        show=False,
        title=name,
        legend_loc="right margin",
    )

fig.suptitle(f"UMAPs colored by '{UMAP_GROUP}'", y=1.02)
fig.tight_layout()
out1 = os.path.join(FIG_DIR, f"umap_side_by_side_{UMAP_GROUP}.png")
fig.savefig(out1, dpi=300, bbox_inches="tight")
print("Saved ➜", out1)


# ---------------------------
# 3. Figure 2: Silhouette scores
# ---------------------------
print("\nComputing silhouette scores…")
labels = mdata["rna"].obs[UMAP_GROUP].cat.codes.values
sil_scores = []
for name, lat in latents.items():
    score = silhouette_score(lat, labels)
    sil_scores.append(score)
    print(f"  {name}: {score:.3f}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(list(model_paths.keys()), sil_scores)
ax.set_ylabel("Silhouette score")
ax.set_title(f"Silhouette scores for '{UMAP_GROUP}'")
plt.xticks(rotation=45, ha="right")
fig.tight_layout()
out2 = os.path.join(FIG_DIR, f"silhouette_{UMAP_GROUP}.png")
fig.savefig(out2, dpi=300)
print("Saved ➜", out2)


# ---------------------------
# 4. Figure 3: Subcluster reproducibility
# ---------------------------

def get_latents_by_modality(model, modality="joint"):
    if modality == "joint":
        return model.get_latent_representation()
    else:
        return model.get_latent_representation(modality=modality)

# Prepare result collection
records = []

print("\nRunning subcluster reproducibility…")
for name, path in model_paths.items():
    print(f"\nModel '{name}':")
    # reload with full mdata for modality extraction
    model = scvi.model.MULTIVISPLICE.load(path, adata=mdata)
    # extract all three latents
    Z = {
        "joint": get_latents_by_modality(model, "joint"),
        "expression": get_latents_by_modality(model, "expression"),
        "splicing": get_latents_by_modality(model, "splicing"),
    }

    # choose cells
    obs = mdata["rna"].obs
    if TARGET_CELL_TYPE:
        mask = obs[CELL_TYPE_COLUMN] == TARGET_CELL_TYPE
        cells = obs.index[mask].to_list()
    else:
        cells = obs.index.to_list()

    # shuffle & split
    random.seed(RANDOM_SEED)
    cells_shuf = random.sample(cells, len(cells))
    half = len(cells_shuf) // 2
    c1, c2 = cells_shuf[:half], cells_shuf[half:]
    idx1 = np.isin(mdata["rna"].obs_names, c1)
    idx2 = np.isin(mdata["rna"].obs_names, c2)

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

# assemble DataFrame
df = pd.DataFrame.from_records(records)

# plot figure 3
print("\nPlotting accuracy by modality…")
n_models = len(model_paths)
fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), squeeze=False)

for i, name in enumerate(model_paths):
    sub = df[df.model == name]
    pivot = sub.pivot(index="k", columns="modality", values="accuracy")
    pivot = pivot[["joint", "expression", "splicing"]]
    pivot.plot(kind="bar", ax=axes[0, i])
    axes[0, i].set_title(name)
    axes[0, i].set_xlabel("Number of clusters (k)")
    axes[0, i].set_ylabel("Classification accuracy")
    axes[0, i].legend(title="Modality")
    axes[0, i].tick_params(rotation=0)

fig.suptitle("Subcluster reproducibility across modalities", y=1.02)
fig.tight_layout()
out3 = os.path.join(FIG_DIR, "subcluster_reproducibility.png")
fig.savefig(out3, dpi=300, bbox_inches="tight")
print("Saved ➜", out3)
