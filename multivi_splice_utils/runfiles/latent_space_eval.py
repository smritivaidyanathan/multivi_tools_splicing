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

# ---------------------------
# Figure 1: one UMAP per model
# ---------------------------
import matplotlib.pyplot as plt

print("\nGenerating per-model UMAPs…")
for name, lat in latents.items():
    ad = sc.AnnData(lat)
    ad.obs["top10_or_other"] = mdata["rna"].obs["top10_or_other"].values
    ad.obsm["X_input"] = lat

    sc.pp.neighbors(ad, use_rep="X_input")
    sc.tl.umap(ad, min_dist=0.1)
    
    fig, ax = plt.subplots(figsize=(4,4))
    sc.pl.umap(
        ad,
        color="top10_or_other",
        ax=ax,
        palette=palette,
        legend_loc="right margin",
        show=False,
    )
    ax.set_title(f"UMAP — {name}")
    out = os.path.join(FIG_DIR, f"umap_{name}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")

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
valid_ct = counts[counts >= 5000].index.tolist()
print(f"Cell types with ≥5000 cells: {valid_ct}")

def get_latents(model):
    return {
        "joint":     model.get_latent_representation(),
        "expression":model.get_latent_representation(modality="expression"),
        "splicing":  model.get_latent_representation(modality="splicing"),
    }

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

records = []
TARGET_CELL_TYPE = "Excitatory Neurons"

for ct in valid_ct:
    print(f"\n--- Processing cell type: {ct} (n={counts[ct]}) ---")
    obs_ct = mdata["rna"].obs[UMAP_GROUP] == ct
    cells = mdata["rna"].obs_names[obs_ct].tolist()
    random.seed(RANDOM_SEED); random.shuffle(cells)
    half = len(cells) // 2
    idx1 = np.isin(mdata["rna"].obs_names, cells[:half])
    idx2 = np.isin(mdata["rna"].obs_names, cells[half:])

    for name, path in model_paths.items():
        print(f"Model '{name}':")
        model = scvi.model.MULTIVISPLICE.load(path, adata=mdata)
        Z = {
            "joint":      model.get_latent_representation(),
            "expression": model.get_latent_representation(modality="expression"),
            "splicing":   model.get_latent_representation(modality="splicing"),
        }
        x1 = {m: Z[m][idx1] for m in Z}
        x2 = {m: Z[m][idx2] for m in Z}

        for k in CLUSTER_NUMBERS:
            km      = KMeans(n_clusters=k, random_state=RANDOM_SEED)
            labels1 = km.fit_predict(x1["joint"])
            labels2 = km.predict(x2["joint"])

            for mod in ["joint", "expression", "splicing"]:
                # 1) train on half-2
                clf = LogisticRegression(max_iter=200, random_state=RANDOM_SEED)
                clf.fit(x2[mod], labels2)
                # 2) predict on half-1
                pred = clf.predict(x1[mod])

                # 3) compute your metrics
                acc  = (pred == labels1).mean()
                prec = precision_score(labels1, pred, average="weighted", zero_division=0)
                rec  = recall_score(labels1, pred, average="weighted", zero_division=0)
                f1   = f1_score(labels1, pred, average="weighted", zero_division=0)
                for metric, val in [
                    ("accuracy",  acc),
                    ("precision", prec),
                    ("recall",    rec),
                    ("f1",        f1),
                ]:
                    records.append({
                        "model":     name,
                        "cell_type": ct,
                        "modality":  mod,
                        "k":         k,
                        "metric":    metric,
                        "value":     val,
                    })

                print(f"  k={k:2d}, {mod:12s} → acc={acc:.3f}, prec={prec:.3f}, rec={rec:.3f}, f1={f1:.3f}")

                # 4) if this is your target cell‐type, also plot & save the confusion matrix:
                if ct == TARGET_CELL_TYPE:
                    cm   = confusion_matrix(labels1, pred)
                    disp = ConfusionMatrixDisplay(cm)
                    fig_cm, ax_cm = plt.subplots(figsize=(5,5))
                    disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
                    ax_cm.set_title(f"{ct} — {name} — {mod} — k={k}")
                    cm_out = os.path.join(
                        FIG_DIR,
                        f"confusion_{name}_{ct.replace(' ','_')}_{mod}_k{k}.png",
                    )
                    fig_cm.savefig(cm_out, dpi=300, bbox_inches="tight")
                    plt.close(fig_cm)
                    print(f"    saved confusion matrix → {cm_out}")


# assemble and save
df_acc = pd.DataFrame.from_records(records)
df_acc.to_csv(os.path.join(LATENT_EVAL_OUTDIR, "subcluster_metrics.csv"), index=False)
print("Saved all metrics → subcluster_metrics.csv")


# ---- plot mean±SD across cell types for each model ----

print("\nPlotting subcluster metrics…")
metrics = ["accuracy","precision","recall","f1"]
for metric in metrics:
    fig, axes = plt.subplots(1, len(model_paths), figsize=(5*len(model_paths), 4))
    for ax, name in zip(axes, model_paths):
        sub = df_acc[(df_acc.model==name) & (df_acc.metric==metric)]
        pivot = sub.pivot_table(
            index="k",
            columns="modality",
            values="value",
            aggfunc=["mean","std"],
        )
        means = pivot["mean"][["joint","expression","splicing"]]
        stds  = pivot["std"][["joint","expression","splicing"]]
        means.plot.bar(
            yerr=stds,
            capsize=4,
            ax=ax,
        )
        ax.set_title(f"{name} — {metric}")
        ax.set_xlabel("k")
        ax.set_ylabel(metric.title())
        ax.tick_params(rotation=0)
        if ax is not axes[0]:
            ax.get_legend().remove()
    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"subcluster_{metric}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")

