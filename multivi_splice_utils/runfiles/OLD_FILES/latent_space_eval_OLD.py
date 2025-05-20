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
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
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
TARGET_CELL_TYPES = ["Excitatory Neurons", "MICROGLIA"]
CLUSTER_NUMBERS  = [3, 5, 7]
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

# --------------------------------------------------------------------------
# Compute subcluster metrics + random baseline
# --------------------------------------------------------------------------

import scanpy as sc

def plot_latent_umap(latent: np.ndarray,
                     true_labels: np.ndarray,
                     pred_labels: np.ndarray,
                     out_prefix: str,
                     fig_dir: str,
                     n_neighbors: int = 15,
                     min_dist: float = 0.1):
    """
    Given a latent matrix (cells × dims), and two 1d label arrays (true vs pred),
    compute a UMAP on the latent, then save:
      - {fig_dir}/{out_prefix}_true_umap.png
      - {fig_dir}/{out_prefix}_pred_umap.png
    """
    ad = sc.AnnData(latent)
    ad.obsm["X_latent"] = latent
    ad.obs["true"] = true_labels.astype(str)
    ad.obs["pred"] = pred_labels.astype(str)

    sc.pp.neighbors(ad, use_rep="X_latent", n_neighbors=n_neighbors)
    sc.tl.umap(ad, min_dist=min_dist)

    for label_type in ["true", "pred"]:
        fig, ax = plt.subplots(figsize=(4, 4))
        sc.pl.umap(
            ad,
            color=label_type,
            ax=ax,
            show=False,
            legend_loc=None,
        )
        ax.set_title(f"UMAP ({out_prefix}) — {label_type}")
        fname = os.path.join(fig_dir, f"{out_prefix}_{label_type}_umap.png")
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)



print("\nEvaluating subcluster reproducibility…")
records = []

# select big cell‐types
counts = mdata["rna"].obs[UMAP_GROUP].value_counts()
valid_ct = counts[counts >= 5000].index.tolist()

for ct in valid_ct:
    print(f"  cell type {ct} (n={counts[ct]})")
    mask_ct = mdata["rna"].obs[UMAP_GROUP] == ct
    cells   = mdata["rna"].obs_names[mask_ct].tolist()
    random.seed(RANDOM_SEED); random.shuffle(cells)
    half = len(cells) // 2
    idx1 = np.isin(mdata["rna"].obs_names, cells[:half])
    idx2 = np.isin(mdata["rna"].obs_names, cells[half:])

    for name, path in model_paths.items():
        # reload to get all three modalities
        model = scvi.model.MULTIVISPLICE.load(path, adata=mdata)
        Z = {
            "joint":      model.get_latent_representation(),
            "expression": model.get_latent_representation(modality="expression"),
            "splicing":   model.get_latent_representation(modality="splicing"),
        }

        for k in CLUSTER_NUMBERS:
            # cluster on joint
            km       = KMeans(n_clusters=k, random_state=RANDOM_SEED)
            labels1  = km.fit_predict(Z["joint"][idx1])
            labels2  = km.predict    (Z["joint"][idx2])

            # three real modalities
            for mod in ["joint","expression","splicing"]:
                clf = LogisticRegression(max_iter=200, random_state=RANDOM_SEED)
                clf.fit(Z[mod][idx2], labels2)
                pred = clf.predict(Z[mod][idx1])

                for metric_name, func in [
                    ("accuracy",  lambda a,p: (p==a).mean()),
                    ("precision", lambda a,p: precision_score(a,p,average="weighted",zero_division=0)),
                    ("recall",    lambda a,p: recall_score(a,p,average="weighted",zero_division=0)),
                    ("f1",        lambda a,p: f1_score(a,p,average="weighted",zero_division=0)),
                ]:
                    val = func(labels1, pred)
                    records.append({
                        "model":     name,
                        "cell_type": ct,
                        "modality":  mod,
                        "k":         k,
                        "metric":    metric_name,
                        "value":     val,
                    })
                # optional: confusion matrices for some types
                if ct in TARGET_CELL_TYPES:
                    cm = confusion_matrix(labels1, pred)
                    disp = ConfusionMatrixDisplay(cm)
                    fig_cm, ax_cm = plt.subplots(figsize=(5,5))
                    disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
                    ax_cm.set_title(f"{ct} — {name} — {mod} — k={k}")
                    cm_out = os.path.join(FIG_DIR, f"conf_{name}_{ct.replace(' ','_')}_{mod}_k{k}.png")
                    fig_cm.savefig(cm_out, dpi=300, bbox_inches="tight")
                    plt.close(fig_cm)

                    out_pref = f"{name.replace(' ','_')}_{ct.replace(' ','_')}_{mod}_k{k}"
                    plot_latent_umap(
                        latent=Z[mod][idx1],
                        true_labels=labels1,
                        pred_labels=pred,
                        out_prefix=out_pref,
                        fig_dir=FIG_DIR,
                    )

            # random‐baseline modality
            rng = np.random.default_rng(RANDOM_SEED)
            Zrand = rng.standard_normal(Z["joint"].shape)
            clf   = LogisticRegression(max_iter=200, random_state=RANDOM_SEED)
            clf.fit(Zrand[idx2], labels2)
            pred  = clf.predict(Zrand[idx1])
            for metric_name, func in [
                ("accuracy",  lambda a,p: (p==a).mean()),
                ("precision", lambda a,p: precision_score(a,p,average="weighted",zero_division=0)),
                ("recall",    lambda a,p: recall_score(a,p,average="weighted",zero_division=0)),
                ("f1",        lambda a,p: f1_score(a,p,average="weighted",zero_division=0)),
            ]:
                val = func(labels1, pred)
                records.append({
                    "model":     name,
                    "cell_type": ct,
                    "modality":  "random",
                    "k":         k,
                    "metric":    metric_name,
                    "value":     val,
                })
            if ct in TARGET_CELL_TYPES:
                cm = confusion_matrix(labels1, pred)
                disp = ConfusionMatrixDisplay(cm)
                fig_cm, ax_cm = plt.subplots(figsize=(5,5))
                disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
                ax_cm.set_title(f"{ct} — {name} — rand — k={k}")
                cm_out = os.path.join(FIG_DIR, f"conf_{name}_{ct.replace(' ','_')}_rand_k{k}.png")
                fig_cm.savefig(cm_out, dpi=300, bbox_inches="tight")
                plt.close(fig_cm)

# save
df_acc = pd.DataFrame.from_records(records)
df_acc.to_csv(os.path.join(LATENT_EVAL_OUTDIR, "subcluster_metrics.csv"), index=False)

# --------------------------------------------------------------------------
# Plot per‐model line charts including random
# --------------------------------------------------------------------------
metrics    = ["accuracy","precision","recall","f1"]
modalities = ["joint","expression","splicing","random"]
colors     = dict(
    joint      = sns.color_palette("tab10")[0],
    expression = sns.color_palette("tab10")[1],
    splicing   = sns.color_palette("tab10")[2],
    random     = "gray",
)
linestyles = dict(
    joint      = "-",
    expression = "-",
    splicing   = "-",
    random     = ":",
)

for name in model_paths:
    df_mod = df_acc[df_acc.model == name]
    fig, axes = plt.subplots(2,2,figsize=(10,8), sharex=True)
    axes = axes.flatten()
    for ax, metric in zip(axes, metrics):
        for mod in modalities:
            sub = (
                df_mod
                [(df_mod.metric==metric)&(df_mod.modality==mod)]
                .groupby("k")["value"]
                .agg(["mean","std"])
                .sort_index()
            )
            if sub.empty: continue
            ax.errorbar(
                sub.index, sub["mean"], yerr=sub["std"],
                label = mod if mod!="random" else "random baseline",
                color = colors[mod],
                linestyle = linestyles[mod],
                marker="o",
                capsize=4,
            )
        ax.set_title(metric.capitalize())
        ax.set_xlabel("k")
        ax.set_ylabel(metric.capitalize())
        ax.set_xticks(sorted(df_mod.k.unique()))

    # shared legend on far right
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        title="Modality / Baseline",
        loc="center right",
        bbox_to_anchor=(0.98, 0.5),
    )

    plt.tight_layout(rect=[0,0,0.85,1.0])
    out = os.path.join(FIG_DIR, f"subcluster_{name}.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("  wrote", out)