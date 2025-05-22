#!/usr/bin/env python3
"""
latent_space_eval.py

Combined script to evaluate a single trained MULTIVISPLICE model's latent spaces and local neighborhood consistency:
 1. Overall UMAP and silhouette
 2. Subcluster reproducibility
 3. Multiclass logistic regression cell-type prediction
 4. Local neighborhood consistency

Directory structure under LATENT_EVAL_OUTDIR:
  overall/
    figures/
    csv_files/
  subcluster_eval/
    figures/line_plots/
    figures/confusion_matrices/
    figures/umaps/
    csv_files/
  cell_type_classification/
    figures/
    confusion_matrices/
    csv_files/
  local_neighborhood_analysis/
    figures/
    csv_files/

Each section prints start/finish messages and writes outputs accordingly for one model.
"""

import os
import random
import logging
import gc
import numpy as np
import pandas as pd
import mudata as mu
import scvi
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.metrics import (
    silhouette_score, precision_score, recall_score,
    f1_score, confusion_matrix, accuracy_score
)
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

# ----------------------------------------------------------------------------
# 0. Configuration (editable at top)
# ----------------------------------------------------------------------------
# This section sets random seeds, paths, output directories, and core parameters

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Path to the MuData and the single model
MUDATA_PATH = os.environ.get(
    "MUDATA_PATH",
    "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/052025/SUBSETTOP5CELLSTYPES_aligned__ge_splice_combined_20250513_035938.h5mu"
)
MODEL_NAME = "SpliceVI_SUBSET_88EPOCHS_LINEARDECODER_PARTIALENCODER"
MODEL_PATH = "/gpfs/commons/home/svaidyanathan/multi_vi_splice_runs/MultiVISpliceTraining_20250522_132915_job4818968/models"

LATENT_EVAL_OUTDIR = os.environ.get("LATENT_EVAL_OUTDIR", "./latent_eval_output")
OVERALL_DIR    = os.path.join(LATENT_EVAL_OUTDIR, "overall")
SUBCLUSTER_DIR = os.path.join(LATENT_EVAL_OUTDIR, "subcluster_eval")
CLASSIF_DIR    = os.path.join(LATENT_EVAL_OUTDIR, "cell_type_classification")
NEIGHBOR_DIR   = os.path.join(LATENT_EVAL_OUTDIR, "local_neighborhood_analysis")
# make directories
for base in [OVERALL_DIR, SUBCLUSTER_DIR, CLASSIF_DIR, NEIGHBOR_DIR]:
    os.makedirs(os.path.join(base, "figures"), exist_ok=True)
    os.makedirs(os.path.join(base, "csv_files"), exist_ok=True)
os.makedirs(os.path.join(SUBCLUSTER_DIR, "figures", "line_plots"), exist_ok=True)
os.makedirs(os.path.join(SUBCLUSTER_DIR, "figures", "confusion_matrices"), exist_ok=True)
os.makedirs(os.path.join(SUBCLUSTER_DIR, "figures", "umaps"), exist_ok=True)

# Parameters
TOP_N_CELLTYPES   = 5
NEIGHBOR_K         = [30]
CLUSTER_NUMBERS    = [3, 5, 7, 10, 15, 20]
TARGET_CELL_TYPES  = ["Excitatory Neurons", "MICROGLIA"]
CELL_TYPE_COLUMN   = "broad_cell_type"
UMAP_N_NEIGHBORS   = 15  # neighbors for UMAP

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
# Helper functions for plotting UMAPs and subcluster reproducibility results

def plot_umap(adata, rep_name, variable_name, out_dir, num_groups=None):
    logger.info(f"UMAP: {rep_name}, colored by {variable_name}")
    # make neighbors + UMAP once
    sc.pp.neighbors(adata, use_rep='X', n_neighbors=UMAP_N_NEIGHBORS)
    sc.tl.umap(adata, min_dist=0.1)

    if num_groups is not None:
        # categorical mode
        top = (
            adata.obs[variable_name]
            .value_counts()
            .head(num_groups)
            .index
            .tolist()
            if num_groups
            else adata.obs[variable_name].unique().tolist()
        )
        # highlight only the top groups, everything else as "Other"
        adata.obs['group_highlighted'] = np.where(
            adata.obs[variable_name].isin(top),
            adata.obs[variable_name].astype(str),
            'Other',
        )
        # build a discrete palette
        cmap_mod = cm.get_cmap('tab20', len(top))
        colors = {grp: cmap_mod(i) for i, grp in enumerate(top)}
        colors['Other'] = (0.9, 0.9, 0.9, 1.0)
        # plot
        plt.figure(figsize=(8, 5))
        fig = sc.pl.umap(
            adata,
            color='group_highlighted',
            palette=colors,
            show=False,
            frameon=True,
            legend_fontsize=10,
            legend_loc='right margin',
            return_fig=True,
        )
        plt.title(f'UMAP by {variable_name} (Top {num_groups} Highlighted)')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fname = f"umap_{MODEL_NAME}_{rep_name}_{variable_name}_top{num_groups}.png"

    else:
        # continuous mode
        plt.figure(figsize=(8, 5))
        fig = sc.pl.umap(
            adata,
            color=variable_name,
            cmap='viridis',
            vmin=0,
            vmax=1,
            show=False,
            frameon=True,
            legend_loc='right margin',
            legend_fontsize=10,
            return_fig=True,
        )
        plt.title(f'UMAP by {variable_name}')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fname = f"umap_{MODEL_NAME}_{rep_name}_{variable_name}_continuous.png"

    fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_subcluster_umap(latent, true_labels, pred_labels_dict, out_prefix, fig_dir,
                         n_neighbors=UMAP_N_NEIGHBORS, min_dist=0.1):
    # build AnnData once
    ad = sc.AnnData(latent)
    ad.obsm['X_latent'] = latent
    ad.obs['true'] = true_labels.astype(str)

    # compute UMAP embedding
    sc.pp.neighbors(ad, use_rep='X_latent', n_neighbors=n_neighbors)
    sc.tl.umap(ad, min_dist=min_dist)

    # plot the “true” labels
    fig, ax = plt.subplots(figsize=(4,4))
    sc.pl.umap(ad, color='true', ax=ax, show=False, frameon=True, legend_loc=None)
    ax.set_title(f"{MODEL_NAME} {out_prefix} — true labels")
    fname = f"{MODEL_NAME}_{out_prefix}_true_umap.png"
    fig.savefig(os.path.join(fig_dir, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # now add one obs‐column + plot per modality
    for modality, labels in pred_labels_dict.items():
        ad.obs[modality] = labels.astype(str)
        fig, ax = plt.subplots(figsize=(4,4))
        sc.pl.umap(ad, color=modality, ax=ax, show=False, frameon=True, legend_loc=None)
        ax.set_title(f"{MODEL_NAME} {out_prefix} — {modality} predicted labels")
        fname = f"{MODEL_NAME}_{out_prefix}_{modality}_umap.png"
        fig.savefig(os.path.join(fig_dir, fname), dpi=300, bbox_inches='tight')
        plt.close(fig)

# ----------------------------------------------------------------------------
# 1. Overall latent eval: UMAP + silhouette
# ----------------------------------------------------------------------------
# This section computes the joint, expression-specific, and splicing-specific latent
# embeddings for all cells, generates UMAP visualizations colored by cell type, and
# calculates silhouette scores to assess cluster separation in each latent space.

logger.info("[Overall] Computing latents & plots...")
mdata = mu.read_h5mu(MUDATA_PATH)
# ensure psi_mask
sp = mdata['splicing']
cluster = sp.layers['cell_by_cluster_matrix']
if not sparse.isspmatrix_csr(cluster): cluster = sparse.csr_matrix(cluster)
mask = cluster.copy(); mask.data = np.ones_like(cluster.data, dtype=np.uint8)
sp.layers['psi_mask'] = mask
# load model once
model = scvi.model.MULTIVISPLICE.load(MODEL_PATH, adata=mdata)
# compute representations
Z_joint = model.get_latent_representation()
Z_ge    = model.get_latent_representation(modality='expression')
Z_as    = model.get_latent_representation(modality='splicing')
# UMAP & silhouette
groups = {'joint': Z_joint, 'expression': Z_ge, 'splicing': Z_as}
labels = mdata['rna'].obs[CELL_TYPE_COLUMN].astype('category').cat.codes.values
sil_records = []
for rep, Z in groups.items():
    ad = sc.AnnData(Z)
    ad.obs[CELL_TYPE_COLUMN] = mdata['rna'].obs[CELL_TYPE_COLUMN].values
    plot_umap(ad, rep, CELL_TYPE_COLUMN, os.path.join(OVERALL_DIR,'figures'), TOP_N_CELLTYPES)
    sil_records.append({'model':MODEL_NAME,'rep':rep,'silhouette_score':silhouette_score(Z,labels)})
df_sil = pd.DataFrame(sil_records)
df_sil.to_csv(os.path.join(OVERALL_DIR,'csv_files','silhouette_scores.csv'), index=False)
# barplot
fig,ax=plt.subplots(figsize=(4,3))
sns.barplot(data=df_sil, x='rep', y='silhouette_score', palette='tab10', ax=ax)
ax.set_title(f"{MODEL_NAME} silhouette scores")
fig.savefig(os.path.join(OVERALL_DIR,'figures','silhouette_bar.png'), dpi=300)
plt.close(fig)
logger.info("[Overall] Done")

# ----------------------------------------------------------------------------
# 2. Subcluster reproducibility
# ----------------------------------------------------------------------------
# This section evaluates how well latent spaces reproduce clusters within each large cell type.
# For each cell type having >=5000 cells, we split cells into two halves,
# cluster using k-means on the joint embedding, and train logistic regression
# classifiers on each latent space to predict cluster labels across halves.

logger.info("[Subcluster] Evaluating reproducibility...")
counts = mdata['rna'].obs[CELL_TYPE_COLUMN].value_counts()
valid_ct = counts[counts>=5000].index.tolist()
records_sub=[]
for ct in valid_ct:
    mask_ct = mdata['rna'].obs[CELL_TYPE_COLUMN]==ct
    cells = mdata['rna'].obs_names[mask_ct].tolist()
    random.shuffle(cells)
    half = len(cells)//2
    idx1 = np.isin(mdata.obs_names, cells[:half])
    idx2 = np.isin(mdata.obs_names, cells[half:])
    for k in CLUSTER_NUMBERS:
        um_dir = os.path.join(SUBCLUSTER_DIR,'figures','umaps')
        cm_dir = os.path.join(SUBCLUSTER_DIR,'figures','confusion_matrices')
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED)
        lab1 = km.fit_predict(Z_joint[idx1])
        lab2 = km.predict(Z_joint[idx2])
        # dictionary to hold every modality’s predictions
        preds = {}
        for rep, Z in groups.items():
            clf = LogisticRegression(max_iter=200, random_state=RANDOM_SEED)
            clf.fit(Z[idx2], lab2)
            pred = clf.predict(Z[idx1])
            preds[rep] = pred
            for mname, func in [('accuracy',lambda a,p:(p==a).mean()),
                                ('precision',lambda a,p:precision_score(a,p,average='weighted',zero_division=0)),
                                ('recall',lambda a,p:recall_score(a,p,average='weighted',zero_division=0)),
                                ('f1',lambda a,p:f1_score(a,p,average='weighted',zero_division=0))]:
                records_sub.append({'model':MODEL_NAME,'cell_type':ct,'rep':rep,'k':k,'metric':mname,'value':func(lab1,pred)})
            if ct in TARGET_CELL_TYPES:
                conf_mat = confusion_matrix(lab1, pred)
                fig,ax=plt.subplots(figsize=(4,4))
                ConfusionMatrixDisplay(conf_mat).plot(ax=ax, cmap='Blues', colorbar=False)
                fname = f"{MODEL_NAME}_{ct.replace(' ','_')}_{rep}_k{k}_conf_mat.png"
                fig.savefig(os.path.join(cm_dir,fname), dpi=300, bbox_inches='tight'); plt.close(fig)
        # ─── random‐baseline modality ───────────────────────────────
        # generate a random latent space with the same shape as Z_joint
        rng   = np.random.default_rng(RANDOM_SEED)
        Zrand = rng.standard_normal(Z_joint.shape)

        # fit & predict exactly as you do for each real modality
        clf  = LogisticRegression(max_iter=200, random_state=RANDOM_SEED)
        clf.fit(Zrand[idx2], lab2)
        pred = clf.predict(Zrand[idx1])
        preds['random'] = pred

        # record all four metrics under rep="random"
        for mname, func in [
            ('accuracy',  lambda a,p: (p==a).mean()),
            ('precision', lambda a,p: precision_score(a,p,average='weighted',zero_division=0)),
            ('recall',    lambda a,p: recall_score(a,p,average='weighted',zero_division=0)),
            ('f1',        lambda a,p: f1_score(a,p,average='weighted',zero_division=0)),
        ]:
            records_sub.append({
                'model':     MODEL_NAME,
                'cell_type': ct,
                'rep':       'random',
                'k':         k,
                'metric':    mname,
                'value':     func(lab1, pred),
            })

        # if you also want confusion matrices for the random baseline:
        if ct in TARGET_CELL_TYPES:
            conf_mat   = confusion_matrix(lab1, pred)
            fig, ax = plt.subplots(figsize=(4,4))
            ConfusionMatrixDisplay(conf_mat).plot(ax=ax, cmap='Blues', colorbar=False)
            fname = f"{MODEL_NAME}_{ct.replace(' ','_')}_random_k{k}_cm.png"
            fig.savefig(os.path.join(cm_dir, fname), dpi=300, bbox_inches='tight')
            plt.close(fig)
            # —— now plot all UMAPs at once for Z_joint, if this cell type is one you care about:
            plot_subcluster_umap(
                Z_joint[idx1],
                lab1,
                preds,
                f"{ct.replace(' ', '_')}_k{k}",
                um_dir
            )
            logger.info(f"Saved UMAPs to {um_dir} for k={k}, ct={ct}")
        # ─────────────────────────────────────────────────────────────

_df_sub = pd.DataFrame.from_records(records_sub)
_df_sub.to_csv(os.path.join(SUBCLUSTER_DIR,'csv_files','subcluster_metrics.csv'), index=False)

# ───────────────────────────────────────────────────────────────────────────────
# Plot per‐model line charts including random baseline
# ───────────────────────────────────────────────────────────────────────────────
import seaborn as sns

metrics    = ["accuracy","precision","recall","f1"]
# rep holds your modalities plus the 'random' baseline
modalities = list(groups.keys()) + ["random"]
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

# since you only have one model:
df_plot = _df_sub[_df_sub.model == MODEL_NAME]

fig, axes = plt.subplots(2,2,figsize=(10,8), sharex=True)
axes = axes.flatten()
for ax, metric in zip(axes, metrics):
    for mod in modalities:
        sub = (
            df_plot
            [(df_plot.metric==metric)&(df_plot.rep==mod)]
            .groupby("k")["value"]
            .agg(["mean","std"])
            .sort_index()
        )
        if sub.empty: 
            continue
        ax.errorbar(
            sub.index, sub["mean"], yerr=sub["std"],
            label = mod if mod!="random" else "random baseline",
            color = colors.get(mod),
            linestyle = linestyles.get(mod),
            marker="o",
            capsize=4,
        )
    ax.set_title(metric.capitalize())
    ax.set_xlabel("k")
    ax.set_ylabel(metric.capitalize())
    ax.set_xticks(sorted(df_plot.k.unique()))

# shared legend on the right
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    title="Modality / Baseline",
    loc="center right",
    bbox_to_anchor=(0.98, 0.5),
)

plt.tight_layout(rect=[0,0,0.85,1.0])

LINEPLOT_FIG_DIR = os.path.join(SUBCLUSTER_DIR, "figures", "line_plots")

out = os.path.join(LINEPLOT_FIG_DIR, f"subcluster_{MODEL_NAME}.png")
fig.savefig(out, dpi=300)
plt.close(fig)
print("  wrote", out)
# ───────────────────────────────────────────────────────────────────────────────

logger.info("[Subcluster] Done")

# ----------------------------------------------------------------------------
# 3. Cell-Type Classification Performance
# ----------------------------------------------------------------------------
# This section trains multiclass logistic regression on each latent space
# to predict broad cell type labels across all cells, reporting per-type
# and overall accuracy, precision, recall, and F1. Confusion matrices are also saved.

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt

# set plotting defaults
sns.set_style("whitegrid")
sns.set_context("talk")

logger.info("[Classification] Starting multiclass classification...")
y = mdata['rna'].obs[CELL_TYPE_COLUMN].astype('category')
labels_all = y.cat.codes.values
records_cls = []

for rep, Z in groups.items():
    # stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        Z,
        labels_all,
        test_size=0.2,
        stratify=labels_all,
        random_state=RANDOM_SEED,
    )

    # unified LogisticRegression protocol
    clf = LogisticRegression(
        max_iter=500,
        multi_class='ovr',
        solver='lbfgs',
        C=1.0,
        penalty='l2',
        random_state=RANDOM_SEED
    ).fit(X_train, y_train)

    # predict & record metrics
    y_pred = clf.predict(X_test)
    for idx, ct in enumerate(y.cat.categories):
        mask_ct = (y_test == idx)
        if mask_ct.sum() > 0:
            records_cls.append({
                'model':     MODEL_NAME,
                'rep':       rep,
                'cell_type': ct,
                'accuracy':  accuracy_score(y_test[mask_ct],  y_pred[mask_ct]),
                'precision': precision_score(y_test[mask_ct], y_pred[mask_ct],
                                             average='macro', zero_division=0),
                'recall':    recall_score(y_test[mask_ct],    y_pred[mask_ct],
                                          average='macro', zero_division=0),
                'f1':        f1_score(y_test[mask_ct],        y_pred[mask_ct],
                                      average='macro', zero_division=0),
            })

    # overall metrics
    records_cls.append({
        'model':     MODEL_NAME,
        'rep':       rep,
        'cell_type': 'ALL',
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall':    recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1':        f1_score(y_test, y_pred, average='macro', zero_division=0),
    })

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=range(TOP_N_CELLTYPES))
    disp = ConfusionMatrixDisplay(cm,
        display_labels=y.cat.categories[:TOP_N_CELLTYPES]
    )
    fig, ax = plt.subplots(figsize=(8,8))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
    fig.tight_layout()
    fname = f"conf_mat_{MODEL_NAME}_{rep}.png"
    fig.savefig(os.path.join(CLASSIF_DIR, 'figures', fname),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

_df_cls = pd.DataFrame.from_records(records_cls)
_df_cls.to_csv(os.path.join(CLASSIF_DIR, 'csv_files',
                            'cell_type_classification.csv'),
               index=False)

# barplots
for metric in ['accuracy','precision','recall','f1']:
    fig, ax = plt.subplots(figsize=(14,6))
    sns.barplot(
        data=_df_cls[_df_cls.cell_type!='ALL'],
        x='cell_type', y=metric, hue='rep', ax=ax
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    fig.tight_layout()
    fname = f'{metric}_bar.png'
    fig.savefig(os.path.join(CLASSIF_DIR, 'figures', fname),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

logger.info("[Classification] Complete")


# ----------------------------------------------------------------------------
# 4. Local Neighborhood Consistency
# ----------------------------------------------------------------------------
# This section measures the overlap between each cell's k-nearest neighbors
# in the joint embedding versus the expression- and splicing-specific embeddings.
# Reports S_GE and S_AS (fractional overlaps) and saves histograms, UMAPs, and a scatter plot.

logger.info("[Neighborhood] Computing overlap scores...")
records_nb=[]
for k in NEIGHBOR_K:
    nbr_j = NearestNeighbors(n_neighbors=k+1).fit(Z_joint)
    idx_j = nbr_j.kneighbors(return_distance=False)[:,1:]
    nbr_ge= NearestNeighbors(n_neighbors=k+1).fit(Z_ge)
    idx_ge= nbr_ge.kneighbors(return_distance=False)[:,1:]
    nbr_as= NearestNeighbors(n_neighbors=k+1).fit(Z_as)
    idx_as= nbr_as.kneighbors(return_distance=False)[:,1:]
    for i in range(Z_joint.shape[0]):
        setj, setge, setas = set(idx_j[i]), set(idx_ge[i]), set(idx_as[i])
        records_nb.append({'cell':i,'k':k,
                           'S_GE':len(setj&setge)/k,
                           'S_AS':len(setj&setas)/k})
_df_nb=pd.DataFrame.from_records(records_nb)
_df_nb.to_csv(os.path.join(NEIGHBOR_DIR,'csv_files','neighborhood_overlap.csv'), index=False)

from scipy.stats import spearmanr
rho, pval = spearmanr(_df_nb['S_GE'], _df_nb['S_AS'])
logger.info(f"[Neighborhood] Spearman ρ (S_GE vs S_AS): {rho:.3f}, p-value: {pval:.3g}")

# plots
for sc_score in ['S_GE','S_AS']:
    fig,ax=plt.subplots(); sns.histplot(_df_nb[sc_score], kde=True, ax=ax)
    fig.savefig(os.path.join(NEIGHBOR_DIR,'figures',f'{sc_score}_hist.png'), dpi=300); plt.close(fig)
    ad = sc.AnnData(Z_joint)
    ad.obs[sc_score] = _df_nb[sc_score].values
    plot_umap(ad, f'Z_joint_{sc_score}', sc_score, os.path.join(NEIGHBOR_DIR,'figures'), None)
# scatter
fig,ax=plt.subplots(figsize=(5,5))
sns.scatterplot(x='S_GE',y='S_AS',data=_df_nb,alpha=0.1)
fname = os.path.join(NEIGHBOR_DIR,'figures','scatter_SGE_SAS.png')
fig.savefig(fname, dpi=300); plt.close(fig)
logger.info("[Neighborhood] Complete")
