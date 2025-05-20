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
    "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/052025/aligned__ge_splice_combined_20250513_035938.h5mu"
)
MODEL_NAME = "SpliceVI_Mockup"
MODEL_PATH = "/gpfs/commons/home/svaidyanathan/multi_vi_splice_runs/MultiVISpliceTraining_20250515_224315_job4683389/models"

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
TOP_N_CELLTYPES   = 10
NEIGHBOR_K         = [30]
CLUSTER_NUMBERS    = [3, 5]
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
    if num_groups:
        top = adata.obs[variable_name].value_counts().head(num_groups).index.tolist()
    else:
        top = adata.obs[variable_name].unique().tolist()
    adata.obs['group_highlighted'] = np.where(
        adata.obs[variable_name].isin(top), adata.obs[variable_name], 'Other'
    )
    cmap = cm.get_cmap('tab20', len(top))
    colors = {grp: cmap(i) for i, grp in enumerate(top)}
    colors['Other'] = (0.9,0.9,0.9,1.0)
    sc.pp.neighbors(adata, use_rep='X', n_neighbors=UMAP_N_NEIGHBORS)
    sc.tl.umap(adata, min_dist=0.1)
    fig = sc.pl.umap(
        adata, color='group_highlighted', palette=colors,
        show=False, frameon=True, legend_loc='right margin', return_fig=True
    )
    fname = f"umap_{MODEL_NAME}_{rep_name}_{variable_name}_top{num_groups or 'all'}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_subcluster_umap(latent, true_labels, pred_labels, out_prefix, fig_dir,
                         n_neighbors=UMAP_N_NEIGHBORS, min_dist=0.1):
    ad = sc.AnnData(latent)
    ad.obsm['X_latent'] = latent
    ad.obs['true'] = true_labels.astype(str)
    ad.obs['pred'] = pred_labels.astype(str)
    sc.pp.neighbors(ad, use_rep='X_latent', n_neighbors=n_neighbors)
    sc.tl.umap(ad, min_dist=min_dist)
    for label_type in ['true','pred']:
        fig, ax = plt.subplots(figsize=(4,4))
        sc.pl.umap(ad, color=label_type, ax=ax, show=False, legend_loc=None)
        ax.set_title(f"{MODEL_NAME} {out_prefix} — {label_type}")
        fname = f"{MODEL_NAME}_{out_prefix}_{label_type}_umap.png"
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
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED)
        lab1 = km.fit_predict(Z_joint[idx1])
        lab2 = km.predict(Z_joint[idx2])
        for rep, Z in groups.items():
            clf = LogisticRegression(max_iter=200, random_state=RANDOM_SEED)
            clf.fit(Z[idx2], lab2)
            pred = clf.predict(Z[idx1])
            for mname, func in [('accuracy',lambda a,p:(p==a).mean()),
                                ('precision',lambda a,p:precision_score(a,p,average='weighted',zero_division=0)),
                                ('recall',lambda a,p:recall_score(a,p,average='weighted',zero_division=0)),
                                ('f1',lambda a,p:f1_score(a,p,average='weighted',zero_division=0))]:
                records_sub.append({'model':MODEL_NAME,'cell_type':ct,'rep':rep,'k':k,'metric':mname,'value':func(lab1,pred)})
            if ct in TARGET_CELL_TYPES:
                cm = confusion_matrix(lab1, pred)
                fig,ax=plt.subplots(figsize=(4,4))
                ConfusionMatrixDisplay(cm).plot(ax=ax, cmap='Blues', colorbar=False)
                um_dir = os.path.join(SUBCLUSTER_DIR,'figures','umaps')
                cm_dir = os.path.join(SUBCLUSTER_DIR,'figures','confusion_matrices')
                fname = f"{MODEL_NAME}_{ct.replace(' ','_')}_{rep}_k{k}_cm.png"
                fig.savefig(os.path.join(cm_dir,fname), dpi=300, bbox_inches='tight'); plt.close(fig)
                plot_subcluster_umap(Z[idx1], lab1, pred,
                                     f"{ct.replace(' ','_')}_{rep}_k{k}",
                                     um_dir)

_df_sub = pd.DataFrame.from_records(records_sub)
_df_sub.to_csv(os.path.join(SUBCLUSTER_DIR,'csv_files','subcluster_metrics.csv'), index=False)
logger.info("[Subcluster] Done")

# ----------------------------------------------------------------------------
# 3. Cell-Type Classification Performance
# ----------------------------------------------------------------------------
# This section trains multiclass logistic regression on each latent space
# to predict broad cell type labels across all cells, reporting per-type
# and overall accuracy, precision, recall, and F1. Confusion matrices are also saved.

logger.info("[Classification] Starting multiclass classification...")
y = mdata['rna'].obs[CELL_TYPE_COLUMN].astype('category')
labels_all = y.cat.codes.values
records_cls=[]
for rep, Z in groups.items():
    Xtr, Xte, ytr, yte = train_test_split(Z, labels_all,
                                         stratify=labels_all, test_size=0.2,
                                         random_state=RANDOM_SEED)
    clf = LogisticRegression(max_iter=200, random_state=RANDOM_SEED).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    for idx, ct in enumerate(y.cat.categories):
        mask_ct = (yte==idx)
        if mask_ct.sum()>0:
            records_cls.append({'model':MODEL_NAME,'rep':rep,'cell_type':ct,
                                'accuracy':accuracy_score(yte[mask_ct],pred[mask_ct]),
                                'precision':precision_score(yte[mask_ct],pred[mask_ct],average='macro',zero_division=0),
                                'recall':recall_score(yte[mask_ct],pred[mask_ct],average='macro',zero_division=0),
                                'f1':f1_score(yte[mask_ct],pred[mask_ct],average='macro',zero_division=0)})
    records_cls.append({'model':MODEL_NAME,'rep':rep,'cell_type':'ALL',
                        'accuracy':accuracy_score(yte,pred),
                        'precision':precision_score(yte,pred,average='macro',zero_division=0),
                        'recall':recall_score(yte,pred,average='macro',zero_division=0),
                        'f1':f1_score(yte,pred,average='macro',zero_division=0)})
    cm = confusion_matrix(yte, pred, labels=range(TOP_N_CELLTYPES))
    fig,ax=plt.subplots(figsize=(5,5))
    ConfusionMatrixDisplay(cm, display_labels=y.cat.categories[:TOP_N_CELLTYPES]).plot(ax=ax, cmap='Blues', colorbar=False)
    fname = f"cm_{MODEL_NAME}_{rep}.png"
    fig.savefig(os.path.join(CLASSIF_DIR,'confusion_matrices',fname), dpi=300, bbox_inches='tight'); plt.close(fig)
_df_cls=pd.DataFrame.from_records(records_cls)
_df_cls.to_csv(os.path.join(CLASSIF_DIR,'csv_files','cell_type_classification.csv'), index=False)
# barplots
for m in ['accuracy','precision','recall','f1']:
    fig,ax=plt.subplots(figsize=(6,4))
    sns.barplot(data=_df_cls[_df_cls.cell_type!='ALL'], x='cell_type', y=m, hue='rep', ax=ax)
    plt.xticks(rotation=45, ha='right'); fig.tight_layout()
    fig.savefig(os.path.join(CLASSIF_DIR,'figures',f'{m}_bar.png'), dpi=300); plt.close(fig)
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
# plots
for sc_score in ['S_GE','S_AS']:
    fig,ax=plt.subplots(); sns.histplot(_df_nb[sc_score], kde=True, ax=ax)
    fig.savefig(os.path.join(NEIGHBOR_DIR,'figures',f'{sc_score}_hist.png'), dpi=300); plt.close(fig)
    ad = sc.AnnData(Z_joint)
    ad.obs[sc_score] = _df_nb[sc_score].values
    plot_umap(ad, f'Z_joint_{sc_score}', sc_score, os.path.join(NEIGHBOR_DIR,'figures'), None)
# scatter
fig,ax=plt.subplots(figsize=(5,5))
sns.scatterplot(x='S_GE',y='S_AS',data=_df_nb,alpha=0.5)
fname = os.path.join(NEIGHBOR_DIR,'figures','scatter_SGE_SAS.png')
fig.savefig(fname, dpi=300); plt.close(fig)
logger.info("[Neighborhood] Complete")
