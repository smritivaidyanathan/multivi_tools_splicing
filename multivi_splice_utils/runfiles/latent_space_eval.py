#!/usr/bin/env python3
#!/usr/bin/env python3
"""
latent_space_eval.py

Combined script to evaluate latent spaces and local neighborhood consistency for any number of trained models:
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
    csv_files/
  cell_type_classification/
    figures/
    confusion_matrices/
    csv_files/
  local_neighborhood_analysis/
    figures/
    csv_files/

Each section prints start/finish messages and writes outputs accordingly.
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
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

MUDATA_PATH = os.environ.get(
    "MUDATA_PATH",
    "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/052025/aligned__ge_splice_combined_20250513_035938.h5mu"
)
MODEL_PATHS = {
    "dataset_batch_key": "/gpfs/commons/home/svaidyanathan/"
                         "multi_vi_splice_runs/"
                         "MultiVISpliceTraining_20250515_224315_job4683389/models",
}

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

# Parameters
TOP_N_CELLTYPES   = 10
NEIGHBOR_K         = [15, 30]
CLUSTER_NUMBERS    = [3, 5, 7, 10, 20]
TARGET_CELL_TYPES  = []  # list of cell types for which to save confusion matrices, e.g. ["Excitatory Neurons"]
CELL_TYPE_COLUMN   = "broad_cell_type"
UMAP_N_NEIGHBORS   = 15  # number of neighbors for UMAP

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Helper: plot_umap with tab20 formatting
# ----------------------------------------------------------------------------
def plot_umap(adata, rep_name, variable_name, out_dir, num_groups=None):
    """
    Generate and save a UMAP highlighting top groups in variable_name.
    """
    logger.info(f"UMAP: {rep_name}, colored by {variable_name}")
    # top groups
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
        show=False, frameon=True, legend_loc='right margin'
    )
    fname = f"umap_{rep_name}_{variable_name}_top{num_groups or 'all'}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved UMAP: {fname}")

# ----------------------------------------------------------------------------
# 1. Overall latent eval: UMAP + silhouette
# ----------------------------------------------------------------------------
logger.info("[Overall] Computing latents & plots...")
mdata = mu.read_h5mu(MUDATA_PATH)
# ensure psi_mask
sp = mdata['splicing']
cluster = sp.layers['cell_by_cluster_matrix']
if not sparse.isspmatrix_csr(cluster): cluster = sparse.csr_matrix(cluster)
mask = cluster.copy(); mask.data = np.ones_like(mask.data, dtype=np.uint8)
sp.layers['psi_mask'] = mask
# compute latents
latents = {}
for name,path in MODEL_PATHS.items():
    logger.info(f"Loading model {name}")
    model = scvi.model.MULTIVISPLICE.load(path, adata=mdata)
    latents[name] = model.get_latent_representation()
# UMAP & silhouette
labels = mdata['rna'].obs[CELL_TYPE_COLUMN].astype('category').cat.codes.values
sil = []
for name,Z in latents.items():
    ad = sc.AnnData(Z)
    ad.obs[CELL_TYPE_COLUMN] = mdata['rna'].obs[CELL_TYPE_COLUMN].values
    plot_umap(ad, name, CELL_TYPE_COLUMN, os.path.join(OVERALL_DIR,'figures'), TOP_N_CELLTYPES)
    sil.append({'model':name,'silhouette_score':silhouette_score(Z,labels)})
df_sil = pd.DataFrame(sil)
df_sil.to_csv(os.path.join(OVERALL_DIR,'csv_files','silhouette_scores.csv'),index=False)
# barplot
fig,ax=plt.subplots(); ax.bar(df_sil.model, df_sil.silhouette_score)
fig.savefig(os.path.join(OVERALL_DIR,'figures','silhouette_bar.png'),dpi=300);
plt.close(fig)
logger.info("[Overall] Done")

# ----------------------------------------------------------------------------
# 2. Subcluster reproducibility
# ----------------------------------------------------------------------------
logger.info("[Subcluster] Evaluating reproducibility...")
# select cell types with >=5000 cells
counts = mdata['rna'].obs[CELL_TYPE_COLUMN].value_counts()
valid_ct = counts[counts >= 5000].index.tolist()
logger.info(f"Cell types >=5000 cells: {valid_ct}")
records_sub = []
for ct in valid_ct:
    logger.info(f" Cell type {ct} (n={counts[ct]})")
    mask_ct = mdata['rna'].obs[CELL_TYPE_COLUMN] == ct
    cells = mdata['rna'].obs_names[mask_ct].tolist()
    random.shuffle(cells)
    half = len(cells)//2
    idx1 = np.isin(mdata['rna'].obs_names, cells[:half])
    idx2 = np.isin(mdata['rna'].obs_names, cells[half:])
    for name,path in MODEL_PATHS.items():
        model = scvi.model.MULTIVISPLICE.load(path, adata=mdata)
        Zs = {
            'joint': model.get_latent_representation(),
            'expression': model.get_latent_representation(modality='expression'),
            'splicing': model.get_latent_representation(modality='splicing'),
        }
        for k in CLUSTER_NUMBERS:
            km = KMeans(n_clusters=k, random_state=RANDOM_SEED)
            labels1 = km.fit_predict(Zs['joint'][idx1])
            labels2 = km.predict(Zs['joint'][idx2])
            for mod in ['joint','expression','splicing']:
                clf = LogisticRegression(max_iter=200, random_state=RANDOM_SEED)
                clf.fit(Zs[mod][idx2], labels2)
                pred = clf.predict(Zs[mod][idx1])
                for mname, func in [
                    ('accuracy', lambda a,p: (p==a).mean()),
                    ('precision', lambda a,p: precision_score(a,p,average='weighted',zero_division=0)),
                    ('recall', lambda a,p: recall_score(a,p,average='weighted',zero_division=0)),
                    ('f1', lambda a,p: f1_score(a,p,average='weighted',zero_division=0)),
                ]:
                    records_sub.append({
                        'model': name, 'cell_type': ct,
                        'modality': mod, 'k': k,
                        'metric': mname, 'value': func(labels1, pred)
                    })
                if ct in TARGET_CELL_TYPES:
                    cm = confusion_matrix(labels1, pred)
                    disp = ConfusionMatrixDisplay(cm)
                    fig,ax=plt.subplots(figsize=(5,5))
                    disp.plot(ax=ax, cmap='Blues', colorbar=False)
                    out = os.path.join(SUBCLUSTER_DIR, 'figures', 'confusion_matrices', f'conf_{name}_{ct}_{mod}_k{k}.png')
                    fig.savefig(out, dpi=300, bbox_inches='tight'); plt.close(fig)
                    logger.info(f"Saved subcluster CM: {out}")
            # random baseline
            rng = np.random.default_rng(RANDOM_SEED)
            Zrand = rng.standard_normal(Zs['joint'].shape)
            clf = LogisticRegression(max_iter=200, random_state=RANDOM_SEED)
            clf.fit(Zrand[idx2], labels2)
            pred = clf.predict(Zrand[idx1])
            for mname, func in [
                ('accuracy', lambda a,p: (p==a).mean()),
                ('precision', lambda a,p: precision_score(a,p,average='weighted',zero_division=0)),
                ('recall', lambda a,p: recall_score(a,p,average='weighted',zero_division=0)),
                ('f1', lambda a,p: f1_score(a,p,average='weighted',zero_division=0)),
            ]:
                records_sub.append({
                    'model': name, 'cell_type': ct,
                    'modality': 'random', 'k': k,
                    'metric': mname, 'value': func(labels1, pred)
                })
            if ct in TARGET_CELL_TYPES:
                cm = confusion_matrix(labels1, pred)
                disp = ConfusionMatrixDisplay(cm)
                fig,ax=plt.subplots(figsize=(5,5))
                disp.plot(ax=ax, cmap='Blues', colorbar=False)
                out = os.path.join(SUBCLUSTER_DIR, 'figures', 'confusion_matrices', f'conf_{name}_{ct}_random_k{k}.png')
                fig.savefig(out, dpi=300, bbox_inches='tight'); plt.close(fig)
                logger.info(f"Saved subcluster CM: {out}")
# save subcluster metrics
_df_sub = pd.DataFrame.from_records(records_sub)
_df_sub.to_csv(os.path.join(SUBCLUSTER_DIR, 'csv_files', 'subcluster_metrics.csv'), index=False)
logger.info("[Subcluster] Done")

# ----------------------------------------------------------------------------
# 3. Cell-Type Classification Performance
# ----------------------------------------------------------------------------
logger.info("[Classification] Starting multiclass LR evaluation...")
y = mdata['rna'].obs[CELL_TYPE_COLUMN].astype('category')
labels = y.cat.codes.values
records = []
for name,lat in latents.items():
    for modality in ['joint','expression','splicing']:
        # load Z for modality
        model = scvi.model.MULTIVISPLICE.load(MUDATA_PATH, adata=mdata)
        Z = (model.get_latent_representation() if modality=='joint'
             else model.get_latent_representation(modality=modality))
        # split
        from sklearn.model_selection import train_test_split
        Xtr,Xte,ytr,yte = train_test_split(Z,labels,
            stratify=labels,test_size=0.2,random_state=RANDOM_SEED)
        clf = LogisticRegression(max_iter=200,random_state=RANDOM_SEED).fit(Xtr,ytr)
        pred=clf.predict(Xte)
        # per-type + overall
        for ct_idx,ct in enumerate(y.cat.categories):
            mask_ct = (yte==ct_idx)
            if mask_ct.sum()>0:
                records.append({
                    'model':name,'modality':modality,'cell_type':ct,
                    'accuracy':accuracy_score(yte[mask_ct],pred[mask_ct]),
                    'precision':precision_score(yte[mask_ct],pred[mask_ct],average='macro',zero_division=0),
                    'recall':recall_score(yte[mask_ct],pred[mask_ct],average='macro',zero_division=0),
                    'f1':f1_score(yte[mask_ct],pred[mask_ct],average='macro',zero_division=0)
                })
        # overall
        records.append({
            'model':name,'modality':modality,'cell_type':'ALL',
            'accuracy':accuracy_score(yte,pred),
            'precision':precision_score(yte,pred,average='macro',zero_division=0),
            'recall':recall_score(yte,pred,average='macro',zero_division=0),
            'f1':f1_score(yte,pred,average='macro',zero_division=0)
        })
        # confusion
        cm=confusion_matrix(yte,pred,labels=range(TOP_N_CELLTYPES))
        disp=ConfusionMatrixDisplay(cm,display_labels=y.cat.categories[:TOP_N_CELLTYPES])
        fig,ax=plt.subplots(figsize=(6,6))
        disp.plot(ax=ax,cmap='Blues',colorbar=False)
        fn=os.path.join(CLASSIF_DIR,'confusion_matrices',f'cm_{name}_{modality}.png')
        fig.savefig(fn,dpi=300,bbox_inches='tight'); plt.close(fig)
        logger.info(f"Saved CM: {fn}")
# save CSV
_df_cls=pd.DataFrame.from_records(records)
_df_cls.to_csv(os.path.join(CLASSIF_DIR,'csv_files','cell_type_classification.csv'),index=False)
# barplots
for m in ['accuracy','precision','recall','f1']:
    fig,ax=plt.subplots(figsize=(8,6))
    sns.barplot(data=_df_cls[_df_cls.cell_type!='ALL'],x='cell_type',y=m,hue='modality',ax=ax)
    plt.xticks(rotation=45,ha='right'); fig.tight_layout()
    fn=os.path.join(CLASSIF_DIR,'figures',f'{m}_bar.png')
    fig.savefig(fn,dpi=300); plt.close(fig)
    logger.info(f"Saved bar: {fn}")
logger.info("[Classification] Complete")

# ----------------------------------------------------------------------------
# 4. Local Neighborhood Consistency
# ----------------------------------------------------------------------------
logger.info("[Neighborhood] Computing overlap scores...")
# load Z spaces
Zj=latents[list(latents)[0]]
Zge=scvi.model.MULTIVISPLICE.load(MUDATA_PATH, adata=mdata).get_latent_representation(modality='expression')
Zas=scvi.model.MULTIVISPLICE.load(MUDATA_PATH, adata=mdata).get_latent_representation(modality='splicing')

records=[]
for k in NEIGHBOR_K:
    nbrs_j=NearestNeighbors(n_neighbors=k+1).fit(Zj)
    idx_j=nbrs_j.kneighbors(return_distance=False)[:,1:]
    nbrs_ge=NearestNeighbors(n_neighbors=k+1).fit(Zge)
    idx_ge=nbrs_ge.kneighbors(return_distance=False)[:,1:]
    nbrs_as=NearestNeighbors(n_neighbors=k+1).fit(Zas)
    idx_as=nbrs_as.kneighbors(return_distance=False)[:,1:]
    for i in range(Zj.shape[0]):
        setj, setge, setas = set(idx_j[i]), set(idx_ge[i]), set(idx_as[i])
        records.append({'cell':i,'k':k,
                        'S_GE':len(setj&setge)/k,
                        'S_AS':len(setj&setas)/k})
# save
_df_nb=pd.DataFrame.from_records(records)
_df_nb.to_csv(os.path.join(NEIGHBOR_DIR,'csv_files','neighborhood_overlap.csv'),index=False)
# plots
for sc_score in ['S_GE','S_AS']:
    fig,ax=plt.subplots(); sns.histplot(_df_nb[sc_score],kde=True,ax=ax)
    out=os.path.join(NEIGHBOR_DIR,'figures',f'{sc_score}_hist.png')
    fig.savefig(out,dpi=300); plt.close(fig)
    logger.info(f"Saved hist: {out}")
    # UMAP colored
    ad=sc.AnnData(Zj); ad.obs[sc_score]=_df_nb[sc_score].values
    plot_umap(ad, f'Z_joint_{sc_score}', sc_score, os.path.join(NEIGHBOR_DIR,'figures'))
# scatter
fig,ax=plt.subplots(figsize=(6,6))
sns.scatterplot(x='S_GE',y='S_AS',data=_df_nb,alpha=0.5)
out=os.path.join(NEIGHBOR_DIR,'figures','scatter_SGE_SAS.png')
fig.savefig(out,dpi=300); plt.close(fig)
logger.info("[Neighborhood] Complete")
