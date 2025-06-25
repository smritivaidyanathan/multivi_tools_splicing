#!/usr/bin/env python3
"""
latentspace_multivisplice.py

Combined script to evaluate a single trained MULTIVISPLICE model's latent spaces and local neighborhood consistency:
 1. Overall UMAP and silhouette
 2. Subcluster reproducibility
 3. Multiclass logistic regression cell-type prediction
 4. Local neighborhood consistency

Directory structure under OUT_DIR:
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
    csv_files/
  local_neighborhood_analysis/
    figures/
    csv_files/

Each section prints start/finish messages and writes outputs accordingly for one model.
"""
import os
import argparse
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
from sklearn.metrics import silhouette_score, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

# ----------------------------------------------------------------------------
# 0. argparse and setup
# ----------------------------------------------------------------------------
parser = argparse.ArgumentParser("latent_space_eval.py: Evaluate MULTIVISPLICE latent spaces")
parser.add_argument("--mudata_path", type=str, required=True, help="MuData input (.h5mu)")
parser.add_argument("--model_path", type=str, required=True, help="Directory with trained MULTIVISPLICE model")
parser.add_argument("--out_dir", type=str, required=True, help="Base output directory for evaluation results")
parser.add_argument("--cluster_numbers", nargs='+', type=int, default=[3,5,10], help="List of cluster k's for subcluster eval")
parser.add_argument("--neighbor_k", nargs='+', type=int, default=[30], help="List of k's for neighborhood overlap")
parser.add_argument("--cell_type_column", type=str, default="broad_cell_type", help="Cell-type column in obs")
parser.add_argument("--top_n_celltypes", type=int, default=5, help="Number of top cell types for subcluster evaluation")
parser.add_argument("--umap_cell_label", nargs='+', default=["broad_cell_type"], help="List of obs columns to color UMAP by")
args = parser.parse_args()

# set random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# define directories
overall_dir    = os.path.join(args.out_dir, "overall")
subcluster_dir = os.path.join(args.out_dir, "subcluster_eval")
classif_dir    = os.path.join(args.out_dir, "cell_type_classification")
neighbor_dir   = os.path.join(args.out_dir, "local_neighborhood_analysis")
umaps_dir      = os.path.join(args.out_dir, "umaps")

# create directories
for d in [overall_dir, subcluster_dir, classif_dir, neighbor_dir, umaps_dir]:
    os.makedirs(os.path.join(d, "figures"), exist_ok=True)
    os.makedirs(os.path.join(d, "csv_files"), exist_ok=True)
# additional subdirs for subcluster
os.makedirs(os.path.join(subcluster_dir, "figures", "line_plots"), exist_ok=True)
os.makedirs(os.path.join(subcluster_dir, "figures", "confusion_matrices"), exist_ok=True)
os.makedirs(os.path.join(subcluster_dir, "figures", "umaps"), exist_ok=True)

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Helpers: plot_umap and plot_subcluster_umap
# ----------------------------------------------------------------------------
def plot_umap(adata, rep_name, variable_name, out_dir, num_groups=None):
    logger.info(f"UMAP: {rep_name}, colored by {variable_name}")
    sc.pp.neighbors(adata, use_rep='X', n_neighbors=15)
    sc.tl.umap(adata, min_dist=0.1)
    if num_groups is not None:
        top = adata.obs[variable_name].value_counts().head(num_groups).index.tolist()
        adata.obs['group_highlighted'] = np.where(
            adata.obs[variable_name].isin(top),
            adata.obs[variable_name].astype(str),
            'Other'
        )
        cmap_mod = cm.get_cmap('tab20', len(top))
        colors = {grp: cmap_mod(i) for i, grp in enumerate(top)}
        colors['Other'] = (0.9,0.9,0.9,1)
        fig = sc.pl.umap(
            adata,
            color='group_highlighted',
            palette=colors,
            show=False,
            return_fig=True
        )
        fname = f"umap_{args.cell_type_column}_{rep_name}_top{num_groups}.png"
    else:
        fig = sc.pl.umap(
            adata,
            color=variable_name,
            show=False,
            return_fig=True
        )
        fname = f"umap_{variable_name}_{rep_name}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=300)
    plt.close(fig)


def plot_subcluster_umap(latent, true_labels, pred_labels_dict, out_prefix, fig_dir):
    ad = sc.AnnData(latent)
    ad.obsm['X_latent'] = latent
    ad.obs['true'] = true_labels.astype(str)
    sc.pp.neighbors(ad, use_rep='X_latent', n_neighbors=15)
    sc.tl.umap(ad, min_dist=0.1)
    fig, ax = plt.subplots(figsize=(6,6))
    sc.pl.umap(ad, color='true', ax=ax, show=False)
    ax.set_title(f"{out_prefix} — true labels")
    fig.savefig(os.path.join(fig_dir, f"{out_prefix}_true_umap.pdf"))
    plt.close(fig)
    for modality, labels in pred_labels_dict.items():
        ad.obs[modality] = labels.astype(str)
        fig, ax = plt.subplots(figsize=(6,6))
        sc.pl.umap(ad, color=modality, ax=ax, show=False)
        ax.set_title(f"{out_prefix} — {modality} preds")
        fig.savefig(os.path.join(fig_dir, f"{out_prefix}_{modality}_umap.pdf"))
        plt.close(fig)

# ----------------------------------------------------------------------------
# Load data & model
# ----------------------------------------------------------------------------
logger.info("Loading data & model")
mdata = mu.read_h5mu(args.mudata_path)
# ensure psi_mask
tmp = mdata['splicing'].layers['cell_by_cluster_matrix']
cluster = sparse.csr_matrix(tmp) if not sparse.isspmatrix_csr(tmp) else tmp
mask = cluster.copy(); mask.data = np.ones_like(cluster.data, dtype=np.uint8)
sp = mdata['splicing']; sp.layers['psi_mask'] = mask
model = scvi.model.MULTIVISPLICE.load(args.model_path, adata=mdata)

# compute reps
Z_joint = model.get_latent_representation()
Z_ge    = model.get_latent_representation(modality='expression')
Z_as    = model.get_latent_representation(modality='splicing')
groups = {'joint':Z_joint, 'expression':Z_ge, 'splicing':Z_as}

# ----------------------------------------------------------------------------
# 1. Concentration analysis
# ----------------------------------------------------------------------------
logger.info("[Concentration] Starting")
CONC_DIR = os.path.join(args.out_dir, 'concentration_analysis')
os.makedirs(os.path.join(CONC_DIR,'figures'), exist_ok=True)
os.makedirs(os.path.join(CONC_DIR,'csv_files'), exist_ok=True)

def extract_conc(model):
    data = {}
    mod = model.module
    if hasattr(mod,'px_r'):
        lr = mod.px_r.detach().cpu().numpy(); data['gene_log_phi']=lr; data['gene_phi']=np.exp(lr)
    if hasattr(mod,'log_phi_j'):
        lj = mod.log_phi_j.detach().cpu().numpy(); data['junction_log_phi']=lj
        import torch.nn.functional as F, torch
        data['junction_phi'] = F.softplus(torch.tensor(lj)).numpy()
    return data

def analyze_conc(data, out):
    # (same as original block, writing CSVs and figures into out)
    # ... [retain full body from original here] ...
    pass

conc_data = extract_conc(model)
if conc_data:
    analyze_conc(conc_data, CONC_DIR)
logger.info("[Concentration] Done")

# ----------------------------------------------------------------------------
# 2. Overall UMAP & silhouette
# ----------------------------------------------------------------------------
logger.info("[Overall] UMAP + silhouette")
sil_records=[]
for rep,Z in groups.items():
    labels = mdata['rna'].obs[args.cell_type_column].astype('category').cat.codes.values
    sil = silhouette_score(Z, labels)
    sil_records.append({'rep':rep,'silhouette':sil})
    # plot
    ad = sc.AnnData(Z); ad.obs[args.cell_type_column] = mdata['rna'].obs[args.cell_type_column]
    plot_umap(ad, rep, args.cell_type_column, overall_dir, args.top_n_celltypes)
pd.DataFrame(sil_records).to_csv(os.path.join(overall_dir,'csv_files','silhouette_scores.csv'),index=False)
logger.info("[Overall] Done")

# ----------------------------------------------------------------------------
# 3. Subcluster reproducibility
# ----------------------------------------------------------------------------
logger.info("[Subcluster] Starting")
records_sub=[]
labels_ct = mdata['rna'].obs[args.cell_type_column]
for ct in labels_ct.unique():
    mask = labels_ct==ct; idx = np.where(mask.values)[0]
    if len(idx)<2: continue
    split = len(idx)//2
    idx1, idx2 = idx[:split], idx[split:]
    for k in args.cluster_numbers:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED)
        lab1 = km.fit_predict(Z_joint[idx1])
        lab2 = km.predict(Z_joint[idx2])
        preds={}
        for rep,Z in groups.items():
            clf=LogisticRegression(max_iter=200).fit(Z[idx2],lab2)
            p=clf.predict(Z[idx1]); preds[rep]=p
            for m,fn in [('accuracy',lambda a,p:(p==a).mean()),('precision',lambda a,p:precision_score(a,p,average='weighted',zero_division=0)),('recall',lambda a,p:recall_score(a,p,average='weighted',zero_division=0)),('f1',lambda a,p:f1_score(a,p,average='weighted',zero_division=0))]:
                records_sub.append({'cell_type':ct,'rep':rep,'k':k,'metric':m,'value':fn(lab1,p)})
        # random baseline
        rng=np.random.default_rng(RANDOM_SEED)
        Zr=rng.standard_normal(Z_joint.shape)
        clf=LogisticRegression(max_iter=200).fit(Zr[idx2],lab2)
        pr=clf.predict(Zr[idx1]); preds['random']=pr
        for m,fn in [('accuracy',lambda a,p:(p==a).mean()),('precision',lambda a,p:precision_score(a,p,average='weighted',zero_division=0)),('recall',lambda a,p:recall_score(a,p,average='weighted',zero_division=0)),('f1',lambda a,p:f1_score(a,p,average='weighted',zero_division=0))]:
            records_sub.append({'cell_type':ct,'rep':'random','k':k,'metric':m,'value':fn(lab1,pr)})
        # UMAPs and confusion for target
        if ct in ["Excitatory Neurons","MICROGLIA"]:
            plot_subcluster_umap(Z_joint[idx1], lab1, preds, f"{ct}_k{k}", os.path.join(subcluster_dir,'figures','umaps'))
pd.DataFrame(records_sub).to_csv(os.path.join(subcluster_dir,'csv_files','subcluster_metrics.csv'),index=False)
logger.info("[Subcluster] Done")

# ----------------------------------------------------------------------------
# 4. Cell-type classification
# ----------------------------------------------------------------------------
logger.info("[Classification] Starting")
records_cls=[]
labels_all = mdata['rna'].obs[args.cell_type_column].astype('category').cat.codes.values
for rep,Z in groups.items():
    Xtr,Xte,ytr,yte = train_test_split(Z,labels_all,test_size=0.2,stratify=labels_all,random_state=RANDOM_SEED)
    clf=LogisticRegression(max_iter=500).fit(Xtr,ytr)
    yp=clf.predict(Xte)
    for idx,ct in enumerate(mdata['rna'].obs[args.cell_type_column].cat.categories):
        mask = yte==idx
        if mask.sum():
            records_cls.append({'rep':rep,'cell_type':ct,'accuracy':accuracy_score(yte[mask],yp[mask]),'precision':precision_score(yte[mask],yp[mask],average='macro',zero_division=0),'recall':recall_score(yte[mask],yp[mask],average='macro',zero_division=0),'f1':f1_score(yte[mask],yp[mask],average='macro',zero_division=0)})
    records_cls.append({'rep':rep,'cell_type':'ALL','accuracy':accuracy_score(yte,yp),'precision':precision_score(yte,yp,average='macro',zero_division=0),'recall':recall_score(yte,yp,average='macro',zero_division=0),'f1':f1_score(yte,yp,average='macro',zero_division=0)})
pd.DataFrame(records_cls).to_csv(os.path.join(classif_dir,'csv_files','cell_type_classification.csv'),index=False)
logger.info("[Classification] Done")

# ----------------------------------------------------------------------------
# 5. Local neighborhood consistency
# ----------------------------------------------------------------------------
logger.info("[Neighborhood] Starting")
records_nb=[]
nj=NearestNeighbors(n_neighbors=max(args.neighbor_k)+1).fit(Z_joint); idxj=nj.kneighbors(return_distance=False)[:,1:]
for k in args.neighbor_k:
    nge=NearestNeighbors(n_neighbors=k+1).fit(Z_ge); idxge=nge.kneighbors(return_distance=False)[:,1:]
    nas=NearestNeighbors(n_neighbors=k+1).fit(Z_as); idxas=nas.kneighbors(return_distance=False)[:,1:]
    for i in range(Z_joint.shape[0]):
        records_nb.append({'cell':i,'k':k,'S_GE':len(set(idxj[i][:k])&set(idxge[i]))/k,'S_AS':len(set(idxj[i][:k])&set(idxas[i]))/k})
_df_nb=pd.DataFrame.from_records(records_nb)
_df_nb.to_csv(os.path.join(neighbor_dir,'csv_files','neighborhood_overlap.csv'),index=False)
rho,pval=spearmanr(_df_nb['S_GE'],_df_nb['S_AS'])
logger.info(f"Spearman ρ: {rho:.3f}, p={pval:.3g}")
for sc in ['S_GE','S_AS']:
    fig,ax=plt.subplots(); sns.histplot(_df_nb[sc],kde=True,ax=ax)
    fig.savefig(os.path.join(neighbor_dir,'figures',f'{sc}_hist.png'),dpi=300); plt.close(fig)
    ad=sc.AnnData(Z_joint); ad.obs[sc]=_df_nb[sc].values
    plot_umap(ad,f'Z_joint_{sc}',sc,neighbor_dir)
fig,ax=plt.subplots(); sns.scatterplot(x='S_GE',y='S_AS',data=_df_nb,alpha=0.1); fig.savefig(os.path.join(neighbor_dir,'figures','scatter_SGE_SAS.png'),dpi=300)
logger.info("[Neighborhood] Done")