# %% cell: silhouette analysis and save outputs
import os
import scipy.sparse as sp
import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
import anndata as ad
import pandas as pd
import scipy.sparse as sp
import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack
import h5py
import anndata as ad
import mudata as mu
import scanpy as sc

# --- Config ---
CELL_TYPE_COLUMN = 'broad_cell_type'
TOP_N_CELLTYPES = 20
OUT_DIR = "/gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/figures"  # <-- update this to your desired output folder


ROOT_PATH = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/052025/"

ATSE_DATA_PATH = ROOT_PATH + "aligned_splicing_data_20250513_035938.h5ad"
GE_DATA_PATH = ROOT_PATH + "aligned_gene_expression_data_20250513_035938.h5ad"
OUTPUT_MUDATA_PATH = ROOT_PATH + "aligned__ge_splice_combined_20250513_035938.h5mu"
REDO_JUNC_RATIO = False

print("ATSE data path:", ATSE_DATA_PATH)
print("GE data path:  ", GE_DATA_PATH)
print("Output MuData path:", OUTPUT_MUDATA_PATH)

os.makedirs(OUT_DIR, exist_ok=True)

# --- 1) Prepare expression embedding ---
ad_expr = ad.read_h5ad(GE_DATA_PATH)
print("GE AnnData:", ad_expr)
ln = ad_expr.layers['length_norm']
ad_expr.X = ln.toarray() if sp.issparse(ln) else ln.copy()
sc.pp.log1p(ad_expr)
sc.pp.pca(ad_expr, n_comps=20, svd_solver='arpack')

# --- 2) Prepare splicing embedding ---
ad_splice = ad.read_h5ad(ATSE_DATA_PATH)
print("ATSE AnnData:", ad_splice)
jr = ad_splice.layers['junc_ratio']
ad_splice.X = jr.toarray() if sp.issparse(jr) else jr.copy()
sc.pp.pca(ad_splice, n_comps=20, svd_solver='arpack')

# --- 3) Align cells & labels ---
common = ad_expr.obs_names.intersection(ad_splice.obs_names)
ad_expr = ad_expr[common]
ad_splice = ad_splice[common]
labels = ad_expr.obs[CELL_TYPE_COLUMN].values

# --- 4) Compute silhouette for each cell in each space ---
expr_sil = silhouette_samples(ad_expr.obsm['X_pca'], labels)
splice_sil = silhouette_samples(ad_splice.obsm['X_pca'], labels)

# --- 5) Summarize by cell type ---
df = pd.DataFrame({
    'expr_sil': expr_sil,
    'splice_sil': splice_sil,
    'cell_type': labels
})
summary = df.groupby('cell_type')[['expr_sil','splice_sil']].mean()
summary['delta'] = summary['splice_sil'] - summary['expr_sil']

# --- 6) Save results ---
df.to_csv(os.path.join(OUT_DIR, 'silhouette_per_cell.csv'), index=False)
summary.to_csv(os.path.join(OUT_DIR, 'silhouette_summary_by_celltype.csv'))

# --- 7) Scatter plot of per-cell scores ---
plt.figure(figsize=(6,6))
plt.scatter(df['expr_sil'], df['splice_sil'], s=5, alpha=0.3)
xmin, xmax = df['expr_sil'].min(), df['expr_sil'].max()
plt.plot([xmin, xmax], [xmin, xmax], 'k--', lw=1)
plt.xlabel('Expression silhouette')
plt.ylabel('Splicing silhouette')
plt.title('Per-cell separation by modality')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'silhouette_scatter.png'), dpi=300, bbox_inches='tight')
plt.close()
