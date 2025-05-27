#!/usr/bin/env python3
"""
latent_space_eval.py

Combined script to evaluate a single trained MULTIVISPLICE model's latent spaces and local neighborhood consistency:
 1. Overall UMAP and silhouette

Each section prints start/finish messages and writes outputs accordingly for one model.
"""

import os
import random
import logging
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import torch
from sklearn.decomposition import PCA

# set plotting defaults
sns.set_style("whitegrid")
sns.set_context("talk")

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Double-check torch uses CPU
assert not torch.cuda.is_available(), "Torch sees a GPU — CPU-only mode failed!"

# Path to the MuData and the single model
USE_FULL_DATASET = True
FULL_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/052025/aligned__ge_splice_combined_20250513_035938.h5mu" # all 5k genes here...
SUBSET_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/052025/SUBSETTOP5CELLSTYPES_aligned__ge_splice_combined_20250513_035938.h5mu"

_default_mudata = FULL_PATH if USE_FULL_DATASET else SUBSET_PATH
MUDATA_PATH = os.environ.get("MUDATA_PATH", _default_mudata)
MODEL_NAME = "MultiVISpliceTraining_20250526_180124_job4883640"
MODEL_PATH = "/gpfs/commons/home/kisaev/multi_vi_splice_runs/MultiVISpliceTraining_20250526_180124_job4883640/models"

# Make output directory to save things in 
OUTDIR = "/gpfs/commons/home/kisaev/multivi_tools_splicing/results/latent_space_eval/PCA_test"
os.makedirs(OUTDIR, exist_ok=True)

# ----------------------------------------------------------------------------
# 1. Overall latent eval: UMAP + silhouette
# ----------------------------------------------------------------------------
# This section computes the joint, expression-specific, and splicing-specific latent
# embeddings for all cells, generates UMAP visualizations colored by cell type, and
# calculates silhouette scores to assess cluster separation in each latent space.

print(f"Reading in MuData from {MUDATA_PATH}")
mdata = mu.read_h5mu(MUDATA_PATH)
print(mdata)

# Ensure psi_mask
sp = mdata['splicing']
cluster = sp.layers['cell_by_cluster_matrix']
if not sparse.isspmatrix_csr(cluster): cluster = sparse.csr_matrix(cluster)
mask = cluster.copy(); mask.data = np.ones_like(cluster.data, dtype=np.uint8)
sp.layers['psi_mask'] = mask

# Load model once
print(f"Loading model from {MODEL_PATH}")
model = scvi.model.MULTIVISPLICE.load(MODEL_PATH, adata=mdata, accelerator="cpu", device="auto")
print(f"Model loaded successfully")

# compute representations
print(f"Extracting latent representations...")
Z_joint = model.get_latent_representation()
Z_ge = model.get_latent_representation(modality='expression')
Z_as = model.get_latent_representation(modality='splicing')

# For each latent space, calculate PCA on it + number of PCs needed to explain 90% of variance
def perform_pca_analysis(Z, name, outdir, perc_explained=0.9, n_components=20):
    """Perform PCA analysis on a latent space and create visualizations.
    
    Args:
        Z: Latent space data
        name: Name of the latent space
        outdir: Output directory for plots
        perc_explained: Percentage of variance to explain (default: 0.9)
        n_components: Number of principal components to compute (default: 20)
    """
    print(f"\nAnalyzing {name} latent space...")
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    Z_pca = pca.fit_transform(Z)
    
    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components_explained = np.argmax(cumulative_variance >= perc_explained) + 1
    print(f"Total variance explained by first {n_components_explained} PCs: {cumulative_variance[n_components_explained-1]:.3f}")
    
    # Plot cumulative explained variance
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'b-', linewidth=2)
    plt.axhline(y=perc_explained, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=n_components_explained, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Number of Components', fontsize=12)
    plt.ylabel('Cumulative Explained Variance', fontsize=12)
    plt.title(f'PCA Cumulative Explained Variance - {name}', fontsize=14)
    plt.grid(False)
    plt.savefig(os.path.join(outdir, f'pca_variance_{name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save PCA results
    results = {
        'name': name,
        'perc_explained': perc_explained,
        'n_components': n_components,
        'n_components_explained': n_components_explained,
        'total_variance_explained': cumulative_variance[n_components_explained-1],
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': cumulative_variance
    }
    
    return results

# Perform PCA analysis for each latent space
latent_spaces = {
    'joint': Z_joint,
    'expression': Z_ge,
    'splicing': Z_as
}

pca_results = {}
for name, Z in latent_spaces.items():
    pca_results[name] = perform_pca_analysis(Z, name, OUTDIR, n_components=20)

# Save summary results from analysis 
summary = pd.DataFrame({
    'Latent Space': [results['name'] for results in pca_results.values()],
    'Percentage Explained': [results['perc_explained'] for results in pca_results.values()],
    'Number of Components': [results['n_components'] for results in pca_results.values()],
    'Components for Target Variance': [results['n_components_explained'] for results in pca_results.values()],
    'Total Variance Explained': [results['total_variance_explained'] for results in pca_results.values()]
})

summary.to_csv(os.path.join(OUTDIR, 'pca_summary.csv'), index=False)
print("\nPCA analysis complete. Results saved to:", OUTDIR)

# from loaded model print the learned concentration parameters
# print(model.get_concentration_parameters())

# conda activate scvi-env
# export LATENT_EVAL_OUTDIR=/gpfs/commons/home/kisaev/multivi_tools_splicing/results/latent_space_eval	
# mkdir -p $LATENT_EVAL_OUTDIR/logs
#sbatch --partition=gpu \
#       --gres=gpu:1 \
#       --mem=300G \
#       --job-name=latent_PCA \
#       --output=$LATENT_EVAL_OUTDIR/logs/latent_eval_%j.out \
#      --error=$LATENT_EVAL_OUTDIR/logs/latent_eval_%j.err \
#     --wrap="python /gpfs/commons/home/kisaev/multivi_tools_splicing/multivi_splice_utils/runfiles/PC_test_latent_spaces.py"
