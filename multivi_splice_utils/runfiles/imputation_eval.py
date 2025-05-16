#!/usr/bin/env python3
"""
imputation_benchmark.py

Benchmark imputation accuracy of multiple models across varying
fractions of missing gene‐expression and splicing data.
"""

import os
import random
import numpy as np
import pandas as pd
import mudata as mu
import scvi
import scanpy as sc
from scipy import sparse
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# ---------------------------
# User configurations
# ---------------------------

MUDATA_PATH = "/gpfs/commons/groups/knowles_lab/Karin/TMS_MODELING/DATA_FILES/ALL_CELLS/022025/TMS_MUData_GE_ATSE_20250209_165655.h5mu"
RESULTS_CSV = "./imputation_results.csv"
FIG_DIR = "./figures_imputation"
os.makedirs(FIG_DIR, exist_ok=True)
UMAP_GROUP = "cell_type_grouped"
# list of (pct_rna_missing, pct_splice_missing)
MISSING_PCT_PAIRS = [(0.0, 0.2), (0.2, 0.0), (0.2, 0.2)]
SEED = 42

# ---------------------------
# Utilities: corrupt & evaluate
# ---------------------------

def corrupt_mudata(mdata, pct_rna=0.0, pct_splice=0.0, seed=None):
    rng = np.random.default_rng(seed)
    corrupted = mdata.copy()
    orig = {'rna': None, 'splice': None}
    
    # RNA masking
    if pct_rna > 0:
        X = corrupted['rna'].layers['raw_counts']
        arr = X.toarray() if sparse.issparse(X) else X.copy()
        nz = np.argwhere(arr != 0)
        nrm = int(len(nz) * pct_rna)
        sel = rng.choice(len(nz), nrm, replace=False)
        coords = nz[sel]
        vals = arr[coords[:,0], coords[:,1]].copy()
        arr[coords[:,0], coords[:,1]] = 0
        corrupted['rna'].layers['raw_counts'] = (
            sparse.csr_matrix(arr) if sparse.issparse(X) else arr
        )
        orig['rna'] = (coords, vals)
    
    # Splicing masking
    if pct_splice > 0:
        atse = corrupted['splicing'].layers['cell_by_cluster_matrix']
        junc = corrupted['splicing'].layers['cell_by_junction_matrix']
        ratio = corrupted['splicing'].layers['junc_ratio']

        a_arr = atse.toarray() if sparse.issparse(atse) else atse.copy()
        j_arr = junc.toarray() if sparse.issparse(junc) else junc.copy()
        r_arr = (
            ratio.toarray().astype(float) if sparse.issparse(ratio)
            else np.array(ratio, copy=True, dtype=float)
        )

        valid = np.argwhere((a_arr > 0) & (j_arr >= 0) & (~np.isnan(r_arr)))
        nrm = int(len(valid) * pct_splice)
        sel = rng.choice(len(valid), nrm, replace=False)
        coords = valid[sel]
        orig_vals = np.vstack([
            a_arr[coords[:,0], coords[:,1]],
            j_arr[coords[:,0], coords[:,1]],
            r_arr[coords[:,0], coords[:,1]],
        ]).T

        a_arr[coords[:,0], coords[:,1]] = 0
        j_arr[coords[:,0], coords[:,1]] = 0
        r_arr[coords[:,0], coords[:,1]] = 0

        corrupted['splicing'].layers['cell_by_cluster_matrix'] = (
            sparse.csr_matrix(a_arr) if sparse.issparse(atse) else a_arr
        )
        corrupted['splicing'].layers['cell_by_junction_matrix'] = (
            sparse.csr_matrix(j_arr) if sparse.issparse(junc) else j_arr
        )
        corrupted['splicing'].layers['junc_ratio'] = r_arr
        orig['splice'] = (coords, orig_vals)

    return corrupted, orig


def evaluate_imputation(original, imputed):
    coords, vals = original
    pred = imputed[coords[:,0], coords[:,1]]
    if vals.ndim == 2 and vals.shape[1] == 3:
        atse, true_j, _ = vals.T
        imp_counts = pred * atse
        diff = imp_counts - true_j
        x1, x2 = true_j, imp_counts
    else:
        true_c = vals
        diff = pred - true_c
        x1, x2 = true_c, pred

    return {
        'mse': float(np.mean(diff**2)),
        'median_l1': float(np.median(np.abs(diff))),
        'spearman': float(spearmanr(x1, x2).correlation),
    }

# ---------------------------
# Define your models
# ---------------------------

def define_models():
    def setup_multivisplice(mdata_corr):
        scvi.model.MULTIVISPLICE.setup_mudata(
            mdata_corr,
            batch_key="mouse.id",
            rna_layer="raw_counts",
            junc_ratio_layer="junc_ratio",
            atse_counts_layer="cell_by_cluster_matrix",
            junc_counts_layer="cell_by_junction_matrix",
        )
        return scvi.model.MULTIVISPLICE(
            mdata_corr,
            n_genes=(mdata_corr['rna'].var['modality']=="Gene_Expression").sum(),
            n_junctions=(mdata_corr['splicing'].var['modality']=="Splicing").sum(),
        )

    return {"MULTIVISPLICE": setup_multivisplice}

# ---------------------------
# Main
# ---------------------------

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print("Loading full MuData…")
    mdata_full = mu.read_h5mu(MUDATA_PATH)

    models = define_models()
    all_metrics = []

    for pct_rna, pct_splice in MISSING_PCT_PAIRS:
        label = f"r{pct_rna:.2f}_s{pct_splice:.2f}"
        print(f"\n--- Missingness {label} ---")
        mdata_corr, orig_vals = corrupt_mudata(
            mdata_full, pct_rna=pct_rna, pct_splice=pct_splice, seed=SEED
        )
        latents = {}

        for name, setup_fn in models.items():
            print(f"Training {name} …")
            mdata_copy = mdata_corr.copy()
            model = setup_fn(mdata_copy)
            model.train()
            latents[name] = model.get_latent_representation()

            expr = model.get_normalized_expression(return_numpy=True)
            lib  = model.get_library_size_factors()['expression']
            imp_expr = expr * lib[:, None]
            imp_spl  = model.get_normalized_splicing(return_numpy=True)

            m_rna = evaluate_imputation(orig_vals['rna'], imp_expr)
            m_spl = evaluate_imputation(orig_vals['splice'], imp_spl)

            all_metrics.append({
                'model': name,
                'pct_rna': pct_rna,
                'pct_splice': pct_splice,
                'label': label,
                **{f"rna_{k}": v for k,v in m_rna.items()},
                **{f"spl_{k}": v for k,v in m_spl.items()},
            })
            print(f" → RNA MSE={m_rna['mse']:.3f}, SPL MSE={m_spl['mse']:.3f}")

        # UMAP
        fig, axes = plt.subplots(
            1, len(models), figsize=(5*len(models), 4), squeeze=False
        )
        for i, (name, lat) in enumerate(latents.items()):
            ad = sc.AnnData(lat)
            ad.obs = mdata_corr['rna'].obs.copy()
            sc.pp.neighbors(ad, use_rep='X', show=False)
            sc.tl.umap(ad, min_dist=0.2, show=False)
            sc.pl.umap(
                ad, color=UMAP_GROUP, ax=axes[0,i],
                show=False, title=name,
                legend_loc='right margin'
            )
        fig.suptitle(f"UMAPs {label}", y=1.02)
        fig.tight_layout()
        fp = os.path.join(FIG_DIR, f"umap_{label}.png")
        fig.savefig(fp, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Saved", fp)

    # save CSV
    df = pd.DataFrame(all_metrics)
    df.to_csv(RESULTS_CSV, index=False)
    print("Metrics →", RESULTS_CSV)

    # bar charts
    metrics = [c for c in df.columns if c not in ('model','pct_rna','pct_splice','label')]
    for met in metrics:
        plt.figure(figsize=(6,4))
        pivot = df.pivot(index='label', columns='model', values=met)
        pivot.plot(kind='bar')
        plt.title(met)
        plt.ylabel(met)
        plt.xlabel('missingness')
        plt.tight_layout()
        fp = os.path.join(FIG_DIR, f"{met}_bar.png")
        plt.savefig(fp, dpi=300)
        plt.close()
        print("Saved", fp)

if __name__ == "__main__":
    main()
