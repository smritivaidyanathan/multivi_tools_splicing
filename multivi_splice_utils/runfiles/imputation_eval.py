#!/usr/bin/env python3
"""
imputation_benchmark.py

Benchmark imputation accuracy of multiple models across varying
fractions of missing gene‐expression and splicing data,
with aggressive memory cleanup.
"""

import os
import random
import gc
import numpy as np
import pandas as pd
import mudata as mu
import scvi
import scanpy as sc
from scipy import sparse
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Config from environment / defaults
# ------------------------------------------------------------------------------
IMPUTATION_EVAL_OUTDIR = os.environ.get("IMPUTATION_EVAL_OUTDIR", "./imputation_eval_output")
FIG_DIR = os.path.join(IMPUTATION_EVAL_OUTDIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
CSV_OUT = os.path.join(IMPUTATION_EVAL_OUTDIR, "imputation_results.csv")

UMAP_GROUP = "cell_type_grouped"
# list of (pct_rna_missing, pct_splice_missing)
MISSING_PCT_PAIRS = [(0.0, 0.2), (0.2, 0.0), (0.2, 0.2)]
SEED = 42

# ------------------------------------------------------------------------------
# Utilities: corrupt & evaluate
# ------------------------------------------------------------------------------
def corrupt_mudata(mdata, pct_rna=0.0, pct_splice=0.0, seed=None):
    """
    Only copies & modifies the layers we need, not the full MuData
    """
    rng = np.random.default_rng(seed)
    corrupted = mdata.copy()  # unfortunately MuData.copy is deep, so we'll gc after
    orig = {'rna': None, 'splice': None}

    # --- RNA masking on sparse ---
    if pct_rna > 0:
        X = corrupted['rna'].layers['raw_counts']
        # get nonzero coords
        rows, cols = X.nonzero()
        nrm = int(len(rows) * pct_rna)
        idx = rng.choice(len(rows), nrm, replace=False)
        coords = np.stack([rows[idx], cols[idx]], axis=1)
        # pull values
        vals = X[coords[:,0], coords[:,1]].A1 if sparse.isspmatrix(X) else X[coords[:,0], coords[:,1]]
        # zero them
        X = X.tolil()
        X[coords[:,0], coords[:,1]] = 0
        corrupted['rna'].layers['raw_counts'] = X.tocsr()
        orig['rna'] = (coords, vals)

    # --- Splicing masking on sparse ---
    if pct_splice > 0:
        atse = corrupted['splicing'].layers['cell_by_cluster_matrix'].tocoo()
        junc = corrupted['splicing'].layers['cell_by_junction_matrix'].tocoo()
        ratio = corrupted['splicing'].layers['junc_ratio']
        # build mask of valid entries
        # atse.row, atse.col, atse.data; pick those with data>0
        valid_mask = atse.data > 0
        vr, vc = atse.row[valid_mask], atse.col[valid_mask]
        # also need to check junc and ratio at same coords
        # build dict for junc and ratio lookups
        j_dict = {(r,c): v for r,c,v in zip(junc.row, junc.col, junc.data)}
        r_arr = ratio.toarray() if sparse.isspmatrix(ratio) else np.array(ratio)
        valid = [(r,c) for r,c in zip(vr,vc) if j_dict.get((r,c),-1) >= 0 and not np.isnan(r_arr[r,c])]
        nrm = int(len(valid) * pct_splice)
        sel = rng.choice(len(valid), nrm, replace=False)
        coords = np.array([valid[i] for i in sel])
        orig_vals = np.vstack([
            atse.data[valid_mask][sel],
            [j_dict[(r,c)] for r,c in coords],
            [r_arr[r,c] for r,c in coords],
        ]).T
        # zero them
        atse_lil = atse.tolil(); junc_lil = junc.tolil()
        for r,c in coords:
            atse_lil[r,c] = 0
            junc_lil[r,c] = 0
            r_arr[r,c] = 0
        corrupted['splicing'].layers['cell_by_cluster_matrix'] = atse_lil.tocsr()
        corrupted['splicing'].layers['cell_by_junction_matrix'] = junc_lil.tocsr()
        corrupted['splicing'].layers['junc_ratio'] = sparse.csr_matrix(r_arr) if sparse.isspmatrix(ratio) else r_arr
        orig['splice'] = (coords, orig_vals)

    # rebuild psi_mask
    sp = corrupted['splicing']
    clu = sp.layers['cell_by_cluster_matrix']
    mask = (clu > 0).astype(np.uint8) if not sparse.isspmatrix_csr(clu) else clu.copy()
    if not sparse.isspmatrix_csr(mask):
        mask = sparse.csr_matrix(mask)
    sp.layers['psi_mask'] = mask

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
        'mse': float((diff**2).mean()),
        'median_l1': float(abs(diff).median()),
        'spearman': float(spearmanr(x1, x2).correlation),
    }


# ------------------------------------------------------------------------------
# Define your models
# ------------------------------------------------------------------------------
def define_models():
    def build(mdata_corr, latent_dim):
        scvi.model.MULTIVISPLICE.setup_mudata(
            mdata_corr,
            batch_key="mouse.id",
            size_factor_key="X_library_size",
            rna_layer="raw_counts",
            junc_ratio_layer="junc_ratio",
            atse_counts_layer="cell_by_cluster_matrix",
            junc_counts_layer="cell_by_junction_matrix",
            psi_mask_layer="psi_mask",
            modalities={"rna_layer": "rna", "junc_ratio_layer": "splicing"},
        )
        return scvi.model.MULTIVISPLICE(
            mdata_corr,
            n_genes=(mdata_corr['rna'].var['modality']=="Gene_Expression").sum(),
            n_junctions=(mdata_corr['splicing'].var['modality']=="Splicing").sum(),
            n_latent=latent_dim,
        )

    return {
        "z30": lambda md: build(md, 30),
        "z20": lambda md: build(md, 20),
    }


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print("Reading full MuData…")
    mdata_full = mu.read_h5mu(MUDATA_PATH)
    models = define_models()
    records = []

    for pct_rna, pct_splice in MISSING_PCT_PAIRS:
        label = f"r{pct_rna:.2f}_s{pct_splice:.2f}"
        print(f"\n--- Missingness {label} ---")

        # corrupt copy
        mdata_corr, orig_vals = corrupt_mudata(mdata_full, pct_rna, pct_splice, SEED)
        del mdata_full  # free
        gc.collect()

        for name, setup_fn in models.items():
            print(f"Training {name} …")
            md = mdata_corr.copy()  # copy just once per model
            model = setup_fn(md)
            model.train(max_epochs=10)

            # impute & eval
            expr = model.get_normalized_expression(return_numpy=True)
            lib  = model.get_library_size_factors()['expression']
            imp_expr = expr * lib[:,None]
            imp_spl  = model.get_normalized_splicing(return_numpy=True)
            m_rna = evaluate_imputation(orig_vals['rna'], imp_expr)
            m_spl = evaluate_imputation(orig_vals['splice'], imp_spl)

            # record
            records.append(dict(
                model=name, pct_rna=pct_rna, pct_splice=pct_splice, label=label,
                **{f"rna_{k}":v for k,v in m_rna.items()},
                **{f"spl_{k}":v for k,v in m_spl.items()},
            ))

            # UMAP per‐model
            lat = model.get_latent_representation()
            ad = sc.AnnData(lat)
            ad.obs = md['rna'].obs.copy()
            sc.pp.neighbors(ad, use_rep="X")
            sc.tl.umap(ad, min_dist=0.2)
            fig = sc.pl.umap(ad, color=UMAP_GROUP, show=False, return_fig=True, title=f"{name} {label}")
            fig.savefig(os.path.join(FIG_DIR, f"{name}_umap_{label}.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)

            # cleanup
            del lat, ad, fig
            del model, md
            gc.collect()

        # cleanup per‐mask
        del mdata_corr, orig_vals
        gc.collect()

    # write out metrics
    df = pd.DataFrame.from_records(records)
    df.to_csv(CSV_OUT, index=False)
    print("Metrics →", CSV_OUT)


if __name__ == "__main__":
    main()
