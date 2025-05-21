#!/usr/bin/env python3
"""
imputation_benchmark.py

Benchmark imputation accuracy under varying
fractions of missing gene‐expression and splicing data,
with minimal peak memory usage—and verbose logging.
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
IMPUTATION_EVAL_OUTDIR = os.environ.get(
    "IMPUTATION_EVAL_OUTDIR", "./imputation_eval_output"
)
FIG_DIR = os.path.join(IMPUTATION_EVAL_OUTDIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
CSV_OUT = os.path.join(IMPUTATION_EVAL_OUTDIR, "imputation_results.csv")

MUDATA_PATH = ("/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/052025/SUBSETTOP5CELLSTYPES_aligned__ge_splice_combined_20250513_035938.h5mu")

UMAP_GROUP = "broad_cell_type"
MISSING_PCT_PAIRS = [(0.0, 0.2), (0.2, 0.0), (0.2, 0.2)]
SEED = 42

# ------------------------------------------------------------------------------
# Utilities: corrupt & evaluate
# ------------------------------------------------------------------------------
import numpy as np
from scipy import sparse

def corrupt_mudata_inplace(mdata, pct_rna, pct_splice, seed):
    print(f"  → corrupt_mudata_inplace(pct_rna={pct_rna}, pct_splice={pct_splice}) start", flush=True)
    rng = np.random.default_rng(seed)
    orig = {'rna': None, 'splice': None}

    # --- RNA masking (in place) ---
    if pct_rna > 0:
        print("    * RNA masking...", flush=True)
        X_csr = mdata['rna'].layers['raw_counts']
        if not sparse.isspmatrix_csr(X_csr):
            X_csr = X_csr.tocsr()

        # pick random nonzero entries
        rows, cols = X_csr.nonzero()
        data = X_csr.data
        nnz = len(rows)
        nrm = int(nnz * pct_rna)
        idx = rng.choice(nnz, nrm, replace=False)
        mask_rows, mask_cols = rows[idx], cols[idx]
        coords = np.stack([mask_rows, mask_cols], axis=1)
        orig_vals = data[idx]

        # build a sparse mask and zero out in one shot
        mask_mat = sparse.coo_matrix(
            (np.ones(nrm, dtype=X_csr.dtype), (mask_rows, mask_cols)),
            shape=X_csr.shape
        ).tocsr()
        X_csr = X_csr.multiply(1 - mask_mat)

        # write back
        mdata['rna'].layers['raw_counts'] = X_csr
        orig['rna'] = (coords, orig_vals)
        print(f"      masked {nrm} RNA entries", flush=True)

    # --- Splicing masking (in place) ---
    if pct_splice > 0:
        print("    * Splicing masking...", flush=True)
        sp_mod = 'splicing'
        A_csr = mdata[sp_mod].layers['cell_by_cluster_matrix']
        if not sparse.isspmatrix_csr(A_csr):
            A_csr = A_csr.tocsr()
        J_csr = mdata[sp_mod].layers['cell_by_junction_matrix']
        if not sparse.isspmatrix_csr(J_csr):
            J_csr = J_csr.tocsr()
        R_layer = mdata[sp_mod].layers['junc_ratio']
        if sparse.isspmatrix(R_layer):
            R_csr = R_layer.tocsr()

        # gather valid nonzeros
        rows, cols = A_csr.nonzero()
        data = A_csr.data
        j_vals = J_csr[rows, cols].A1
        if sparse.isspmatrix(R_layer):
            r_vals = R_csr[rows, cols].A1
        else:
            r_vals = R_layer[rows, cols]

        good = (data > 0) & (j_vals >= 0) & (~np.isnan(r_vals))
        rows, cols, data, j_vals, r_vals = (
            rows[good], cols[good], data[good], j_vals[good], r_vals[good]
        )
        nnz_all = len(rows)
        nrm = int(nnz_all * pct_splice)
        idx2 = rng.choice(nnz_all, nrm, replace=False)
        mask_rows, mask_cols = rows[idx2], cols[idx2]
        coords = np.stack([mask_rows, mask_cols], axis=1)
        orig_vals = np.vstack([data[idx2], j_vals[idx2], r_vals[idx2]]).T

        # build mask and apply
        mask_mat = sparse.coo_matrix(
            (np.ones(nrm, dtype=A_csr.dtype), (mask_rows, mask_cols)),
            shape=A_csr.shape
        ).tocsr()
        A_csr = A_csr.multiply(1 - mask_mat)
        J_csr = J_csr.multiply(1 - mask_mat)
        if sparse.isspmatrix(R_layer):
            R_csr = R_csr.multiply(1 - mask_mat)
        else:
            R_layer[mask_rows, mask_cols] = 0

        # write back
        mdata[sp_mod].layers['cell_by_cluster_matrix'] = A_csr
        mdata[sp_mod].layers['cell_by_junction_matrix'] = J_csr
        if sparse.isspmatrix(R_layer):
            mdata[sp_mod].layers['junc_ratio'] = R_csr
        else:
            mdata[sp_mod].layers['junc_ratio'] = R_layer

        orig['splice'] = (coords, orig_vals)
        print(f"      masked {nrm} splicing entries (out of {nnz_all} valid)", flush=True)

    # --- rebuild psi_mask as before ---
    print("    * rebuilding psi_mask layer", flush=True)
    clu = mdata['splicing'].layers['cell_by_cluster_matrix']
    if sparse.isspmatrix_csr(clu):
        mask = clu.copy()
        mask.data = np.ones_like(mask.data, dtype=np.uint8)
    else:
        arr = (clu > 0).astype(np.uint8)
        mask = sparse.csr_matrix(arr)
    mdata['splicing'].layers['psi_mask'] = mask

    print("  → corrupt_mudata_inplace done", flush=True)
    return orig



def evaluate_imputation(original, imputed):
    print("    * evaluate_imputation...", flush=True)
    coords, vals = original
    pred = imputed[coords[:,0], coords[:,1]]
    if vals.ndim == 2:
        atse, true_j, _ = vals.T
        imp_counts = pred * atse
        diff = imp_counts - true_j
        x1, x2 = true_j, imp_counts
    else:
        true_c = vals
        diff = pred - true_c
        x1, x2 = true_c, pred
    res = {
        'mse': float((diff**2).mean()),
        'median_l1': float(np.median(np.abs(diff))),
        'spearman': float(spearmanr(x1, x2).correlation),
    }
    print(f"      -> eval results {res}", flush=True)
    return res

# ------------------------------------------------------------------------------
# Model factory
# ------------------------------------------------------------------------------
def define_models():
    def build(mdata, latent_dim, distribution):
        scvi.model.MULTIVISPLICE.setup_mudata(
            mdata,
            size_factor_key="X_library_size",
            rna_layer="raw_counts",
            junc_ratio_layer="junc_ratio",
            atse_counts_layer="cell_by_cluster_matrix",
            junc_counts_layer="cell_by_junction_matrix",
            psi_mask_layer="psi_mask",
            modalities={"rna_layer":"rna", "junc_ratio_layer":"splicing"},
        )
        return scvi.model.MULTIVISPLICE(
            mdata,
            n_genes=(mdata['rna'].var['modality']=="Gene_Expression").sum(),
            n_junctions=(mdata['splicing'].var['modality']=="Splicing").sum(),
            n_latent=latent_dim,
            splicing_architecture = "partial", 
            splicing_loss_type=distribution
        )

    return {
        "Splice-VI(Binomial Z=20)": lambda md: build(md, 20, "binomial"),
        "Splice-VI(Binomial Z=30)": lambda md: build(md, 30, "binomial"),
        "Splice-VI(Binomial Z=40)": lambda md: build(md, 40, "binomial"),
        "Splice-VI(Beta-Binomial Z=20)": lambda md: build(md, 20, "beta_binomial"),
        "Splice-VI(Beta-Binomial Z=30)": lambda md: build(md, 30, "beta_binomial"),
        "Splice-VI(Beta-Binomial Z=40)": lambda md: build(md, 40, "beta_binomial"),
    }

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    records = []

    for pct_rna, pct_splice in MISSING_PCT_PAIRS:
        label = f"r{pct_rna:.2f}_s{pct_splice:.2f}"
        print(f"\n--- Missingness {label} ---", flush=True)

        print("  * loading fresh MuData…", flush=True)
        mdata = mu.read_h5mu(MUDATA_PATH)

        # ─────────── subsample 5% of cells to avoid OOM ───────────
        n_cells = mdata.n_obs
        n_sub   = max(1, int(0.5 * n_cells))          # 5% of the cells
        rng     = np.random.default_rng(SEED)
        sub_idx = rng.choice(n_cells, size=n_sub, replace=False)
        # this slices _both_ modalities down to the same obs:
        mdata   = mdata[sub_idx, :]
        print(f"  → working on a {n_sub}/{n_cells} (~{n_sub/n_cells:.1%}) subset of cells")
        # ─────────────────────────────────────────────────────────────

        print("  * loaded MuData", flush=True)

        orig = corrupt_mudata_inplace(mdata, pct_rna, pct_splice, SEED)

        models = define_models()
        for name, setup_fn in models.items():
            print(f"  * Training {name} …", flush=True)
            model = setup_fn(mdata)
            print("    - calling model.train()", flush=True)
            model.view_anndata_setup()
            model.train(max_epochs=5, batch_size = 256)
            print("    - training complete", flush=True)

            print("    - computing imputation…", flush=True)
            expr = model.get_normalized_expression(return_numpy=True)
            lib  = model.get_library_size_factors()['expression']

            print(lib[:, None])
            # compute
            imp_expr = expr 
            imp_spl  = model.get_normalized_splicing(return_numpy=True)

            # set numpy print options if you like
            np.set_printoptions(precision=3, suppress=True)

            # choose a small slice: first 5 cells × first 5 features
            n_cells = min(5, imp_expr.shape[0])
            n_feats = min(5, imp_expr.shape[1])

            print("Imputed expression (first cells × first genes):")
            print(imp_expr[:n_cells, :n_feats])

            print("\nImputed splicing (first cells × first junctions):")
            print(imp_spl[:n_cells, :n_feats])

            if orig['rna'] is not None:
                m_rna = evaluate_imputation(orig['rna'], imp_expr)
            else:
                m_rna = {'mse': np.nan, 'median_l1': np.nan, 'spearman': np.nan}
            if orig['splice'] is not None:
                m_spl = evaluate_imputation(orig['splice'], imp_spl)
            else:
                m_spl = {'mse': np.nan, 'median_l1': np.nan, 'spearman': np.nan}

            records.append({
                'model': name, 'pct_rna': pct_rna, 'pct_splice': pct_splice,
                **{f"rna_{k}":v for k,v in m_rna.items()},
                **{f"spl_{k}":v for k,v in m_spl.items()},
            })

            print("    - generating UMAP plot…", flush=True)
            lat = model.get_latent_representation()
            ad = sc.AnnData(lat)
            ad.obs = mdata['rna'].obs
            sc.pp.neighbors(ad, use_rep="X")
            sc.tl.umap(ad, min_dist=0.2)
            fig = sc.pl.umap(ad, color=UMAP_GROUP, show=False,
                             return_fig=True, title=f"{name} {label}")
            fig.savefig(os.path.join(
                FIG_DIR, f"{name}_umap_{label}.png"
            ), dpi=300, bbox_inches="tight")
            plt.close(fig)
            print("    - saved UMAP", flush=True)

            del model, lat, ad, fig
            gc.collect()
            print("    - cleaned up model objects", flush=True)

        del mdata, orig
        gc.collect()
        print("  * cleaned up MuData", flush=True)

    print("Writing CSV…", flush=True)
    df = pd.DataFrame.from_records(records)
    df.to_csv(CSV_OUT, index=False)
    print("Wrote metrics to", CSV_OUT, flush=True)


if __name__ == "__main__":
    main()
