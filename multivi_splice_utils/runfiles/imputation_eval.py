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
from scipy.stats import spearmanr, linregress
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
MISSING_PCT_PAIRS = [(0.8, 0.0), (0.2, 0.2), (0.5, 0.5), (0.8, 0.8)]
SEED = 42

# ------------------------------------------------------------------------------
# Utilities: corrupt & evaluate
# ------------------------------------------------------------------------------
import numpy as np
from scipy import sparse

import numpy as np
import mudata as mu
from scipy import sparse

import numpy as np
import mudata as mu
from scipy import sparse

def plot_real_vs_imputed(x, y, kind, max_points=10000):
    """
    x, y: 1D arrays of true vs imputed values
    kind: 'rna' or 'splice' (used for filename and title)
    """

    if len(x) > max_points:
        idx = np.random.choice(len(x), size=max_points, replace=False)
        x = x[idx]
        y = y[idx]
    # linear fit
    slope, intercept, r_value, p_value, _ = linregress(x, y)

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=5, alpha=0.2, edgecolors='none')  # transparent small points

    # best-fit line over the data range
    x0 = np.array([x.min(), x.max()])
    plt.plot(x0, intercept + slope * x0, color='red', linewidth=2.5, zorder=10)

    plt.xlabel("True values")
    plt.ylabel("Imputed values")
    plt.title(f"{kind.upper()} real vs imputed\nR={r_value:.2f}, p={p_value:.2e}")
    out_path = os.path.join(FIG_DIR, f"{kind}_real_vs_imputed.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def corrupt_mudata_inplace(
    mdata: mu.MuData,
    pct_rna: float = 0.0,
    pct_splice: float = 0.0,
    seed: int | None = None,
) -> dict:
    """
    Zero out a fraction of RNA and/or splicing data directly in `mdata`.
    Returns:
      - orig: {
          'rna': (coords, values),
          'splice': (coords, [[atse, junc, ratio], ...])
        }
    """
    rng  = np.random.default_rng(seed)
    orig = {'rna': None, 'splice': None}

    print(f"[corrupt] Starting in-place corruption: pct_rna={pct_rna}, pct_splice={pct_splice}", flush=True)

    # 1) RNA masking in place
    if pct_rna > 0:
        print("  [corrupt] → RNA masking...", flush=True)
        X   = mdata['rna'].layers['raw_counts']
        arr = X.toarray() if sparse.isspmatrix(X) else X.copy()

        # find nonzero entries and pick a subset that have less than 10,000 reads
        nz = np.argwhere((arr != 0) & (arr < 10000))
        n_remove = int(len(nz) * pct_rna)
        sel      = rng.choice(len(nz), size=n_remove, replace=False)
        coords   = nz[sel]                       
        values   = arr[coords[:,0], coords[:,1]].copy()

        # zero out those entries
        arr[coords[:,0], coords[:,1]] = 0

        # write back (preserve sparse if it was)
        if sparse.isspmatrix(X):
            mdata['rna'].layers['raw_counts'] = sparse.csr_matrix(arr)
        else:
            mdata['rna'].layers['raw_counts'] = arr

        orig['rna'] = (coords, values)
        print(f"  [corrupt] ← RNA masking complete: masked {len(coords)} entries", flush=True)

    else:
        print("  [corrupt] → Skipping RNA masking (pct_rna=0)", flush=True)

    # 2) Splicing masking in place
    if pct_splice > 0:
        print("  [corrupt] → Splicing masking...", flush=True)
        sp_mod = 'splicing'
        atse   = mdata[sp_mod].layers['cell_by_cluster_matrix']
        junc   = mdata[sp_mod].layers['cell_by_junction_matrix']
        ratio  = mdata[sp_mod].layers['junc_ratio']

        atse_arr = atse.toarray() if sparse.isspmatrix(atse) else atse.copy()
        junc_arr = junc.toarray() if sparse.isspmatrix(junc) else junc.copy()
        if sparse.isspmatrix(ratio):
            ratio_arr = ratio.toarray().astype(float)
        else:
            ratio_arr = np.array(ratio, copy=True, dtype=float)

        valid    = np.argwhere(
            (atse_arr > 0) &
            (junc_arr >= 0) &
            (~np.isnan(ratio_arr))
        )
        n_remove = int(len(valid) * pct_splice)
        sel      = rng.choice(len(valid), size=n_remove, replace=False)
        coords   = valid[sel]                       

        orig_vals = np.vstack([
            atse_arr[coords[:,0], coords[:,1]],
            junc_arr[coords[:,0], coords[:,1]],
            ratio_arr[coords[:,0], coords[:,1]],
        ]).T

        # zero out
        atse_arr[coords[:,0], coords[:,1]] = 0
        junc_arr[coords[:,0], coords[:,1]] = 0
        ratio_arr[coords[:,0], coords[:,1]] = 0

        # write back
        mdata[sp_mod].layers['cell_by_cluster_matrix']   = (
            sparse.csr_matrix(atse_arr) if sparse.isspmatrix(atse) else atse_arr
        )
        mdata[sp_mod].layers['cell_by_junction_matrix']  = (
            sparse.csr_matrix(junc_arr) if sparse.isspmatrix(junc) else junc_arr
        )
        mdata[sp_mod].layers['junc_ratio']               = ratio_arr

        orig['splice'] = (coords, orig_vals)
        print(f"  [corrupt] ← Splicing masking complete: masked {len(coords)} entries", flush=True)

        # 3) Rebuild psi_mask layer
        print("  [corrupt] → Rebuilding psi_mask layer...", flush=True)
        clu = mdata['splicing'].layers['cell_by_cluster_matrix']
        if sparse.isspmatrix(clu):
            psi = clu.copy()  
            psi.data = np.ones_like(psi.data, dtype=np.uint8)  
        else:  
            arr = (clu > 0).astype(np.uint8)  
            psi = sparse.csr_matrix(arr)  
        mdata['splicing'].layers['psi_mask'] = psi
        print("  [corrupt] ← psi_mask rebuild complete", flush=True)

    else:
        print("  [corrupt] → Skipping splicing masking (pct_splice=0)", flush=True)
        print("  [corrupt] → No need to rebuild psi mask.", flush=True)


    print("[corrupt] Done in-place corruption", flush=True)
    return orig



def evaluate_imputation(original, imputed):
    print("    * evaluate_imputation...", flush=True)
    coords, vals = original
    pred = imputed[coords[:,0], coords[:,1]]
    kind = "rna"
    if vals.ndim == 2:
        kind = "splice"
        atse, true_j, true_r = vals.T
        imp_counts = pred * atse
        diff = imp_counts - true_j
        #x1, x2 = true_j, imp_counts
        x1, x2 = true_r, pred
    else:
        true_c = vals
        diff = pred - true_c
        x1, x2 = true_c, pred
    res = {
        'mse': float((diff**2).mean()),
        'median_l1': float(np.median(np.abs(diff))),
        'spearman': float(spearmanr(x1, x2).correlation),
    }
    plot_real_vs_imputed(x1, x2, kind)
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
            batch_key="dataset",
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
            expression_architecture = "linear",
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
            model.train(max_epochs=20, batch_size = 256, n_epochs_kl_warmup = 10, lr_scheduler_type="step", lr_scheduler_step = 5, lr_factor = 0.5)
            print("    - training complete", flush=True)

            print("    - computing imputation…", flush=True)
            expr = model.get_normalized_expression(return_numpy=True)
            lib  = model.get_library_size_factors()['expression']

            print(lib[:, None])
            # compute
            imp_counts = expr * lib[:, None]
            imp_spl  = model.get_normalized_splicing(return_numpy=True)

            # set numpy print options if you like
            np.set_printoptions(precision=3, suppress=True)

            # choose a small slice: first 5 cells × first 5 features
            n_cells = min(5, imp_counts.shape[0])
            n_feats = min(5, imp_counts.shape[1])

            print("Imputed expression (first cells × first genes):")
            print(imp_counts[:n_cells, :n_feats])

            print("\nImputed splicing (first cells × first junctions):")
            print(imp_spl[:n_cells, :n_feats])

            if orig['rna'] is not None:
                m_rna = evaluate_imputation(orig['rna'], imp_counts)
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
