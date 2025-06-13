#!/usr/bin/env python3
"""
imputation_eval_single.py

Single-condition imputation benchmark for parallel execution
"""

import os
import random
import gc
import numpy as np
import pandas as pd
import mudata as mu
import sys
import scvi
import scanpy as sc
from scipy import sparse
from scipy.stats import spearmanr, linregress, pearsonr
import matplotlib.pyplot as plt

# Get condition from environment
PCT_RNA = float(os.environ.get("PCT_RNA", "0.1"))
PCT_SPLICE = float(os.environ.get("PCT_SPLICE", "0.0"))
IMPUTATION_EVAL_OUTDIR = os.environ.get("IMPUTATION_EVAL_OUTDIR", "./imputation_eval_output")

# Setup directories
FIG_DIR = os.path.join(IMPUTATION_EVAL_OUTDIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
CSV_OUT = os.path.join(IMPUTATION_EVAL_OUTDIR, "imputation_results.csv")

# Constants
#MUDATA_PATH = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/052025/SUBSETTOP5CELLSTYPES_aligned__ge_splice_combined_20250513_035938_full_genes.h5mu" all genes
MUDATA_PATH = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/052025/SUBSETTOP5CELLSTYPES_aligned__ge_splice_combined_20250513_035938.h5mu" #top 5000
UMAP_GROUP = "broad_cell_type"
SEED = 42

print(f"→ Running imputation evaluation for RNA={PCT_RNA}, Splice={PCT_SPLICE}")
print(f"→ Output directory: {IMPUTATION_EVAL_OUTDIR}")

# Copy all functions from your original script exactly
def plot_real_vs_imputed(x, y, kind, max_points=20000, evaluation_info="", pct_rna=0.0, pct_splice=0.0):
    """
    Enhanced plotting with better statistics and information.
    """
    if len(x) > max_points:
        idx = np.random.choice(len(x), size=max_points, replace=False)
        x = x[idx]
        y = y[idx]
    
    # Calculate multiple correlation metrics
    slope, intercept, r_pearson, p_value, _ = linregress(x, y)
    r_spearman, p_spearman = spearmanr(x, y)
    
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, s=6, alpha=0.3, edgecolors='none', c='steelblue')

    # Best-fit line
    x0 = np.array([x.min(), x.max()])
    plt.plot(x0, intercept + slope * x0, color='red', linewidth=2)
    
    # Perfect correlation line for reference
    plt.plot(x0, x0, color='gray', linewidth=1, linestyle='--', alpha=0.7)

    # Increase font sizes for x and y tick labels
    plt.xlabel("True values (length-normalized)", fontsize=16)
    plt.ylabel("Imputed values", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Extract Z and likelihood from kind string
    title_parts = []
    if "_z" in kind and "_likelihood" in kind:
        # Extract Z value
        z_part = kind.split("_z")[1].split("_likelihood")[0]
        title_parts.append(f"Z={z_part}")
        # Extract likelihood
        likelihood_part = kind.split("_likelihood")[1]
        title_parts.append(f"Likelihood={likelihood_part}")
    
    # Add missing percentages
    missing_info = f"RNA missing: {pct_rna:.1%}, Splice missing: {pct_splice:.1%}"
    title_parts.append(missing_info)
    
    # Create informative title
    # base_kind = kind.split("_z")[0] if "_z" in kind else kind
    #plt.title(f"{base_kind.upper()} Imputation\n" + 
    #          f"{', '.join(title_parts)}\n" +
    #          f"Pearson r={r_pearson:.3f}, Spearman ρ={r_spearman:.3f}",
    #          fontsize=12)
    
    # plt.legend(fontsize=10)
    plt.grid(False)  # Remove grid
    
    # Add some statistics text with larger font
    stats_text = f"n={len(x):,}\nMSE={((y-x)**2).mean():.3f}\nMAE={np.abs(y-x).mean():.3f}"
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Make filename more descriptive
    safe_kind = kind.replace("(", "").replace(")", "").replace(" ", "_")
    missing_label = f"rna{pct_rna:.2f}_spl{pct_splice:.2f}"
    out_path = os.path.join(FIG_DIR, f"{safe_kind}_{missing_label}_real_vs_imputed.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def corrupt_mudata_inplace_improved(
    mdata: mu.MuData,
    pct_rna: float = 0.0,
    pct_splice: float = 0.0,
    seed: int | None = None,
    min_expression: float = 0.1,  # Minimum expression to consider for masking
    max_expression: float = 1000,  # Maximum expression to consider for masking
) -> dict:
    """
    Improved corruption with better selection criteria for Smart-seq2 data.
    """
    rng = np.random.default_rng(seed)
    orig = {'rna': None, 'splice': None}

    print(f"[corrupt] Starting improved corruption: pct_rna={pct_rna}, pct_splice={pct_splice}", flush=True)
    print(f"[corrupt] Expression range for masking: {min_expression} - {max_expression}", flush=True)

    # 1) RNA masking with improved selection
    if pct_rna > 0:
        print("  [corrupt] → RNA masking (length-normalized data)...", flush=True)
        X = mdata['rna'].layers['length_norm']
        arr = X.toarray() if sparse.isspmatrix(X) else X.copy()

        # Use pre-computed library sizes (sum of length-normalized counts)
        lib_sizes = mdata['rna'].obsm["X_library_size"].flatten()
        print(f"  [corrupt] Library sizes (sum of length-norm): min={lib_sizes.min():.1f}, "
              f"max={lib_sizes.max():.1f}, median={np.median(lib_sizes):.1f}")
        
        # Verify this matches our calculation (should be identical)
        calculated_lib_sizes = arr.sum(axis=1)
        print(f"  [corrupt] Verification - calculated sums match: {np.allclose(lib_sizes, calculated_lib_sizes)}")

        # Better selection: avoid very low and very high expression values
        # Focus on moderately expressed genes for more meaningful evaluation
        nz = np.argwhere((arr >= min_expression) & (arr <= max_expression))
        print(f"  [corrupt] Found {len(nz)} entries in expression range [{min_expression}, {max_expression}]")
        
        if len(nz) == 0:
            print("  [corrupt] Warning: No entries found in specified range, using all nonzero")
            nz = np.argwhere(arr > 0)
        
        n_remove = int(len(nz) * pct_rna)
        sel = rng.choice(len(nz), size=n_remove, replace=False)
        coords = nz[sel]
        values = arr[coords[:,0], coords[:,1]].copy()

        # Store additional information for evaluation
        cell_lib_sizes = lib_sizes[coords[:, 0]]

        # Zero out those entries
        arr[coords[:,0], coords[:,1]] = 0

        # Write back
        if sparse.isspmatrix(X):
            mdata['rna'].layers['length_norm'] = sparse.csr_matrix(arr)
        else:
            mdata['rna'].layers['length_norm'] = arr

        orig['rna'] = {
            'coords': coords, 
            'values': values,
            'lib_sizes': cell_lib_sizes,
            'median_lib_size': np.median(lib_sizes)
        }
        print(f"  [corrupt] ← RNA masking complete: masked {len(coords)} entries", flush=True)
        print(f"  [corrupt] Masked values range: {values.min():.3f} - {values.max():.3f}")

    # 2) Splicing masking (unchanged but with better logging)
    if pct_splice > 0:
        print("  [corrupt] → Splicing masking...", flush=True)
        sp_mod = 'splicing'
        atse = mdata[sp_mod].layers['cell_by_cluster_matrix']
        junc = mdata[sp_mod].layers['cell_by_junction_matrix']
        ratio = mdata[sp_mod].layers['junc_ratio']

        atse_arr = atse.toarray() if sparse.isspmatrix(atse) else atse.copy()
        junc_arr = junc.toarray() if sparse.isspmatrix(junc) else junc.copy()
        if sparse.isspmatrix(ratio):
            ratio_arr = ratio.toarray().astype(float)
        else:
            ratio_arr = np.array(ratio, copy=True, dtype=float)

        valid = np.argwhere(
            (atse_arr > 0) &
            (junc_arr >= 0) &
            (~np.isnan(ratio_arr))
        )
        print(f"  [corrupt] Found {len(valid)} valid splicing entries")
        
        n_remove = int(len(valid) * pct_splice)
        sel = rng.choice(len(valid), size=n_remove, replace=False)
        coords = valid[sel]

        orig_vals = np.vstack([
            atse_arr[coords[:,0], coords[:,1]],
            junc_arr[coords[:,0], coords[:,1]],
            ratio_arr[coords[:,0], coords[:,1]],
        ]).T

        # Zero out
        atse_arr[coords[:,0], coords[:,1]] = 0
        junc_arr[coords[:,0], coords[:,1]] = 0
        ratio_arr[coords[:,0], coords[:,1]] = 0

        # Write back
        mdata[sp_mod].layers['cell_by_cluster_matrix'] = (
            sparse.csr_matrix(atse_arr) if sparse.isspmatrix(atse) else atse_arr
        )
        mdata[sp_mod].layers['cell_by_junction_matrix'] = (
            sparse.csr_matrix(junc_arr) if sparse.isspmatrix(junc) else junc_arr
        )
        mdata[sp_mod].layers['junc_ratio'] = ratio_arr

        orig['splice'] = (coords, orig_vals)
        print(f"  [corrupt] ← Splicing masking complete: masked {len(coords)} entries", flush=True)

        # Rebuild psi_mask layer
        print("  [corrupt] → Rebuilding psi_mask layer...", flush=True)
        clu = mdata['splicing'].layers['cell_by_cluster_matrix']
        if sparse.isspmatrix(clu):
            psi = clu.copy()
            psi.data = np.ones_like(psi.data, dtype=np.uint8)
        else:
            arr = (clu > 0).astype(np.uint8)
            psi = sparse.csr_matrix(arr)
        mdata['splicing'].layers['psi_mask'] = psi

    print("[corrupt] Done improved corruption", flush=True)
    return orig


def evaluate_imputation_improved(original, imputed, Z=None, likelihood=None, target_lib_size=1e4, 
                               pct_rna=0.0, pct_splice=0.0):
    """
    Improved evaluation for length-normalized Smart-seq2 data.
    Provides multiple evaluation strategies.
    """
    print("    * evaluate_imputation_improved...", flush=True)
    
    if isinstance(original, dict):  # RNA data with improved structure
        coords = original['coords']
        true_vals = original['values']
        lib_sizes = original['lib_sizes'] 
        median_lib = original['median_lib_size']
        
        pred = imputed[coords[:,0], coords[:,1]]
        
        # Strategy 1: Direct comparison (both are length-normalized)
        # Scale true values to target library size for fair comparison
        true_scaled = true_vals * target_lib_size / median_lib
        
        print(f"      -> True values (scaled to {target_lib_size}): {true_scaled.min():.3f} - {true_scaled.max():.3f}")
        print(f"      -> Pred values: {pred.min():.3f} - {pred.max():.3f}")
        
        diff = pred - true_scaled
        x1, x2 = true_scaled, pred
        kind = "rna"
        evaluation_info = f"Length-normalized, scaled to {target_lib_size}"
        
    else:  # Splicing data (unchanged)
        coords, vals = original
        pred = imputed[coords[:,0], coords[:,1]]
        kind = "splice"
        evaluation_info = "Splicing ratios"
        
        if vals.ndim == 2:
            atse, true_j, true_r = vals.T
            imp_counts = pred * atse
            diff = imp_counts - true_j
            x1, x2 = true_r, pred
        else:
            true_c = vals
            diff = pred - true_c
            x1, x2 = true_c, pred

    # Append the current Z and likelihood used to kind 
    if Z is not None and likelihood is not None:
        kind = f"{kind}_z{Z}_likelihood{likelihood}"

    # Calculate comprehensive metrics
    res = {
        'mse': float((diff**2).mean()),
        'mae': float(np.abs(diff).mean()),
        'median_l1': float(np.median(np.abs(diff))),
        'spearman': float(spearmanr(x1, x2).correlation),
        'pearson': float(pearsonr(x1, x2)[0]),
        'r2': float(1 - np.sum((x1 - x2)**2) / np.sum((x1 - np.mean(x1))**2)),
    }
    
    plot_real_vs_imputed(x1, x2, kind, evaluation_info=evaluation_info, 
                        pct_rna=pct_rna, pct_splice=pct_splice)
    print(f"      -> eval results: {res}", flush=True)
    return res


def define_models():
    def build(mdata, latent_dim, distribution):
        scvi.model.MULTIVISPLICE.setup_mudata(
            mdata,
            size_factor_key="X_library_size",
            batch_key="dataset",
            rna_layer="length_norm",  # Using length_norm layer
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
        "Splice-VI(Binomial Z=40)": lambda md: build(md, 40, "binomial"),
        "Splice-VI(Binomial Z=30)": lambda md: build(md, 30, "binomial"),
        "Splice-VI(Beta-Binomial Z=20)": lambda md: build(md, 20, "beta_binomial"),
        "Splice-VI(Beta-Binomial Z=40)": lambda md: build(md, 40, "beta_binomial"),
        "Splice-VI(Beta-Binomial Z=30)": lambda md: build(md, 30, "beta_binomial"),
    }


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    
    # Single condition instead of loop
    label = f"r{PCT_RNA:.2f}_s{PCT_SPLICE:.2f}"
    print(f"\n--- Missingness {label} ---", flush=True)

    print("  * loading fresh MuData…", flush=True)
    mdata = mu.read_h5mu(MUDATA_PATH)

    # Subsample cells
    n_cells = mdata.n_obs
    n_sub = max(1, int(0.5 * n_cells))
    rng = np.random.default_rng(SEED)
    sub_idx = rng.choice(n_cells, size=n_sub, replace=False)
    mdata = mdata[sub_idx, :]
    print(f"  → working on a {n_sub}/{n_cells} (~{n_sub/n_cells:.1%}) subset of cells")

    # Print data statistics
    if 'length_norm' in mdata['rna'].layers:
        X = mdata['rna'].layers['length_norm']
        if sparse.isspmatrix(X):
            X_arr = X.toarray()
        else:
            X_arr = X
        print(f"  → Length-norm expression: {X_arr.min():.3f} - {X_arr.max():.3f}, "
              f"median={np.median(X_arr[X_arr > 0]):.3f}")

    # Corruption
    orig = corrupt_mudata_inplace_improved(mdata, PCT_RNA, PCT_SPLICE, SEED)

    # Train all models for this condition
    models = define_models()
    results = []
    
    for name, setup_fn in models.items():
        print(f"  * Training {name} …", flush=True)
        model = setup_fn(mdata)
        print("    - calling model.train()", flush=True)
        model.train(lr=0.00001, max_epochs=20, batch_size=256, 
                   n_epochs_kl_warmup=10, lr_scheduler_type="step", 
                   step_size=10, lr_factor=0.5)
        print("    - training complete", flush=True)

        print("    - computing imputation…", flush=True)
        # Get imputed gene expression - check available methods
        try:
            # Try the standard method without library_size parameter
            expr = model.get_normalized_expression(return_numpy=True)
            print(f"    - got normalized expression shape: {expr.shape}")
        except Exception as e:
            print(f"    - get_normalized_expression failed: {e}")
            # Alternative: try get_expression or other methods
            try:
                expr = model.get_expression(return_numpy=True)
                print(f"    - got expression shape: {expr.shape}")
            except Exception as e2:
                print(f"    - get_expression also failed: {e2}")
                # Last resort: use latent representation or skip
                print("    - Skipping expression imputation due to method errors")
                continue
        
        # Get imputed splicing
        try:
            imp_spl = model.get_normalized_splicing(return_numpy=True)
            print(f"    - got normalized splicing shape: {imp_spl.shape}")
        except Exception as e:
            print(f"    - get_normalized_splicing failed: {e}")
            try:
                imp_spl = model.get_splicing(return_numpy=True)
                print(f"    - got splicing shape: {imp_spl.shape}")
            except Exception as e2:
                print(f"    - get_splicing also failed: {e2}")
                imp_spl = None

        # Check what we got and manually scale to match expected range
        if expr is not None:
            print(f"    - Raw imputed expression range: {expr.min():.3f} - {expr.max():.3f}")
            print(f"    - Raw imputed expression median (nonzero): {np.median(expr[expr > 0]):.3f}")
            
            # Get the original data scale for comparison
            X_orig = mdata['rna'].layers['length_norm']
            if sparse.isspmatrix(X_orig):
                X_orig_arr = X_orig.toarray()
            else:
                X_orig_arr = X_orig
            print(f"    - Original length-norm range: {X_orig_arr.min():.3f} - {X_orig_arr.max():.3f}")
            print(f"    - Original length-norm median (nonzero): {np.median(X_orig_arr[X_orig_arr > 0]):.3f}")
            
            # Manual scaling using library sizes from obsm
            try:
                lib_sizes = mdata['rna'].obsm["X_library_size"].flatten()
                median_lib_size = np.median(lib_sizes)
                target_lib_size = median_lib_size  # Use the actual median as target, not 1e4
                
                print(f"    - Library size stats: min={lib_sizes.min():.0f}, max={lib_sizes.max():.0f}, median={median_lib_size:.0f}")
                print(f"    - This is normal for Smart-seq2 data across {X_orig_arr.shape[1]} genes")
                
                # The model likely returns expression already normalized by library size
                # So we need to scale back up to the original scale
                original_expr = expr.copy()
                
                # First, try scaling by median library size
                expr = expr * median_lib_size
                print(f"    - Scaled by median library size: {median_lib_size:.0f}")
                print(f"    - After lib scaling: {expr.min():.3f} - {expr.max():.3f}")
                
                # Check if this gives us reasonable values compared to original data
                original_p95 = np.percentile(X_orig_arr[X_orig_arr > 0], 95)
                current_p95 = np.percentile(expr[expr > 0], 95) if np.any(expr > 0) else 0
                
                print(f"    - 95th percentiles: original={original_p95:.1f}, current={current_p95:.1f}")
                
                # If still very different, apply additional correction
                if current_p95 > 0 and (current_p95 < 0.1 * original_p95 or current_p95 > 10 * original_p95):
                    correction_factor = original_p95 / current_p95
                    expr = expr * correction_factor
                    print(f"    - Applied correction factor: {correction_factor:.3f}")
                    print(f"    - Final range: {expr.min():.3f} - {expr.max():.3f}")
                    
                    final_p95 = np.percentile(expr[expr > 0], 95) if np.any(expr > 0) else 0
                    print(f"    - Final 95th percentile: {final_p95:.1f}")
                else:
                    print(f"    - Scaling looks good, keeping current values")
                    
            except Exception as e:
                print(f"    - Could not scale using library sizes: {e}")
                print(f"    - Using expression as-is")

        # Extract Z and likelihood from model name for evaluation
        Z = None
        likelihood = None
        if "Z=" in name:
            try:
                Z = int(name.split("Z=")[1].split(")")[0])
            except:
                Z = None
        if "Binomial" in name:
            if "Beta-Binomial" in name:
                likelihood = "beta_binomial"
            else:
                likelihood = "binomial"

        # Evaluation and result collection
        if orig['rna'] is not None and expr is not None:
            actual_target = np.median(mdata['rna'].obsm["X_library_size"].flatten())
            m_rna = evaluate_imputation_improved(orig['rna'], expr, Z=Z, likelihood=likelihood,
                                               target_lib_size=actual_target, pct_rna=PCT_RNA, pct_splice=PCT_SPLICE)
        else:
            m_rna = {k: np.nan for k in ['mse', 'mae', 'median_l1', 'spearman', 'pearson', 'r2']}
        
        if orig['splice'] is not None and imp_spl is not None:
            m_spl = evaluate_imputation_improved(orig['splice'], imp_spl, Z=Z, likelihood=likelihood,
                                               pct_rna=PCT_RNA, pct_splice=PCT_SPLICE)
        else:
            m_spl = {k: np.nan for k in ['mse', 'mae', 'median_l1', 'spearman', 'pearson', 'r2']}

        record = {
            'model': name, 'pct_rna': PCT_RNA, 'pct_splice': PCT_SPLICE,
            **{f"rna_{k}": v for k, v in m_rna.items()},
            **{f"spl_{k}": v for k, v in m_spl.items()},
        }
        results.append(record)

        # UMAP
        print("    - generating UMAP plot…", flush=True)
        lat = model.get_latent_representation()
        ad = sc.AnnData(lat)
        ad.obs = mdata['rna'].obs
        sc.pp.neighbors(ad, use_rep="X")
        sc.tl.umap(ad, min_dist=0.2)
        fig = sc.pl.umap(ad, color=UMAP_GROUP, show=False,
                         return_fig=True, title=f"{name} {label}")
        fig.savefig(os.path.join(FIG_DIR, f"{name}_umap_{label}.png"), 
                   dpi=300, bbox_inches="tight")
        plt.close(fig)

        del model, lat, ad, fig
        gc.collect()
        print("    - cleaned up model objects", flush=True)

    # Save results for this condition
    df = pd.DataFrame(results)
    df.to_csv(CSV_OUT, index=False)
    print(f"→ Results saved to: {CSV_OUT}")

    del mdata, orig
    gc.collect()
    print("  * cleaned up MuData", flush=True)

if __name__ == "__main__":
    main()