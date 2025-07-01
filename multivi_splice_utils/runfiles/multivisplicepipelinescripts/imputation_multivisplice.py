#!/usr/bin/env python3
"""
imputation_multivisplice.py

Evaluate cross‐modality imputation performance using a pre‐trained MULTIVISPLICE model.

Usage:
  python imputation_multivisplice.py \
    --test_mudata_path <test.h5mu> \
    --model_path <model_dir> \
    --out_dir <output_folder>
"""
import os
import argparse

import numpy as np
import pandas as pd
import mudata as mu
import scvi
from scipy import sparse
from scipy.stats import mean_squared_error, mean_absolute_error, pearsonr, spearmanr


def zero_out_splicing(mdata):
    sp = mdata["splicing"]
    for layer in (
        "cell_by_cluster_matrix",
        "cell_by_junction_matrix",
        "junc_ratio",
        "psi_mask",
    ):
        X = sp.layers.get(layer)
        if X is None:
            continue
        if sparse.isspmatrix(X):
            sp.layers[layer] = sparse.csr_matrix(X.shape)
        else:
            sp.layers[layer] = np.zeros_like(X)


def zero_out_expression(mdata):
    ge = mdata["rna"]
    X = ge.layers.get("length_norm")
    if X is None:
        return
    if sparse.isspmatrix(X):
        ge.layers["length_norm"] = sparse.csr_matrix(X.shape)
    else:
        ge.layers["length_norm"] = np.zeros_like(X)


def compute_metrics(true, pred):
    mask = ~np.isnan(true)
    t = true[mask].ravel()
    p = pred[mask].ravel()
    mse = mean_squared_error(t, p)
    mae = mean_absolute_error(t, p)
    r, _ = pearsonr(t, p)
    rho, _ = spearmanr(t, p)
    ss_res = ((t - p) ** 2).sum()
    ss_tot = ((t - t.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    return dict(mse=mse, mae=mae, pearson=r, spearman=rho, r2=r2)


def parse_args():
    p = argparse.ArgumentParser(
        description="Cross‐modality imputation eval for MULTIVISPLICE"
    )
    p.add_argument(
        "--test_mudata_path", type=str, required=True, help="Path to test .h5mu"
    )
    p.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Directory where the trained model is saved",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Base output folder; will create figures/ and csv_files/",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Create output folders
    fig_dir = os.path.join(args.out_dir, "figures")
    csv_dir = os.path.join(args.out_dir, "csv_files")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    print(f"[INFO] Loading test MuData from {args.test_mudata_path}")
    test_mdata = mu.read_h5mu(args.test_mudata_path)

    print("[INFO] Setting up MuData for MULTIVISPLICE")
    scvi.model.MULTIVISPLICE.setup_mudata(
        test_mdata,
        batch_key="dataset",
        size_factor_key="X_library_size",
        rna_layer="length_norm",
        junc_ratio_layer="junc_ratio",
        atse_counts_layer="cell_by_cluster_matrix",
        junc_counts_layer="cell_by_junction_matrix",
        psi_mask_layer="psi_mask",
        modalities={"rna_layer": "rna", "junc_ratio_layer": "splicing"},
    )

    print(f"[INFO] Loading trained model from {args.model_path}")
    model = scvi.model.MULTIVISPLICE.load(args.model_path, adata=test_mdata)

    records = []

    # 1) Expression-only → impute splicing
    print("[INFO] Imputing splicing from expression only")
    m_expr = test_mdata.copy()
    zero_out_splicing(m_expr)
    imp_spl = model.get_normalized_splicing(adata=m_expr, return_numpy=True)
    gt_spl = test_mdata["splicing"].layers["junc_ratio"]
    if sparse.isspmatrix(gt_spl):
        gt_spl = gt_spl.toarray()
    rec = compute_metrics(gt_spl, imp_spl)
    rec["mode"] = "expr_only"
    records.append(rec)

    # 2) Splicing-only → impute expression
    print("[INFO] Imputing expression from splicing only")
    m_spl = test_mdata.copy()
    zero_out_expression(m_spl)
    imp_expr = model.get_normalized_expression(adata=m_spl, return_numpy=True)
    gt_expr = test_mdata["rna"].layers["length_norm"]
    if sparse.isspmatrix(gt_expr):
        gt_expr = gt_expr.toarray()
    rec = compute_metrics(gt_expr, imp_expr)
    rec["mode"] = "spl_only"
    records.append(rec)

    # Save metrics
    out_csv = os.path.join(csv_dir, "imputation_metrics.csv")
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"[INFO] Saved imputation metrics to {out_csv}")

    print("[INFO] Imputation evaluation complete.")


if __name__ == "__main__":
    main()
