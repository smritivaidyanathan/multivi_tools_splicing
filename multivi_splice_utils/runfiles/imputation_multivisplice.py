#!/usr/bin/env python3
"""
imputation_multivisplice.py

Evaluate cross-modality imputation performance using a pre-trained MULTIVISPLICE model.

Usage:
  python imputation_eval.py \
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

# Zeroing functions

def zero_out_splicing(mdata):
    sp = mdata['splicing']
    for layer in ['cell_by_cluster_matrix','cell_by_junction_matrix','junc_ratio','psi_mask']:
        X = sp.layers.get(layer)
        if X is None: continue
        if sparse.isspmatrix(X):
            sp.layers[layer] = sparse.csr_matrix(X.shape)
        else:
            sp.layers[layer] = np.zeros_like(X)


def zero_out_expression(mdata):
    ge = mdata['rna']
    X = ge.layers['length_norm']
    if sparse.isspmatrix(X):
        ge.layers['length_norm'] = sparse.csr_matrix(X.shape)
    else:
        ge.layers['length_norm'] = np.zeros_like(X)

# Metrics
def evaluate(true, pred):
    mask = ~np.isnan(true)
    t = true[mask].ravel()
    p = pred[mask].ravel()
    mse = mean_squared_error(t,p)
    mae = mean_absolute_error(t,p)
    r, _ = pearsonr(t,p)
    rho, _ = spearmanr(t,p)
    ss_res = ((t-p)**2).sum()
    ss_tot = ((t - t.mean())**2).sum()
    r2 = 1 - ss_res/ss_tot
    return dict(mse=mse,mae=mae,pearson=r,spearman=rho,r2=r2)

# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mudata_path', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    df_records = []

    # Load test MuData
    test_mdata = mu.read_h5mu(args.test_mudata_path)
    # Setup modalities as in training
    scvi.model.MULTIVISPLICE.setup_mudata(
        test_mdata,
        batch_key='dataset', size_factor_key='X_library_size',
        rna_layer='length_norm', junc_ratio_layer='junc_ratio',
        atse_counts_layer='cell_by_cluster_matrix',
        junc_counts_layer='cell_by_junction_matrix',
        psi_mask_layer='psi_mask',
        modalities={'rna_layer':'rna','junc_ratio_layer':'splicing'}
    )
    # Load model
    model = scvi.model.MULTIVISPLICE.load(args.model_path, adata=test_mdata)

    # 1) Expression-only → impute splicing
    m_expr = test_mdata.copy()
    zero_out_splicing(m_expr)
    imp_spl = model.get_normalized_splicing(adata=m_expr, return_numpy=True)
    gt_spl = test_mdata['splicing'].layers['junc_ratio']
    if sparse.isspmatrix(gt_spl): gt_spl = gt_spl.toarray()
    rec = evaluate(gt_spl, imp_spl)
    rec['mode'] = 'expr_only'
    df_records.append(rec)

    # 2) Splicing-only → impute expression
    m_spl = test_mdata.copy()
    zero_out_expression(m_spl)
    imp_expr = model.get_normalized_expression(adata=m_spl, return_numpy=True)
    gt_expr = test_mdata['rna'].layers['length_norm']
    if sparse.isspmatrix(gt_expr): gt_expr = gt_expr.toarray()
    rec = evaluate(gt_expr, imp_expr)
    rec['mode'] = 'spl_only'
    df_records.append(rec)

    # Save metrics
    out_csv = os.path.join(args.out_dir,'imputation_metrics.csv')
    pd.DataFrame(df_records).to_csv(out_csv,index=False)
    print(f"Saved imputation metrics to {out_csv}")

if __name__=='__main__': main()