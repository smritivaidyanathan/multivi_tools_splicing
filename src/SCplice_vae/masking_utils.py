"""
Masking and missing data utilities for SCpliceVAE.

This module provides functions for handling masked data and missing values in AnnData objects
for the SCpliceVAE model.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, issparse
import torch
import matplotlib.pyplot as plt
import os 

# ------------------------------
# Reconstruction Masking Functions
# ------------------------------

def generate_recon_mask(adata, layer_key="junc_ratio", mask_percentage=0.1, seed=42, randomize_seed=False):
    """
    Generate a boolean mask for the specified layer in an AnnData object.
    Only non-NaN entries are considered for masking.
    
    Parameters:
      adata: AnnData object.
      layer_key: the key of the layer to use (default "junc_ratio").
      mask_percentage: fraction of valid (non-NaN) entries to mask.
      seed: integer seed for reproducibility.
      randomize_seed: if True, a random seed is used.
      
    Returns:
      artificial_mask: Boolean numpy array of same shape as the layer.
      seed: the seed used.
    """
    if randomize_seed:
        seed = np.random.randint(0, 1000000)
    np.random.seed(seed)
    
    # Get the layer data (dense version)
    layer_data = adata.layers[layer_key]
    if sp.issparse(layer_data):
        layer_data = layer_data.toarray()
    else:
        layer_data = layer_data.copy()
    
    # Identify valid (non-NaN) entries
    valid_mask = ~np.isnan(layer_data)
    valid_indices = np.array(np.where(valid_mask)).T  #each row is [i, j]
    num_valid = valid_indices.shape[0]
    
    num_to_mask = int(num_valid * mask_percentage)
    if num_to_mask >= num_valid:
        raise ValueError("mask_percentage is too high; not enough valid entries remain.")
    
    # Randomly select indices to mask
    chosen_indices = valid_indices[np.random.choice(num_valid, num_to_mask, replace=False)]
    
    # Create an empty mask and mark the chosen indices as True
    artificial_mask = np.zeros_like(layer_data, dtype=bool)
    for idx in chosen_indices:
        artificial_mask[idx[0], idx[1]] = True
    
    print(f"Total valid entries: {num_valid}, entries to mask: {np.sum(artificial_mask)}")
    return artificial_mask, seed

def apply_recon_mask_to_anndata(adata, artificial_mask, layer_key="junc_ratio"):
    """
    Apply the reconstruction mask to the specified layer in an AnnData object.
    For each masked (held-out) entry in the junc_ratio layer, set its value to NaN.
    Also, update corresponding entries in cell_by_junction_matrix and cell_by_cluster_matrix to 0.
    
    Parameters:
      adata: AnnData object.
      artificial_mask: Boolean numpy array (same shape as the junc_ratio layer) indicating which entries to mask.
      layer_key: the key of the layer to mask (default "junc_ratio").
      
    Returns:
      adata: the modified AnnData object.
    """
    # Backup original junc_ratio if not already stored.
    backup_key = "original_" + layer_key
    if backup_key not in adata.layers:
        layer_data = adata.layers[layer_key]
        if sp.issparse(layer_data):
            layer_data = layer_data.toarray()
        else:
            layer_data = layer_data.copy()
        adata.layers[backup_key] = layer_data.copy()
        print(f"Backup of {layer_key} stored as {backup_key}.")

    # Update junc_ratio layer: set masked entries to NaN.
    layer_data = adata.layers[layer_key]
    if sp.issparse(layer_data):
        layer_data = layer_data.toarray()
    else:
        layer_data = layer_data.copy()
    
    layer_data[artificial_mask] = np.nan
    if sp.issparse(adata.layers[layer_key]):
        adata.layers[layer_key] = sp.csr_matrix(layer_data)
    else:
        adata.layers[layer_key] = layer_data
    
    # For cell_by_junction_matrix: set corresponding entries to 0.
    cj_key = "cell_by_junction_matrix"
    cj_data = adata.layers[cj_key]
    if sp.issparse(cj_data):
        cj_data = cj_data.toarray()
    cj_data[artificial_mask] = 0
    adata.layers[cj_key] = sp.csr_matrix(cj_data) if sp.issparse(adata.layers[cj_key]) else cj_data
    
    # For cell_by_cluster_matrix: set corresponding entries to 0.
    cc_key = "cell_by_cluster_matrix"
    cc_data = adata.layers[cc_key]
    if sp.issparse(cc_data):
        cc_data = cc_data.toarray()
    cc_data[artificial_mask] = 0
    adata.layers[cc_key] = sp.csr_matrix(cc_data) if sp.issparse(adata.layers[cc_key]) else cc_data
    
    print("Artificial mask applied: junc_ratio entries set to NaN and corresponding count matrices updated to 0.")
    return adata

def compute_reconstruction_accuracy(model, adata, mask, layer_key="junc_ratio", metric="MAE", plot_dir=None):
    """
    Compute the reconstruction error for held-out entries and Spearman correlation.
    
    Parameters:
    - model: the trained VAE model.
    - adata: the AnnData object containing the data.
    - mask: a boolean mask indicating held-out (corrupted) entries.
    - layer_key: the key in adata.layers that is used for reconstruction.
    - metric: which error metric to use. Options are:
        "MAE" (mean absolute error, default),
        "MSE" (mean squared error),
        "median_L1" (median absolute error).
    - plot_dir: directory to save the plot (if None, plot is not saved).
    
    Returns:
    - results: dictionary containing error metrics and correlation coefficients
    - fig: matplotlib figure object of the scatter plot (or None if not plotted)
    """
    import scipy.stats as stats # type: ignore
    
    # Get the original data from the backup layer
    original_data = adata.layers["original_" + layer_key]
    if sp.issparse(original_data):
        original_data = original_data.toarray()
    
    # Get the corrupted input data (that was used during training)
    input_data = adata.layers[layer_key]
    if sp.issparse(input_data):
        input_data = input_data.toarray()
    
    # Convert input data to tensor and move to the appropriate device
    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(next(model.parameters()).device)
    
    # Obtain model reconstruction
    model.eval()
    with torch.no_grad():
        reconstruction, _, _ = model(input_tensor)
    reconstruction = reconstruction.detach().cpu().numpy()
    
    # Apply sigmoid to get probabilities if using logits
    reconstruction_probs = 1/(1 + np.exp(-reconstruction))
    
    # Get original and reconstructed values for held-out entries
    original_masked = original_data[mask]
    recon_masked = reconstruction_probs[mask]
    
    # Compute error metrics
    if metric == "MSE":
        error = np.mean((original_masked - recon_masked) ** 2)
    elif metric == "median_L1":
        error = np.median(np.abs(original_masked - recon_masked))
    else:  # default to MAE (mean absolute error)
        error = np.mean(np.abs(original_masked - recon_masked))
    
    # Compute Spearman correlation
    spearman_corr, spearman_pval = stats.spearmanr(original_masked, recon_masked)
    
    # Compute Pearson correlation
    pearson_corr, pearson_pval = stats.pearsonr(original_masked, recon_masked)
    
    # Create a scatter plot of original vs. imputed values
    fig = None
    if plot_dir is not None:
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create scatter plot with density coloring
        counts, xedges, yedges, im = ax.hist2d(
            original_masked, recon_masked, 
            bins=50, cmap='viridis', alpha=0.75,
            range=[[0, 1], [0, 1]]
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Density')
        
        # Add identity line
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.7)
        
        # Add correlation text
        textstr = f'Spearman ρ = {spearman_corr:.3f} (p = {spearman_pval:.1e})\n'
        textstr += f'Pearson r = {pearson_corr:.3f} (p = {pearson_pval:.1e})\n'
        textstr += f'{metric} = {error:.3f}'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Set labels and title
        ax.set_xlabel('Original Values')
        ax.set_ylabel('Imputed Values')
        ax.set_title(f'Original vs. Imputed Values for Held-out Entries\n{layer_key}', fontsize=14)
        
        # Make plot square and set equal aspect ratio
        ax.set_aspect('equal')
        
        # Save plot if directory is provided
        if plot_dir:
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, f"imputation_scatter_{layer_key}_{metric}.png")
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            print(f"Scatter plot saved to: {plot_path}")
    
    # Create results dictionary
    results = {
        "error": error,
        "error_metric": metric,
        "spearman_correlation": spearman_corr,
        "spearman_pvalue": spearman_pval,
        "pearson_correlation": pearson_corr,
        "pearson_pvalue": pearson_pval,
        "num_masked_entries": len(original_masked)
    }
    
    # Print summary
    print(f"Reconstruction metrics on {len(original_masked)} held-out entries:")
    print(f"  {metric}: {error:.4f}")
    print(f"  Spearman correlation: {spearman_corr:.4f} (p = {spearman_pval:.2e})")
    print(f"  Pearson correlation: {pearson_corr:.4f} (p = {pearson_pval:.2e})")
    
    return results, fig

def handle_missing_data(adata, method="MaskOut", layer_key="junc_ratio"):
    """
    Handle missing data (NaNs) in the specified layer based on the method provided.
    
    Parameters:
      adata: AnnData object.
      method: One of "ZeroOut", "MaskOut", "LeafletFA".
              - "ZeroOut": fill NaNs with 0 and do NOT create a mask.
              - "MaskOut": fill NaNs with 0 and create a mask to exclude from loss.
              - "LeafletFA": use 'imputed_PSI' layer instead of 'junc_ratio'.
      layer_key: the key of the layer to handle missing data for.
      
    Returns:
      adata: the modified AnnData object.
    """
    if method == "ZeroOut":
        print(f"Filling NaNs with 0 in {layer_key}, no mask used.")
        layer = adata.layers[layer_key]
        if issparse(layer):
            dense_layer = layer.toarray()
        else:
            dense_layer = layer
        # Replace NaNs with 0
        dense_layer = np.nan_to_num(dense_layer, nan=0.0)
        # Overwrite layer
        adata.layers[layer_key] = csr_matrix(dense_layer) if issparse(layer) else dense_layer

    elif method == "MaskOut":
        print(f"MaskOut mode: fill {layer_key} with 0 for the encoder, but also store a mask to exclude from loss.")
        layer = adata.layers[layer_key]
        if issparse(layer):
            dense_layer = layer.toarray()
        else:
            dense_layer = layer

        # Create a mask: True if not NaN
        mask_key = f"{layer_key}_NaN_mask"
        junc_ratio_mask = ~np.isnan(dense_layer)
        # Fill everything else with 0
        dense_layer_filled = np.nan_to_num(dense_layer, nan=0.0)

        # Overwrite layer with the zero-filled version
        adata.layers[layer_key] = csr_matrix(dense_layer_filled) if issparse(layer) else dense_layer_filled
        # Also store the mask
        adata.layers[mask_key] = junc_ratio_mask

    elif method == "LeafletFA":
        print("Using 'imputed_PSI' layer instead of 'junc_ratio', no zero fill, no mask.")
        if 'imputed_PSI' not in adata.layers:
            raise ValueError("No 'imputed_PSI' layer found in AnnData. Please provide it or switch method.")
        adata.layers[layer_key] = adata.layers['imputed_PSI']
        
    else:
        raise ValueError("Method must be one of 'ZeroOut', 'MaskOut', 'LeafletFA'")
    
    return adata

def sanity_check_recon_mask(adata, artificial_mask, count_layer="Cluster_Counts"):
    """
    Verify that all True values in the reconstruction mask correspond to 
    non-zero values in the original count layer.
    
    Parameters:
        adata: Original AnnData object (before masking)
        artificial_mask: Boolean mask used for reconstruction
        count_layer: Name of the layer containing count data
        
    Returns:
        is_valid: Boolean indicating if the check passed
        mismatch_count: Number of masked positions with zero counts
    """
    # Get the count data
    count_data = adata.layers[count_layer]
    if sp.issparse(count_data):
        count_data = count_data.toarray()
    
    # Find positions that are True in artificial_mask
    masked_positions = artificial_mask
    
    # Check if these positions have non-zero counts in the original data
    zero_count_positions = count_data == 0
    
    # Find cases where we masked a position that had zero counts
    invalid_masks = np.logical_and(masked_positions, zero_count_positions)
    mismatch_count = np.sum(invalid_masks)
    
    # Print results
    total_masked = np.sum(masked_positions)
    if mismatch_count == 0:
        print(f"✓ All {total_masked} masked positions had non-zero counts in {count_layer}")
        is_valid = True
    else:
        print(f"✗ Found {mismatch_count} out of {total_masked} masked positions with zero counts in {count_layer}")
        is_valid = False
        
        # Optional: Calculate percentage
        percentage = (mismatch_count / total_masked) * 100
        print(f"  {percentage:.2f}% of masked positions had zero counts")
    
    return is_valid, mismatch_count