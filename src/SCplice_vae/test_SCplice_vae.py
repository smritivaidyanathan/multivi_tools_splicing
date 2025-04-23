"""
Test script for SCpliceVAE.

This script provides functions to test the SCpliceVAE model components.
"""

import os
import numpy as np
import torch
import anndata as ad
import scipy.sparse as sp
from scipy.sparse import csr_matrix

# Import from our modules
from masking_utils import (
    generate_recon_mask, 
    apply_recon_mask_to_anndata, 
    compute_reconstruction_accuracy,
    handle_missing_data
)
from dataloader import AnnDataDataset, construct_input_dataloaders
from vae import (
    Encoder, 
    Decoder, 
    VAE, 
    binomial_loss_function, 
    beta_binomial_loss_function,
    plot_losses
)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

def create_synthetic_data(n_cells=100, n_features=50, sparsity=0.3):
    """Create synthetic data for testing."""
    print("Creating synthetic AnnData with simulated splicing data...")
    
    # Create random data matrices
    junc_ratio = np.random.random((n_cells, n_features))
    
    # Add some NaNs to junc_ratio to simulate missing data
    mask = np.random.random((n_cells, n_features)) < sparsity
    junc_ratio[mask] = np.nan
    
    # Create count matrices
    max_counts = 100
    cell_by_junction_matrix = np.random.randint(0, max_counts, (n_cells, n_features))
    cell_by_cluster_matrix = np.random.randint(0, max_counts, (n_cells, n_features))
    
    # Apply missing data mask to count matrices too
    cell_by_junction_matrix[mask] = 0
    cell_by_cluster_matrix[mask] = 0
    
    # Create AnnData object
    adata = ad.AnnData(
        X=np.zeros((n_cells, n_features)),  # Placeholder for X
        obs={"cell_type_grouped": np.random.choice(["TypeA", "TypeB", "TypeC"], n_cells)},
        var={"feature_id": [f"feature_{i}" for i in range(n_features)]}
    )
    
    # Add layers to AnnData
    adata.layers["junc_ratio"] = junc_ratio
    adata.layers["cell_by_junction_matrix"] = cell_by_junction_matrix
    adata.layers["cell_by_cluster_matrix"] = cell_by_cluster_matrix
    
    print(f"Created AnnData with {n_cells} cells and {n_features} features")
    print(f"Missing data percentage: {np.mean(mask) * 100:.2f}%")
    
    return adata

def test_masking_functions(adata, mask_percentage=0.1):
    """Test masking utility functions."""
    print("\n=== Testing Masking Functions ===")
    
    # Test generate_recon_mask
    print("\nTesting generate_recon_mask...")
    mask, seed = generate_recon_mask(adata, layer_key="junc_ratio", mask_percentage=mask_percentage)
    print(f"Generated mask with seed {seed}, shape {mask.shape}")
    
    # Test apply_recon_mask_to_anndata
    print("\nTesting apply_recon_mask_to_anndata...")
    adata_masked = apply_recon_mask_to_anndata(adata, mask, layer_key="junc_ratio")
    
    # Check that we've properly set up the backup
    if "original_junc_ratio" in adata_masked.layers:
        print("Successfully created backup layer 'original_junc_ratio'")
    else:
        print("ERROR: Backup layer not created")
    
    return adata_masked, mask

def test_missing_data_handling(adata):
    """Test different missing data handling methods."""
    print("\n=== Testing Missing Data Handling ===")
    
    # Make copies to test different methods
    adata_zero_out = adata.copy()
    adata_mask_out = adata.copy()
    
    # Test ZeroOut method
    print("\nTesting ZeroOut method...")
    adata_zero_out = handle_missing_data(adata_zero_out, method="ZeroOut")
    
    # Test MaskOut method
    print("\nTesting MaskOut method...")
    adata_mask_out = handle_missing_data(adata_mask_out, method="MaskOut")
    
    # Check that we've created the mask
    if "junc_ratio_NaN_mask" in adata_mask_out.layers:
        print("Successfully created mask layer 'junc_ratio_NaN_mask'")
    else:
        print("ERROR: Mask layer not created")
    
    return adata_zero_out, adata_mask_out

def test_dataset_and_dataloaders(adata, batch_size=16):
    """Test dataset and dataloader functionality."""
    print("\n=== Testing Dataset and DataLoaders ===")
    
    # Test AnnDataDataset
    print("\nTesting AnnDataDataset...")
    dataset = AnnDataDataset(adata)
    print(f"Dataset size: {len(dataset)}")
    
    # Get one item and check its structure
    sample = dataset[0]
    print("Sample keys:", list(sample.keys()))
    print("Sample shapes:", {k: v.shape for k, v in sample.items()})
    
    # Test construct_input_dataloaders
    print("\nTesting construct_input_dataloaders...")
    full_loader, train_loader, val_loader = construct_input_dataloaders(
        adata, batch_size=batch_size
    )
    print(f"Full loader batches: {len(full_loader)}")
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Val loader batches: {len(val_loader)}")
    
    return full_loader, train_loader, val_loader

def test_vae_model(input_dim, hidden_dims, latent_dim, output_dim):
    """Test VAE model components."""
    print("\n=== Testing VAE Model Components ===")
    
    # Test Encoder
    print("\nTesting Encoder...")
    num_hidden_layers = len(hidden_dims) - 1
    encoder = Encoder(input_dim, num_hidden_layers, hidden_dims, latent_dim)
    x = torch.randn(2, input_dim)
    mean, log_var = encoder(x)
    print(f"Encoder input shape: {x.shape}")
    print(f"Encoder mean output shape: {mean.shape}")
    print(f"Encoder log_var output shape: {log_var.shape}")
    
    # Test Decoder
    print("\nTesting Decoder...")
    decoder = Decoder(latent_dim, num_hidden_layers, hidden_dims, output_dim)
    z = torch.randn(2, latent_dim)
    reconstruction = decoder(z)
    print(f"Decoder input shape: {z.shape}")
    print(f"Decoder output shape: {reconstruction.shape}")
    
    # Test full VAE
    print("\nTesting full VAE...")
    vae = VAE(input_dim, num_hidden_layers, hidden_dims, latent_dim, output_dim)
    output, mean, log_var = vae(x)
    print(f"VAE input shape: {x.shape}")
    print(f"VAE output shape: {output.shape}")
    
    return vae

def test_loss_functions(vae_model):
    """Test loss functions."""
    print("\n=== Testing Loss Functions ===")
    
    # Create dummy data
    batch_size = 2
    input_dim = vae_model.encoder.input[0].in_features
    x = torch.randn(batch_size, input_dim)
    junction_counts = torch.randint(0, 10, (batch_size, input_dim)).float()
    n_cluster_counts = junction_counts + torch.randint(0, 10, (batch_size, input_dim)).float()
    
    # Forward pass
    reconstruction, mean, log_vars = vae_model(x)
    
    # Test binomial loss
    print("\nTesting binomial_loss_function...")
    binomial_loss = binomial_loss_function(
        logits=reconstruction,
        junction_counts=junction_counts,
        mean=mean,
        log_vars=log_vars,
        n_cluster_counts=n_cluster_counts,
        n=batch_size,
        k=1
    )
    print(f"Binomial loss: {binomial_loss.item()}")
    
    # Test beta-binomial loss
    print("\nTesting beta_binomial_loss_function...")
    beta_binomial_loss = beta_binomial_loss_function(
        logits=reconstruction,
        junction_counts=junction_counts,
        mean=mean,
        log_vars=log_vars,
        n_cluster_counts=n_cluster_counts,
        n=batch_size,
        k=1,
        concentration=torch.exp(vae_model.log_concentration)
    )
    print(f"Beta-binomial loss: {beta_binomial_loss.item()}")

def train_mini_vae(vae_model, train_loader, val_loader, loss_type="Beta_Binomial", num_epochs=2):
    """Train a miniature version of the VAE for testing."""
    print(f"\n=== Training Mini VAE with {loss_type} Loss ===")
    
    # Set up loss function
    if loss_type == "Binomial":
        loss_function = binomial_loss_function
    else:
        loss_function = beta_binomial_loss_function
    
    # Train model
    train_losses, val_losses, epochs_trained = vae_model.train_model(
        loss_function=loss_function,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        num_epochs=num_epochs,
        learning_rate=0.01,
        patience=5,
        output_dir="./test_output",
        wandb_logging=False
    )
    
    # Plot losses
    os.makedirs("./test_output", exist_ok=True)
    plot_losses(train_losses, val_losses, "./test_output")
    
    return vae_model, train_losses, val_losses

def test_reconstruction_accuracy(vae_model, adata, mask):
    """Test reconstruction accuracy calculation."""
    print("\n=== Testing Reconstruction Accuracy ===")
    
    # Test compute_reconstruction_accuracy for different metrics
    print("\nTesting compute_reconstruction_accuracy...")
    mae = compute_reconstruction_accuracy(
        vae_model, adata, mask, layer_key="junc_ratio", metric="MAE"
    )
    mse = compute_reconstruction_accuracy(
        vae_model, adata, mask, layer_key="junc_ratio", metric="MSE"
    )
    median_l1 = compute_reconstruction_accuracy(
        vae_model, adata, mask, layer_key="junc_ratio", metric="median_L1"
    )
    
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"Median L1: {median_l1}")

def main():
    """Main function to run all tests."""
    print("===== SCpliceVAE Tests =====")
    
    # Create synthetic data
    adata = create_synthetic_data(n_cells=50, n_features=20, sparsity=0.2)
    
    # Test masking functions
    adata_masked, mask = test_masking_functions(adata, mask_percentage=0.1)
    
    # Test missing data handling
    adata_zero_out, adata_mask_out = test_missing_data_handling(adata.copy())
    
    # Test dataset and dataloaders
    full_loader, train_loader, val_loader = test_dataset_and_dataloaders(adata_mask_out, batch_size=16)
    
    # Test VAE model
    input_dim = adata.shape[1]
    hidden_dims = [64, 32]
    latent_dim = 10
    output_dim = input_dim
    vae_model = test_vae_model(input_dim, hidden_dims, latent_dim, output_dim)
    
    # Test loss functions
    test_loss_functions(vae_model)
    
    # Test mini training
    vae_model, train_losses, val_losses = train_mini_vae(
        vae_model, train_loader, val_loader, loss_type="Beta_Binomial", num_epochs=2
    )
    
    # Test reconstruction accuracy
    test_reconstruction_accuracy(vae_model, adata_masked, mask)
    
    print("\n===== All Tests Completed =====")

if __name__ == "__main__":
    main()