"""
VAE model components for SCpliceVAE.

This module provides the Variational Autoencoder model components and loss functions
for the SCpliceVAE model.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
import matplotlib.pyplot as plt

# ------------------------------
# Model Components
# ------------------------------

class Encoder(nn.Module):
    def __init__(self, input_dim, num_hidden_layers, hidden_dims, latent_dim, dropout_rate=0.0):
        """
        Encoder network for VAE.
        
        Parameters:
          input_dim: Dimension of input features.
          num_hidden_layers: Number of hidden layers.
          hidden_dims: List of hidden layer dimensions.
          latent_dim: Dimension of latent space.
          dropout_rate: Dropout rate for regularization.
        """
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            for i in range(num_hidden_layers)
        ])
        self.output_means = nn.Linear(hidden_dims[-1], latent_dim)
        self.output_log_vars = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x):
        x = self.input(x)
        for layer in self.hidden_layers:
            x = layer(x)
        means = self.output_means(x)
        log_vars = self.output_log_vars(x)
        return means, log_vars

class Decoder(nn.Module):
    def __init__(self, z_dim, num_hidden_layers, hidden_dims, output_dim, dropout_rate=0.0):
        """
        Decoder network for VAE.
        
        Parameters:
          z_dim: Dimension of latent space.
          num_hidden_layers: Number of hidden layers.
          hidden_dims: List of hidden layer dimensions.
          output_dim: Dimension of output.
          dropout_rate: Dropout rate for regularization.
        """
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(z_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dims[i+1], hidden_dims[i]),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            for i in reversed(range(num_hidden_layers))
        ])
        self.output = nn.Linear(hidden_dims[0], output_dim)

    def forward(self, x):
        x = self.input(x)
        for layer in self.hidden_layers:
            x = layer(x)
        reconstruction = self.output(x)
        return reconstruction

class VAE(nn.Module):
    def __init__(self, input_dim, num_hidden_layers, hidden_dims, latent_dim, output_dim, dropout_rate=0.0):
        """
        Variational Autoencoder for splicing data.
        
        Parameters:
          input_dim: Dimension of input features.
          num_hidden_layers: Number of hidden layers.
          hidden_dims: List of hidden layer dimensions.
          latent_dim: Dimension of latent space.
          output_dim: Dimension of output.
          dropout_rate: Dropout rate for regularization.
        """
        super().__init__()
        self.encoder = Encoder(input_dim, num_hidden_layers, hidden_dims, latent_dim, dropout_rate)
        self.decoder = Decoder(latent_dim, num_hidden_layers, hidden_dims, output_dim, dropout_rate)
        self.log_concentration = nn.Parameter(torch.tensor(0.0))

    def reparametrize(self, mean, log_vars):
        std = torch.exp(0.5 * log_vars)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        mean, log_vars = self.encoder(x)
        z = self.reparametrize(mean, log_vars)
        reconstruction = self.decoder(z)
        return reconstruction.to(torch.float32), mean.to(torch.float32), log_vars.to(torch.float32)

    def train_model(self, loss_function, train_dataloader, val_dataloader, num_epochs, 
                   learning_rate, patience, schedule_step_size=50, schedule_gamma=0.1,
                   output_dir=None, wandb_logging=False):
        """
        Train the VAE model.
        
        Parameters:
          loss_function: Loss function to use.
          train_dataloader: DataLoader for training data.
          val_dataloader: DataLoader for validation data.
          num_epochs: Number of epochs to train for.
          learning_rate: Learning rate for optimizer.
          patience: Number of epochs to wait for improvement before early stopping.
          schedule_step_size: Number of epochs before learning rate adjustment.
          schedule_gamma: Factor to multiply learning rate by on schedule.
          output_dir: Directory to save model and outputs.
          wandb_logging: Whether to log to Weights & Biases.
          
        Returns:
          train_losses: List of training losses per epoch.
          val_losses: List of validation losses per epoch.
          epochs_trained: Number of epochs actually trained for.
        """
        device = next(self.parameters()).device
        print("Beginning training")
        
        if output_dir:
            model_dir = os.path.join(output_dir, "model")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "best_model.pth")
        else:
            model_path = "best_model.pth"

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=schedule_step_size, gamma=schedule_gamma)

        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        bad_epochs = 0
        epochs_trained = 0

        for epoch in range(num_epochs):
            epochs_trained += 1
            epoch_loss = 0
            self.train()

            # Training loop
            for batch in train_dataloader:
                # Move all batch tensors to the correct device
                for key in batch:
                    batch[key] = batch[key].to(device)

                optimizer.zero_grad()

                # Forward pass (on junc_ratio)
                reconstruction, mean, log_vars = self.forward(batch["junc_ratio"])
                concentration = torch.exp(self.log_concentration)

                # For logging
                probabilities = torch.sigmoid(reconstruction).clamp(1e-6, 1 - 1e-6)
                alpha = probabilities * concentration
                beta = (1 - probabilities) * concentration

                # If "MaskOut" is being used, we'll pass the mask, otherwise pass None
                mask_tensor = None
                if "junc_ratio_NaN_mask" in batch:
                    mask_tensor = batch["junc_ratio_NaN_mask"]

                # Compute loss and backpropagate
                loss = loss_function(
                    logits=reconstruction,
                    junction_counts=batch["cell_by_junction_matrix"],
                    mean=mean,
                    log_vars=log_vars,
                    n_cluster_counts=batch["cell_by_cluster_matrix"],
                    n=len(train_dataloader.dataset),
                    k=len(train_dataloader),
                    concentration=concentration,
                    mask=mask_tensor
                )
                loss.backward()
                epoch_loss += loss.item()
                optimizer.step()

            # Average training loss
            train_epoch_loss = epoch_loss / len(train_dataloader)
            train_losses.append(train_epoch_loss)
            scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            
            # Log to wandb if requested
            if wandb_logging:
                import wandb
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_epoch_loss,
                    "concentration": concentration.item(),
                    "mean_alpha": alpha.mean().item(),
                    "mean_beta": beta.mean().item(), 
                    "learning_rate": current_lr
                })
            
            print(
                f"epoch {epoch+1}/{num_epochs}; train loss = {train_epoch_loss:.4f}; "
                f"concentration = {concentration.item():.4f}, mean alpha = {alpha.mean().item():.4f}, "
                f"mean beta = {beta.mean().item():.4f}",
                flush=True
            )

            # Validation loop
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    for key in batch:
                        batch[key] = batch[key].to(device)
                    reconstruction, mean, log_vars = self.forward(batch["junc_ratio"])
                    concentration = torch.exp(self.log_concentration)

                    mask_tensor = None
                    if "junc_ratio_NaN_mask" in batch:
                        mask_tensor = batch["junc_ratio_NaN_mask"]

                    val_batch_loss = loss_function(
                        logits=reconstruction,
                        junction_counts=batch["cell_by_junction_matrix"],
                        mean=mean,
                        log_vars=log_vars,
                        n_cluster_counts=batch["cell_by_cluster_matrix"],
                        n=len(val_dataloader.dataset),
                        k=len(val_dataloader),
                        concentration=concentration,
                        mask=mask_tensor
                    )
                    val_loss += val_batch_loss.item()

            val_epoch_loss = val_loss / len(val_dataloader)
            val_losses.append(val_epoch_loss)
            
            if wandb_logging:
                wandb.log({"validation_loss": val_epoch_loss})
                
            print(f"validation loss = {val_epoch_loss:.4f}", flush=True)

            # Early stopping check
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                torch.save(self.state_dict(), model_path)
                bad_epochs = 0
            else:
                bad_epochs += 1

            if bad_epochs >= patience:
                print("Early stopping triggered. Ran out of patience", flush=True)
                print(f"Best Validation Loss: {best_val_loss}")
                break

        # Load best model weights
        self.load_state_dict(torch.load(model_path))
        return train_losses, val_losses, epochs_trained

    def get_latent_rep(self, layer):
        """
        Get latent representation for input data.
        
        Parameters:
          layer: Input data tensor.
          
        Returns:
          z: Latent representation (numpy array).
        """
        self.encoder.eval()
        layer = layer.to(next(self.parameters()).device)
        with torch.no_grad():
            mean, log_vars = self.encoder(layer)
            z = self.reparametrize(mean, log_vars)
        return z.cpu().numpy()

# ------------------------------
# Loss Functions
# ------------------------------

def binomial_loss_function(
    logits, junction_counts, mean, log_vars, n_cluster_counts,
    n, k, concentration=0, mask=None
):
    """
    Binomial loss function for VAE.
    
    Parameters:
      logits: Reconstructed logits from decoder.
      junction_counts: Junction counts from data.
      mean: Mean of latent distribution.
      log_vars: Log variance of latent distribution.
      n_cluster_counts: Cluster counts from data.
      n: Number of samples in dataset.
      k: Number of batches in dataloader.
      concentration: Concentration parameter.
      mask: Optional mask to exclude missing values.
      
    Returns:
      total_loss: Combined reconstruction and KL loss.
    """
    # If mask is not None (we are masking NaNs), we apply it to skip missing features in the log-likelihood
    if mask is not None:
        # Do the masking
        logits = logits[mask]
        junction_counts = junction_counts[mask]
        n_cluster_counts = n_cluster_counts[mask]
    # Otherwise, we do not mask, use all entries

    probabilities = torch.sigmoid(logits)
    log_prob = torch.log(probabilities + 1e-10)
    log_one_minus_prob = torch.log(1 - probabilities + 1e-10)
    log_likelihood = (
        junction_counts * log_prob
        + (n_cluster_counts - junction_counts) * log_one_minus_prob
    )
    log_likelihood = log_likelihood * (float(n) / float(k))
    reconstruction_loss = -log_likelihood.mean()

    qz = Normal(mean, torch.sqrt(torch.exp(log_vars)))
    pz = Normal(0, 1)
    kl_div = kl_divergence(qz, pz).sum(dim=1).mean()

    total_loss = reconstruction_loss + kl_div
    return total_loss

def beta_binomial_log_pmf(k, n, alpha, beta):
    """
    Calculate the log probability mass function for the beta-binomial distribution.
    
    Parameters:
      k: Number of successes.
      n: Number of trials.
      alpha: Alpha parameter of beta distribution.
      beta: Beta parameter of beta distribution.
      
    Returns:
      log_pmf: Log of probability mass function.
    """
    return (
        torch.lgamma(n + 1)
        - torch.lgamma(k + 1)
        - torch.lgamma(n - k + 1)
        + torch.lgamma(k + alpha)
        + torch.lgamma(n - k + beta)
        - torch.lgamma(n + alpha + beta)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
        + torch.lgamma(alpha + beta)
    )

def beta_binomial_loss_function(
    logits, junction_counts, mean, log_vars, n_cluster_counts,
    n, k, concentration=0.0, mask=None
):
    """
    Beta-binomial loss function for VAE.
    
    Parameters:
      logits: Reconstructed logits from decoder.
      junction_counts: Junction counts from data.
      mean: Mean of latent distribution.
      log_vars: Log variance of latent distribution.
      n_cluster_counts: Cluster counts from data.
      n: Number of samples in dataset.
      k: Number of batches in dataloader.
      concentration: Concentration parameter.
      mask: Optional mask to exclude missing values.
      
    Returns:
      total_loss: Combined reconstruction and KL loss.
    """
    # If mask is not None (we are masking NaNs), we apply it to skip missing features in the log-likelihood
    if mask is not None:
        # Do the masking
        logits = logits[mask]
        junction_counts = junction_counts[mask]
        n_cluster_counts = n_cluster_counts[mask]
    # Otherwise, we do not mask, use all entries

    probabilities = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    alpha = probabilities * concentration
    beta = (1.0 - probabilities) * concentration

    log_likelihood = beta_binomial_log_pmf(
        k=junction_counts,
        n=n_cluster_counts,
        alpha=alpha,
        beta=beta
    )
    log_likelihood = log_likelihood * (float(n) / float(k))
    reconstruction_loss = -log_likelihood.mean()

    qz = Normal(mean, torch.sqrt(torch.exp(log_vars)))
    pz = Normal(0, 1)
    kl_div = kl_divergence(qz, pz).sum(dim=1).mean()

    total_loss = reconstruction_loss + kl_div
    return total_loss

# ------------------------------
# Visualization Functions
# ------------------------------

def plot_losses(train_losses, val_losses, output_dir="./"):
    """
    Plot training and validation losses.
    
    Parameters:
      train_losses: List of training losses per epoch.
      val_losses: List of validation losses per epoch.
      output_dir: Directory to save the plot.
    """
    fig, ax = plt.subplots()
    ax.plot(train_losses, label='Training Loss')
    ax.plot(val_losses, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    loss_plot_path = os.path.join(output_dir, "loss_plot.png")
    fig.savefig(loss_plot_path)