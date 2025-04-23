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

class PartialEncoder(nn.Module):
    def __init__(self, input_dim: int, h_hidden_dim: int, encoder_hidden_dim: int, 
                 latent_dim: int, code_dim: int, dropout_rate: float = 0.0):
        """
        Encoder network inspired by PointNet for partially observed data.

        Processes each observed feature individually using a shared network ('h_layer')
        combined with learnable feature embeddings and biases, then aggregates
        the results before mapping to the latent space.

        Parameters:
          input_dim (int): Dimension of input features (D). Number of junctions/features.
          h_hidden_dim (int): Hidden dimension for the shared 'h_layer'.
                           (Replaces the misuse of num_hidden_layers in the original h_layer definition).
          encoder_hidden_dim (int): Hidden dimension for the final 'encoder_mlp'.
                                 (Replaces the hardcoded 256 in the original encoder_mlp).
          latent_dim (int): Dimension of latent space (Z).
          code_dim (int): Dimension of feature embeddings and intermediate representations (K).
          dropout_rate (float): Dropout rate for regularization applied within h_layer and encoder_mlp.
        """
        super().__init__()
        self.input_dim = input_dim
        self.code_dim = code_dim
        self.latent_dim = latent_dim

        # Learnable feature embedding (F_d in paper notation)
        # Shape: (D, K)
        self.feature_embedding = nn.Parameter(torch.randn(input_dim, code_dim))

        # Learnable bias term per feature (b_d in paper notation)
        # Shape: (D, 1)
        self.feature_bias = nn.Parameter(torch.zeros(input_dim, 1))

        # Shared function h(.) applied to each feature representation s_d = [x_d, F_d, b_d]
        # Input dim: 1 (feature value) + K (embedding) + 1 (bias) = K + 2
        # Output dim: K (code_dim)
        self.h_layer = nn.Sequential(
            nn.Linear(1 + code_dim + 1, h_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Added dropout
            nn.Linear(h_hidden_dim, code_dim),
            nn.ReLU() # ReLU after last linear is common in intermediate feature extractors
        )

        # MLP to map aggregated representation 'c' to latent distribution parameters
        # Input dim: K (code_dim)
        # Output dim: 2 * Z (for mu and logvar)
        self.encoder_mlp = nn.Sequential(
            nn.Linear(code_dim, encoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Added dropout
            nn.Linear(encoder_hidden_dim, 2 * latent_dim) # outputs both mu and logvar
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input data (batch_size, input_dim). Missing values can be anything (e.g., 0, NaN),
                              as they will be masked out based on the 'mask' tensor.
                              It's crucial that the *observed* values in x are the actual measurements.
            mask (torch.Tensor): Binary mask (batch_size, input_dim). 1 indicates observed, 0 indicates missing.
                               Must be float or long/int and compatible with multiplication.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - mu (torch.Tensor): Mean of latent distribution (batch_size, latent_dim).
                - logvar (torch.Tensor): Log variance of latent distribution (batch_size, latent_dim).
        """
        batch_size = x.size(0)

        # --- Input Validation ---
        if x.shape[1] != self.input_dim or mask.shape[1] != self.input_dim:
             raise ValueError(f"Input tensor feature dimension ({x.shape[1]}) or mask dimension ({mask.shape[1]}) "
                              f"does not match encoder input_dim ({self.input_dim})")
        if x.shape != mask.shape:
             raise ValueError(f"Input tensor shape ({x.shape}) and mask shape ({mask.shape}) must match.")
        if x.ndim != 2 or mask.ndim != 2:
             raise ValueError(f"Input tensor and mask must be 2D (batch_size, input_dim). Got shapes {x.shape} and {mask.shape}")


        # Step 1: Reshape inputs for processing each feature independently
        # Flatten batch and feature dimensions: (B, D) -> (B*D, 1)
        x_flat = x.reshape(-1, 1)                                # Shape: (B*D, 1)
        # mask_flat = mask.reshape(-1, 1)                        # Shape: (B*D, 1) - Not directly used here, but illustrates the mapping

        # Step 2: Prepare feature embeddings and biases for each item in the flattened batch
        # Feature embeddings F_d: (D, K) -> (B*D, K) by repeating for each batch item
        # Feature biases b_d: (D, 1) -> (B*D, 1) by repeating for each batch item

        # Efficient expansion using broadcasting: expand creates views without copying memory initially, reshape makes it contiguous if needed later
        # unsqueeze(0) adds batch dim: (D, K) -> (1, D, K)
        # expand(batch_size, -1, -1) repeats view across batch dim: (1, D, K) -> (B, D, K)
        # reshape(-1, self.code_dim) flattens B and D dims: (B, D, K) -> (B*D, K)
        F_embed = self.feature_embedding.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, self.code_dim) # Shape: (B*D, K)
        b_embed = self.feature_bias.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 1)                   # Shape: (B*D, 1)

        # Step 3: Construct input for the shared 'h' function for each feature instance
        # Input s_d = [x_d, F_d, b_d]
        h_input = torch.cat([x_flat, F_embed, b_embed], dim=1)  # Shape: (B*D, 1 + K + 1)

        # Step 4: Apply the shared h network to each feature representation s_d
        h_out_flat = self.h_layer(h_input)                      # Shape: (B*D, K)

        # Step 5: Reshape back to (batch_size, num_features, code_dim)
        h_out = h_out_flat.view(batch_size, self.input_dim, self.code_dim)  # Shape: (B, D, K)

        # Step 6: Apply the mask. Zero out representations of missing features.
        # Ensure mask is float for multiplication if it isn't already.
        mask_float = mask.float() # Convert mask to float if it's int/bool
        # Expand mask: (B, D) -> (B, D, 1) for broadcasting
        mask_exp = mask_float.unsqueeze(-1)                           # Shape: (B, D, 1)
        h_masked = h_out * mask_exp                             # Shape: (B, D, K)

        # Step 7: Aggregate over observed features (permutation-invariant function g)
        # Summation is a common choice for aggregation.
        # Sum along the feature dimension (dim=1)
        # Combining Features Per Cell 
        c = h_masked.sum(dim=1)                                 # Shape: (B, K)

        # Optional: Normalize aggregation by the number of observed features (Mean aggregation)
        # This can make the aggregated representation 'c' less dependent on the *number* of observed features.
        # num_observed = mask_float.sum(dim=1, keepdim=True) # Shape: (B, 1)
        # # Avoid division by zero if a sample has no observed features (add small epsilon)
        # c = c / (num_observed + 1e-8) # Shape: (B, K)
        # Uncomment the 3 lines above if you prefer mean aggregation over sum aggregation.

        # Step 8: Pass the aggregated representation 'c' through the final MLP (phi)
        enc_out = self.encoder_mlp(c)                           # Shape: (B, 2*Z)

        # Step 9: Split the output into mean (mu) and log variance (logvar)
        mu, logvar = enc_out.chunk(2, dim=-1)                   # Shapes: (B, Z), (B, Z)

        return mu, logvar

# Assume PartialDecoder is defined as we designed it previously:
class PartialDecoder(nn.Module):
    def __init__(self, latent_dim: int, decoder_hidden_dim: int, output_dim: int, code_dim: int, dropout_rate: float = 0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.code_dim = code_dim

        self.z_processor = nn.Sequential(
            nn.Linear(latent_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # Optional project down if needed, depends on how you combine
            # nn.Linear(decoder_hidden_dim, code_dim),
            # nn.ReLU()
        )

        # Input: processed_z + F_d + b_d
        self.j_layer = nn.Sequential(
            nn.Linear(decoder_hidden_dim + code_dim + 1, decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(decoder_hidden_dim, 1) # Predict 1 value per feature
            # Add final activation (e.g., nn.Sigmoid()) if output should be bounded
        )

    def forward(self, z: torch.Tensor, feature_embedding: nn.Parameter, feature_bias: nn.Parameter) -> torch.Tensor:
        batch_size = z.size(0)
        if feature_embedding.shape != (self.output_dim, self.code_dim) or feature_bias.shape != (self.output_dim, 1):
             raise ValueError("Feature embedding/bias shapes mismatch in decoder forward.")

        processed_z = self.z_processor(z)
        processed_z_expanded = processed_z.unsqueeze(1).expand(-1, self.output_dim, -1)
        F_embed_expanded = feature_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        b_embed_expanded = feature_bias.unsqueeze(0).expand(batch_size, -1, -1)

        j_input = torch.cat([processed_z_expanded, F_embed_expanded, b_embed_expanded], dim=2)
        j_input_flat = j_input.view(-1, j_input.shape[-1])
        j_out_flat = self.j_layer(j_input_flat)
        reconstruction = j_out_flat.view(batch_size, self.output_dim)
        # reconstruction = torch.sigmoid(reconstruction) # Example activation
        return reconstruction
    
class PartialVAE(nn.Module):
    def __init__(self, input_dim: int, code_dim: int, h_hidden_dim: int, encoder_hidden_dim: int, latent_dim: int, decoder_hidden_dim: int, dropout_rate: float = 0.0, learn_concentration: bool = True):
        """
        Partial Variational Autoencoder using PointNet-like Encoder/Decoder.

        Parameters:
          input_dim (int): Dimension of input/output features (D).
          code_dim (int): Dimension of feature embeddings (K).
          h_hidden_dim (int): Hidden dimension for the encoder's shared h_layer.
          encoder_hidden_dim (int): Hidden dimension for the encoder's final MLP.
          latent_dim (int): Dimension of latent space (Z).
          decoder_hidden_dim (int): Hidden dimension for the decoder's shared j_layer and z_processor.
          dropout_rate (float): Dropout rate for regularization.
          learn_concentration (bool): If True, add a learnable parameter for beta-binomial concentration.
        """
        super().__init__()

        # --- Parameter Validation ---
        if not all(isinstance(i, int) and i > 0 for i in [input_dim, code_dim, h_hidden_dim, encoder_hidden_dim, latent_dim, decoder_hidden_dim]):
             raise ValueError("All dimensions must be positive integers.")
        if not isinstance(dropout_rate, float) or not (0.0 <= dropout_rate < 1.0):
             raise ValueError("dropout_rate must be a float between 0.0 and 1.0 (exclusive of 1.0)")

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.learn_concentration = learn_concentration

        # Instantiate the Partial Encoder
        self.encoder = PartialEncoder(
            input_dim=input_dim,
            h_hidden_dim=h_hidden_dim,
            encoder_hidden_dim=encoder_hidden_dim,
            latent_dim=latent_dim,
            code_dim=code_dim,
            dropout_rate=dropout_rate
        )

        # Instantiate the Partial Decoder
        self.decoder = PartialDecoder(
            latent_dim=latent_dim,
            decoder_hidden_dim=decoder_hidden_dim,
            output_dim=input_dim, # Output dim must match input dim for reconstruction
            code_dim=code_dim,
            dropout_rate=dropout_rate
        )

        # Add learnable concentration parameter if requested (for beta-binomial)
        if self.learn_concentration:
            # Initialize concentration positively (e.g., starting near 1 or 10)
            # Using log makes optimization more stable. exp(0) = 1. exp(log(10)) = 10.
            self.log_concentration = nn.Parameter(torch.tensor(np.log(10.0))) # Example: start concentration near 10
        else:
             self.log_concentration = None # Indicate it's not used

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * epsilon
        """
        if not self.training:
            # During evaluation, generally use the mean
            return mu
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.

        Args:
            x (torch.Tensor): Input data (batch_size, input_dim). NaNs/zeros expected where mask=0.
            mask (torch.Tensor): Observation mask (batch_size, input_dim), 1=observed, 0=missing.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - reconstruction (torch.Tensor): Reconstructed data parameters (logits) (batch_size, input_dim).
                - mu (torch.Tensor): Latent mean (batch_size, latent_dim).
                - logvar (torch.Tensor): Latent log variance (batch_size, latent_dim).
        """
        # Device handling happens in training loop or inference script
        # x = x.to(next(self.parameters()).device)
        # mask = mask.to(next(self.parameters()).device)

        # 1. Encode - Pass BOTH x and mask
        mu, logvar = self.encoder(x, mask)

        # 2. Reparameterize
        z = self.reparameterize(mu, logvar)

        # 3. Decode - Pass z AND the learned embeddings/biases from the encoder
        reconstruction = self.decoder(z, self.encoder.feature_embedding, self.encoder.feature_bias)

        return reconstruction, mu, logvar

    def train_model(self, loss_function, train_dataloader, val_dataloader, num_epochs,
                   learning_rate, patience, fixed_concentration=None, # Added option for fixed concentration
                   schedule_step_size=50, schedule_gamma=0.1,
                   output_dir=None, wandb_logging=False,
                   input_key='x', # Key for input data in batch dict
                   mask_key='mask', # Key for mask in batch dict
                   junction_counts_key='junction_counts', # Key for junction counts
                   cluster_counts_key='cluster_counts' # Key for cluster counts
                   ):
        """
        Train the VAE model. Handles device placement, logging, early stopping.

        Parameters:
          loss_function: Loss function (e.g., binomial_loss_function, beta_binomial_loss_function).
          train_dataloader: DataLoader for training data.
          val_dataloader: DataLoader for validation data.
          num_epochs: Number of epochs to train for.
          learning_rate: Learning rate for optimizer.
          patience: Epochs to wait for val_loss improvement before early stopping.
          fixed_concentration: If not None, use this fixed value for concentration in beta-binomial loss.
                               Overrides learn_concentration=True in init.
          schedule_step_size: LR scheduler step size (epochs).
          schedule_gamma: LR scheduler multiplicative factor.
          output_dir: Directory to save model and outputs. If None, saves locally.
          wandb_logging: Whether to log metrics to Weights & Biases. Requires wandb to be installed and initialized.
          input_key, mask_key, ... : Keys expected in the batch dictionary from the DataLoader.

        Returns:
          tuple: (train_losses, val_losses, epochs_trained)
        """

        device = next(self.parameters()).device
        print(f"Beginning training on device: {device}")

        if output_dir:
            model_dir = os.path.join(output_dir, "model")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "best_model.pth")
        else:
            # Saves in current working directory if output_dir is None
            model_path = "best_model.pth"
            print(f"Warning: output_dir not specified. Saving best model to {os.path.abspath(model_path)}")


        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=schedule_step_size, gamma=schedule_gamma)

        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        bad_epochs = 0
        epochs_trained = 0

        # Get dataset sizes for loss scaling
        n_train = len(train_dataloader.dataset)
        k_train = len(train_dataloader)
        n_val = len(val_dataloader.dataset)
        k_val = len(val_dataloader)

        for epoch in range(num_epochs):
            epochs_trained += 1
            epoch_train_loss = 0.0
            self.train() # Set model to training mode

            # --- Training Loop ---
            for batch in train_dataloader:
                # Move batch data to device
                x_batch = batch[input_key].to(device)
                mask_batch = batch[mask_key].to(device)
                j_counts_batch = batch[junction_counts_key].to(device)
                c_counts_batch = batch[cluster_counts_key].to(device)

                optimizer.zero_grad()

                # Forward pass - REQUIRES BOTH x AND mask
                reconstruction, mu, logvar = self.forward(x_batch, mask_batch)

                # Determine concentration for loss
                if fixed_concentration is not None:
                    concentration_val = torch.tensor(fixed_concentration, device=device, dtype=torch.float32)
                elif self.learn_concentration and self.log_concentration is not None:
                    concentration_val = torch.exp(self.log_concentration)
                else:
                    # Default if concentration is not applicable (e.g., binomial loss) or not learned
                    concentration_val = torch.tensor(0.0, device=device) # Pass 0 if loss function doesn't use it

                # Compute loss
                loss = loss_function(
                        logits=reconstruction,
                        junction_counts=j_counts_batch,
                        mean=mu,
                        log_vars=logvar,
                        n_cluster_counts=c_counts_batch,
                        n=n_train, # Total training samples
                        k=k_train, # Total training batches
                        concentration=concentration_val,
                        mask=mask_batch # Pass the mask for loss calculation!
                    )

                # Backward pass and optimization
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_train_loss += loss.item()

            # --- End of Training Epoch ---
            avg_train_loss = epoch_train_loss / k_train # Use k_train (num batches) for average per-batch loss
            train_losses.append(avg_train_loss)
            scheduler.step() # Step the LR scheduler

            # Logging training stats
            current_lr = optimizer.param_groups[0]['lr']
            log_msg = (
                f"Epoch {epoch+1:03d}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | LR: {current_lr:.1e}"
            )
            if self.learn_concentration and fixed_concentration is None:
                # Log learned concentration if applicable
                log_msg += f" | Learned Conc: {concentration_val.item():.4f}"
            print(log_msg, flush=True)

            # --- Validation Loop ---
            self.eval() # Set model to evaluation mode
            epoch_val_loss = 0.0
            with torch.no_grad():
                for batch in val_dataloader:
                    # Move batch data to device
                    x_batch = batch[input_key].to(device)
                    mask_batch = batch[mask_key].to(device)
                    j_counts_batch = batch[junction_counts_key].to(device)
                    c_counts_batch = batch[cluster_counts_key].to(device)

                    # Forward pass
                    reconstruction, mu, logvar = self.forward(x_batch, mask_batch)

                    # Determine concentration (consistent with training logic)
                    if fixed_concentration is not None:
                        concentration_val = torch.tensor(fixed_concentration, device=device, dtype=torch.float32)
                    elif self.learn_concentration and self.log_concentration is not None:
                        concentration_val = torch.exp(self.log_concentration)
                    else:
                        concentration_val = torch.tensor(0.0, device=device)

                    # Compute validation loss
                    val_batch_loss = loss_function(
                        logits=reconstruction,
                        junction_counts=j_counts_batch,
                        mean=mu,
                        log_vars=logvar,
                        n_cluster_counts=c_counts_batch,
                        n=n_val, # Use validation dataset size/batches
                        k=k_val,
                        concentration=concentration_val,
                        mask=mask_batch # Pass mask
                    )
                    epoch_val_loss += val_batch_loss.item()

            avg_val_loss = epoch_val_loss / k_val # Average per-batch validation loss
            val_losses.append(avg_val_loss)

            print(f"          | Val Loss:   {avg_val_loss:.4f}", flush=True)

            # --- Early Stopping Check ---
            if avg_val_loss < best_val_loss:
                print(f"          | Val loss improved ({best_val_loss:.4f} -> {avg_val_loss:.4f}")
                best_val_loss = avg_val_loss
                bad_epochs = 0 # Reset patience counter
            else:
                bad_epochs += 1
                print(f"          | Val loss did not improve. Bad epochs: {bad_epochs}/{patience}")

            if bad_epochs >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                break # Exit training loop

        # --- End of Training ---
        print(f"\nTraining finished after {epochs_trained} epochs.")
        print(f"Best Validation Loss: {best_val_loss:.4f}")
        return train_losses, val_losses, epochs_trained

    def get_latent_rep(self, x: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
        """
        Get latent representation (mean) for input data.

        Args:
            x (torch.Tensor): Input data tensor (batch_size, input_dim).
            mask (torch.Tensor): Observation mask tensor (batch_size, input_dim).

        Returns:
            np.ndarray: Latent representation (batch_size, latent_dim).
                        Returns the mean (mu) of the latent distribution in eval mode.
        """
        self.eval() # Set model to evaluation mode
        device = next(self.parameters()).device
        x = x.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            # Pass both x and mask to the encoder
            mean, log_vars = self.encoder(x, mask)
            # In eval mode, reparameterize just returns the mean
            z = self.reparameterize(mean, log_vars)
        return z.cpu().numpy()

    def get_features(self) -> np.ndarray:
        """
        Get the learned feature embeddings from the encoder.

        Returns:
            np.ndarray: Feature embedding matrix of shape (input_dim, code_dim).
        """
        # Detach from graph, move to CPU, convert to numpy
        return self.encoder.feature_embedding.detach().cpu().numpy()

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
    logits = logits[mask]
    junction_counts = junction_counts[mask]
    n_cluster_counts = n_cluster_counts[mask]

    probabilities = torch.sigmoid(logits)
    log_prob = torch.log(probabilities + 1e-10)
    log_one_minus_prob = torch.log(1 - probabilities + 1e-10)
    log_likelihood = (
        junction_counts * log_prob
        + (n_cluster_counts - junction_counts) * log_one_minus_prob
    )
    log_likelihood = log_likelihood * (float(n) / float(k))
    reconstruction_loss = -log_likelihood.mean()

    # --- KL Divergence Calculation ---
    # Clamp log_vars to a reasonable range to prevent exp() from overflowing
    # Values below -10 result in exp() close to 0, values above ~20 can risk overflow/instability
    log_vars_clipped = torch.clamp(log_vars, min=-10, max=20) # Adjust max if needed

    # Calculate standard deviation safely using the clipped log_vars
    std_dev = torch.sqrt(torch.exp(log_vars_clipped))

    # Define the approximate posterior distribution q(z|x)
    # Ensure std_dev is strictly positive (clamping log_vars helps significantly)
    qz = Normal(mean, std_dev)

    # Define the prior distribution p(z) - Ensure it's on the same device
    pz_scale = torch.ones_like(mean) # Create scale tensor on the same device as mean
    pz_loc = torch.zeros_like(mean)  # Create loc tensor on the same device
    pz = Normal(pz_loc, pz_scale) # Standard Normal prior

    # Calculate KL divergence
    # Ensure summing across the correct dimension (latent dimension, usually dim=1)
    kl_div = kl_divergence(qz, pz)
    if kl_div.ndim > 1: # Check if KL divergence is per-latent-dimension
        kl_div = kl_div.sum(dim=1) # Sum across latent dimensions if necessary
    kl_div = kl_div.mean() # Average across the batch

    # --- Total Loss ---
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

    # --- KL Divergence Calculation ---
    # Clamp log_vars to a reasonable range to prevent exp() from overflowing
    # Values below -10 result in exp() close to 0, values above ~20 can risk overflow/instability
    log_vars_clipped = torch.clamp(log_vars, min=-10, max=20) # Adjust max if needed

    # Calculate standard deviation safely using the clipped log_vars
    std_dev = torch.sqrt(torch.exp(log_vars_clipped))

    # Define the approximate posterior distribution q(z|x)
    # Ensure std_dev is strictly positive (clamping log_vars helps significantly)
    qz = Normal(mean, std_dev)

    # Define the prior distribution p(z) - Ensure it's on the same device
    pz_scale = torch.ones_like(mean) # Create scale tensor on the same device as mean
    pz_loc = torch.zeros_like(mean)  # Create loc tensor on the same device
    pz = Normal(pz_loc, pz_scale) # Standard Normal prior

    # Calculate KL divergence
    # Ensure summing across the correct dimension (latent dimension, usually dim=1)
    kl_div = kl_divergence(qz, pz)
    if kl_div.ndim > 1: # Check if KL divergence is per-latent-dimension
        kl_div = kl_div.sum(dim=1) # Sum across latent dimensions if necessary
    kl_div = kl_div.mean() # Average across the batch

    # --- Total Loss ---
    total_loss = reconstruction_loss + kl_div
    return total_loss