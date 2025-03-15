# %% [markdown]
# # Notebook for our single modality splicing VAE (SplicingVAE)

# %% [code]
# ------------------------------
# Imports
# ------------------------------
import os
import sys
import datetime
import json
import matplotlib.pyplot as plt

import numpy as np
from scipy.sparse import csr_matrix, issparse

import wandb

import jax
import jaxlib
print("jax version:", jax.__version__)
print("jaxlib version:", jaxlib.__version__)

import scvi
import h5py
import numpy as np
import anndata as ad
import pandas as pd
import scipy.sparse as sp

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.sparse import issparse

from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from sklearn.decomposition import PCA
import umap
import seaborn as sns
from sklearn.metrics import silhouette_score, silhouette_samples

# ------------------------------
# Global Configurations & Directories
# ------------------------------
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = os.path.join(os.getcwd(), "splice_vae_job_outputs", timestamp)
os.makedirs(output_dir, exist_ok=True)
model_dir = os.path.join(output_dir, "model")
os.makedirs(model_dir, exist_ok=True)

# Redirect stdout and stderr (optional backup logging)
sys.stdout = open(os.path.join(output_dir, "output.log"), "w")
sys.stderr = open(os.path.join(output_dir, "error.log"), "w")

print(f"Outputs are being saved in: {output_dir}")
print(f"Model will be saved in: {model_dir}")

# ------------------------------
# Hyperparameters and Configuration
# ------------------------------
LEARNING_RATE = 0.01
NUM_EPOCHS = 500
BATCH_SIZE = 2048
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')

NUM_HIDDEN_LAYERS = 1
HIDDEN_DIMS = [128, 64]
LATENT_DIM = 30
PATIENCE = 5
SCHEDULE_STEP_SIZE = 50
SCHEDULE_GAMMA = 0.1
TYPE_OF_PLOT = "UMAP" #"UMAP" or "PCA"
LOSS = "Beta_Binomial"  #"Beta_Binomial" or "Binomial"
MISSING_DATA_METHOD = "LeafletFA" #"ZeroOut", "MaskOut", "LeafletFA"
INPUT_DIM = None
OUTPUT_DIM = None #INPUT_DIM set after we read the AnnData

# ------------------------------
# Load Data
# ------------------------------
atse_anndata = ad.read_h5ad('/gpfs/commons/groups/knowles_lab/Karin/TMS_MODELING/DATA_FILES/SIMULATED/simulated_data_2025-03-12.h5ad')
print("Number of features:", atse_anndata.var.shape[0])
INPUT_DIM = atse_anndata.var.shape[0]
OUTPUT_DIM = INPUT_DIM
EPOCHS_TRAINED = 0

params = {
    "LEARNING_RATE": LEARNING_RATE,
    "NUM_EPOCHS": NUM_EPOCHS,
    "BATCH_SIZE": BATCH_SIZE,
    "USE_CUDA": USE_CUDA,
    "INPUT_DIM": INPUT_DIM,
    "NUM_HIDDEN_LAYERS": NUM_HIDDEN_LAYERS,
    "HIDDEN_DIMS": HIDDEN_DIMS,
    "LATENT_DIM": LATENT_DIM,
    "OUTPUT_DIM": OUTPUT_DIM,
    "PATIENCE": PATIENCE,
    "SCHEDULE_STEP_SIZE": SCHEDULE_STEP_SIZE,
    "SCHEDULE_GAMMA": SCHEDULE_GAMMA,
    "LOSS": LOSS,
    "TYPE_OF_PLOT": TYPE_OF_PLOT,
    "MISSING_DATA_METHOD": MISSING_DATA_METHOD
}

params_file = os.path.join(output_dir, "parameters.json")
with open(params_file, "w") as f:
    json.dump(params, f, indent=4)

# ------------------------------
# Initialize Weights & Biases
# ------------------------------
wandb.init(project="splicing_vae_project", config=params)
config = wandb.config

LEARNING_RATE = config.LEARNING_RATE
NUM_EPOCHS = config.NUM_EPOCHS
BATCH_SIZE = config.BATCH_SIZE

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')

NUM_HIDDEN_LAYERS = config.NUM_HIDDEN_LAYERS
HIDDEN_DIMS = config.HIDDEN_DIMS
LATENT_DIM = config.LATENT_DIM
PATIENCE = config.PATIENCE
SCHEDULE_STEP_SIZE = config.SCHEDULE_STEP_SIZE
SCHEDULE_GAMMA = config.SCHEDULE_GAMMA
TYPE_OF_PLOT = config.TYPE_OF_PLOT  # "UMAP" or "PCA"
LOSS = config.LOSS                  # "Beta_Binomial" or "Binomial"
MISSING_DATA_METHOD = config.MISSING_DATA_METHOD  # "ZeroOut", "MaskOut", or "LeafletFA"


# ------------------------------
# Junc Ratio Missing Data Handling
# ------------------------------

# Adjust NaN Handling or LeafletFA According to MISSING_DATA_METHOD
#   - If "ZeroOut": fill the junc_ratio layer's NaNs with 0 and do NOT mask.
#   - If "MaskOut": fill the junc_ratio layer's NaNs with 0 and create 'junc_ratio_NaN_mask'.
#   - If "LeafletFA": assume we have imputed_PSI layer for junc_ratio. No zero fill, no mask.
print(atse_anndata.layers)
if MISSING_DATA_METHOD == "ZeroOut":
    print("Filling NaNs with 0 in junc_ratio, no mask used.")
    layer = atse_anndata.layers['junc_ratio']
    if issparse(layer):
        dense_layer = layer.toarray()
    else:
        dense_layer = layer
    #replace NaNs with 0
    dense_layer = np.nan_to_num(dense_layer, nan=0.0)
    #overwrite junc_ratio
    atse_anndata.layers['junc_ratio'] = csr_matrix(dense_layer) if issparse(layer) else dense_layer

elif MISSING_DATA_METHOD == "MaskOut":
    print("MaskOut mode: fill junc_ratio with 0 for the encoder, but also store a mask to exclude from loss.")
    layer = atse_anndata.layers['junc_ratio']
    if issparse(layer):
        dense_layer = layer.toarray()
    else:
        dense_layer = layer

    #create a mask: True if not NaN
    junc_ratio_mask = ~np.isnan(dense_layer)
    #fill everything else with 0
    dense_layer_filled = np.nan_to_num(dense_layer, nan=0.0)

    #overwrite junc_ratio with the zero-filled version
    atse_anndata.layers['junc_ratio'] = csr_matrix(dense_layer_filled) if issparse(layer) else dense_layer_filled

    #also store the mask
    atse_anndata.layers['junc_ratio_NaN_mask'] = junc_ratio_mask

elif MISSING_DATA_METHOD == "LeafletFA":
    print("Using 'imputed_PSI' layer instead of 'junc_ratio', no zero fill, no mask.")
    if 'imputed_PSI' not in atse_anndata.layers:
        raise ValueError("No 'imputed_PSI' layer found in AnnData. Please provide it or switch MISSING_DATA_METHOD.")
        sys.exit(1)
    atse_anndata.layers['junc_ratio'] = atse_anndata.layers['imputed_PSI']
    print(atse_anndata.layers['junc_ratio'])
    # no mask creation needed
else:
    raise ValueError("MISSING_DATA_METHOD must be one of 'ZeroOut', 'MaskOut', 'LeafletFA'")

# ------------------------------
# DataSet & DataLoader Classes
# ------------------------------
class AnnDataDataset(Dataset):
    def __init__(self, anndata, first_three_only=True):
        """
        We'll grab up to the first three layers that exist in this anndata.
        Specifically, we at least want 'junc_ratio', 'cell_by_junction_matrix', 'cell_by_cluster_matrix'.
        If MISSING_DATA_METHOD is 'MaskOut', also gather 'junc_ratio_NaN_mask'.
        """
        self.tensors = {}
        
        wanted_keys = ["junc_ratio", "cell_by_junction_matrix", "cell_by_cluster_matrix"]
        for k in wanted_keys:
            layer_data = anndata.layers[k]
            if issparse(layer_data):
                layer_data = layer_data.toarray()
            self.tensors[k] = torch.tensor(layer_data, dtype=torch.float32)

        #if we are in MISSING_DATA_METHOD == "MaskOut", also load junc_ratio_NaN_mask
        if MISSING_DATA_METHOD == "MaskOut":
            mask_data = anndata.layers['junc_ratio_NaN_mask']  # a numpy bool array
            self.tensors["junc_ratio_NaN_mask"] = torch.tensor(mask_data, dtype=torch.bool)
        else:
            #no mask otherwise, so just keep going 
            pass

        self.num_samples = self.tensors["junc_ratio"].shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # return a dictionary of the data for this sample
        return {key: val[idx] for key, val in self.tensors.items()}

def construct_input_dataloaders(anndata, batch_size, validation_split=0.2):
    dataset = AnnDataDataset(anndata)
    total_size = len(dataset)
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    full_dataset = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return full_dataset, train_dataloader, val_dataloader

# ------------------------------
# Model Components: Encoder, Decoder, VAE
# ------------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, num_hidden_layers, num_hidden_units, latent_dim, dropout_rate=0.0):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, num_hidden_units[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_hidden_units[i], num_hidden_units[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            for i in range(num_hidden_layers)
        ])
        self.output_means = nn.Linear(num_hidden_units[-1], latent_dim)
        self.output_log_vars = nn.Linear(num_hidden_units[-1], latent_dim)

    def forward(self, x):
        x = self.input(x)
        for layer in self.hidden_layers:
            x = layer(x)
        means = self.output_means(x)
        log_vars = self.output_log_vars(x)
        return means, log_vars

class Decoder(nn.Module):
    def __init__(self, z_dim, num_hidden_layers, num_hidden_units, output_dim, dropout_rate=0.0):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(z_dim, num_hidden_units[-1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_hidden_units[i+1], num_hidden_units[i]),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            for i in reversed(range(num_hidden_layers))
        ])
        self.output = nn.Linear(num_hidden_units[0], output_dim)

    def forward(self, x):
        x = self.input(x)
        for layer in self.hidden_layers:
            x = layer(x)
        reconstruction = self.output(x)
        return reconstruction

class VAE(nn.Module):
    def __init__(self, input_dim, num_hidden_layers, num_hidden_units, latent_dim, output_dim, dropout_rate=0.0):
        super().__init__()
        self.encoder = Encoder(input_dim, num_hidden_layers, num_hidden_units, latent_dim, dropout_rate)
        self.decoder = Decoder(latent_dim, num_hidden_layers, num_hidden_units, output_dim, dropout_rate)
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

    def train_model(self, loss_function, train_dataloader, val_dataloader, num_epochs, learning_rate, patience):
        device = next(self.parameters()).device
        print("beginning training")
        wandb.log({"test_log": 1})

        model_path = os.path.join(model_dir, "best_model.pth")
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULE_STEP_SIZE, gamma=SCHEDULE_GAMMA)

        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        bad_epochs = 0
        EPOCHS_TRAINED = 0

        for epoch in range(num_epochs):
            EPOCHS_TRAINED += 1
            epoch_loss = 0
            self.train()

            # training loop
            for batch in train_dataloader:
                # Move all batch tensors to the correct device
                for key in batch:
                    batch[key] = batch[key].to(device)

                optimizer.zero_grad()

                # Forward pass (on junc_ratio, which might be zero-filled or LeafletFA)
                reconstruction, mean, log_vars = self.forward(batch["junc_ratio"])
                concentration = torch.exp(self.log_concentration)

                # For logging
                probabilities = torch.sigmoid(reconstruction).clamp(1e-6, 1 - 1e-6)
                alpha = probabilities * concentration
                beta = (1 - probabilities) * concentration

                #If "MaskOut" is being used, we'll pass the mask, otherwise pass None
                mask_tensor = None
                if MISSING_DATA_METHOD == "MaskOut":
                    mask_tensor = batch["junc_ratio_NaN_mask"]

                # compute loss and backpropagate
                loss = loss_function(
                    logits=reconstruction,
                    junction_counts=batch["cell_by_junction_matrix"],
                    mean=mean,
                    log_vars=log_vars,
                    n_cluster_counts=batch["cell_by_cluster_matrix"],
                    n=len(train_dataloader.dataset),
                    k=len(train_dataloader),
                    concentration=concentration,
                    mask=mask_tensor  # pass the mask or None
                )
                loss.backward()
                epoch_loss += loss.item()
                optimizer.step()

            # average training loss
            train_epoch_loss = epoch_loss / len(train_dataloader)
            train_losses.append(train_epoch_loss)
            scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
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
                f"concentration = {concentration.item():.4f}, mean alpha = {alpha.mean().item():.4f}, mean beta = {beta.mean().item():.4f}",
                flush=True
            )

            plot_latent_space(self, atse_anndata, output_dir, epoch+1)

            # validation loop
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    for key in batch:
                        batch[key] = batch[key].to(device)
                    reconstruction, mean, log_vars = self.forward(batch["junc_ratio"])
                    concentration = torch.exp(self.log_concentration)

                    mask_tensor = None
                    if MISSING_DATA_METHOD == "MaskOut":
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
            wandb.log({"validation_loss": val_epoch_loss})
            print(f"validation loss = {val_epoch_loss:.4f}", flush=True)

            # early stopping check
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                torch.save(self.state_dict(), model_path)
                bad_epochs = 0
            else:
                bad_epochs += 1

            if bad_epochs >= patience:
                print("Early stopping triggered. Ran out of patience", flush=True)
                break

        # load best model weights
        self.load_state_dict(torch.load(model_path))

        # save the number of epochs trained in the parameters file
        with open(params_file, "r+") as f:
            data = json.load(f)
            data["EPOCHS_TRAINED"] = EPOCHS_TRAINED
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()

        return train_losses, val_losses

    def get_latent_rep(self, layer):
        self.encoder.eval()
        layer = layer.to(next(self.parameters()).device)
        with torch.no_grad():
            mean, log_vars = self.encoder(layer)
            z = self.reparametrize(mean, log_vars)
        return z.cpu().numpy()

# ------------------------------
# Loss Functions — incorporate a 'mask' parameter
# ------------------------------
def binomial_loss_function(
    logits, junction_counts, mean, log_vars, n_cluster_counts,
    n, k, concentration=0, mask=None
):

    #If mask is not None (we are masking NaNs), we apply it to skip missing features in the log-likelihood. If mask is None, we compute over all entries.

    if mask is not None:
        #do the masking
        logits = logits[mask]
        junction_counts = junction_counts[mask]
        n_cluster_counts = n_cluster_counts[mask]
    #otherwise, we do not mask, use all entries

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
    #If mask is not None (we are masking NaNs), we apply it to skip missing features in the log-likelihood. If mask is None, we compute over all entries.

    if mask is not None:
        #do the masking
        logits = logits[mask]
        junction_counts = junction_counts[mask]
        n_cluster_counts = n_cluster_counts[mask]
    #otherwise, we do not mask, use all entries

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
# Plotting Helper Functions
# ------------------------------
def plot_losses(train_losses, val_losses, output_dir):
    fig, ax = plt.subplots()
    ax.plot(train_losses, label='Training Loss')
    ax.plot(val_losses, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    loss_plot_path = os.path.join(output_dir, "example_plot.png")
    fig.savefig(loss_plot_path)
    plt.close(fig)
    wandb.log({"loss_plot": wandb.Image(loss_plot_path)})

def plot_latent_space(model, atse_anndata, output_dir, epoch):
    if epoch % 10 != 0:
        return

    if issparse(atse_anndata.layers['junc_ratio']):
        junc_ratio = torch.tensor(
            atse_anndata.layers['junc_ratio'].toarray(), dtype=torch.float32
        )
    else:
        junc_ratio = torch.tensor(
            atse_anndata.layers['junc_ratio'], dtype=torch.float32
        )

    latent_reps = model.get_latent_rep(junc_ratio)

    # prepare cell type information and labels
    cell_types = atse_anndata.obs['cell_type_grouped']
    labels = cell_types.astype('category').cat.codes

    # Dimensionality reduction based on TYPE_OF_PLOT
    if TYPE_OF_PLOT == "UMAP":
        reducer = umap.UMAP(n_components=2)
        embedding = reducer.fit_transform(latent_reps)
        xlabel = "UMAP 1"
        ylabel = "UMAP 2"
        title = "UMAP of Latent Space"
        plot_path = os.path.join(output_dir, "umap_latent_space.png")
        wandb_key = f"umap_plot_epoch_{epoch}"
    elif TYPE_OF_PLOT == "PCA":
        reducer = PCA(n_components=2)
        embedding = reducer.fit_transform(latent_reps)
        xlabel = "Principal Component 1"
        ylabel = "Principal Component 2"
        title = "PCA of Latent Space"
        plot_path = os.path.join(output_dir, "pca_latent_space.png")
        wandb_key = f"pca_plot_epoch_{epoch}"
    else:
        raise ValueError("TYPE_OF_PLOT must be either 'UMAP' or 'PCA'.")

    # create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(
        embedding[:, 0], embedding[:, 1],
        c=labels,
        cmap='tab10', alpha=0.2, s=2
    )
    legend_labels = cell_types.astype('category').cat.categories
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=plt.cm.tab10(i / len(legend_labels)), markersize=10)
        for i in range(len(legend_labels))
    ]
    plt.legend(legend_handles, legend_labels, title="Cell Type Group",
               bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)

    if len(set(labels)) > 1:
        sil_score = silhouette_score(latent_reps, labels)
        wandb.log({"silhouette_score": sil_score})

    if len(set(labels)) > 1:
        cell_type_silhouette = silhouette_samples(latent_reps, labels)
        obs_df = atse_anndata.obs.copy()
        obs_df['Cell Type Silhouette'] = cell_type_silhouette
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='cell_type_grouped', y='Cell Type Silhouette', data=obs_df)
        plt.title('Cell Type Silhouette Scores by Cell Type Grouped (MultiVI-Splice Latent Space)')
        plt.xlabel('Cell Type Grouped')
        plt.ylabel('Silhouette Score')
        plt.xticks(rotation=45, ha='right')
        silhouette_plot_path = os.path.join(output_dir, f"silhouette_boxplot_epoch_{epoch}.png")
        plt.savefig(silhouette_plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        wandb.log({f"silhouette_boxplot_epoch_{epoch}": wandb.Image(silhouette_plot_path)})

    plt.close()
    wandb.log({wandb_key: wandb.Image(plot_path)})

# ------------------------------
# Execution: DataLoader Setup, Model Training, and Plotting
# ------------------------------
full_dataloader, train_dataloader, val_dataloader = construct_input_dataloaders(atse_anndata, BATCH_SIZE)

model = VAE(INPUT_DIM, NUM_HIDDEN_LAYERS, HIDDEN_DIMS, LATENT_DIM, OUTPUT_DIM)
model.to(device)

if LOSS == "Binomial":
    loss_function = binomial_loss_function
elif LOSS == "Beta_Binomial":
    loss_function = beta_binomial_loss_function
else:
    raise ValueError("LOSS must be 'Binomial' or 'Beta_Binomial'")

train_losses, val_losses = model.train_model(
    loss_function,
    train_dataloader,
    val_dataloader,
    NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    patience=PATIENCE
)
plot_losses(train_losses, val_losses, output_dir)

plot_latent_space(model, atse_anndata, output_dir, epoch=EPOCHS_TRAINED)

print("Script execution completed.", flush=True)
wandb.finish()

# %%
