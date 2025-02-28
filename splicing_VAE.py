# %% [markdown]
# #Notebook for our single modality splicing VAE (SplicingVAE)

# %%
import os
import sys
import datetime
import matplotlib.pyplot as plt

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = os.path.join(os.getcwd(), "splice_vae_job_outputs", timestamp)
os.makedirs(output_dir, exist_ok=True)

model_dir = os.path.join(output_dir, "model")
os.makedirs(model_dir, exist_ok=True)

sys.stdout = open(os.path.join(output_dir, "output.log"), "w")
sys.stderr = open(os.path.join(output_dir, "error.log"), "w")

print(f"Outputs are being saved in: {output_dir}")
print(f"Model will be saved in: {model_dir}")

# %%
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
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from scipy.sparse import issparse

atse_anndata = ad.read_h5ad('/gpfs/commons/groups/knowles_lab/Karin/TMS_MODELING/DATA_FILES/BRAIN_ONLY/02072025/TMS_Anndata_ATSE_counts_with_waypoints_20250207_153520.h5ad')

# %%
atse_anndata.var.shape[0]

# %% [markdown]
# AnnDataSet Class Here!

# %%
class AnnDataDataset(Dataset):
    def __init__(self, layer_tensors):
        self.layer_tensors = layer_tensors
        self.num_samples = list(layer_tensors.values())[0].shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {layer_key: tensor[idx] for layer_key, tensor in self.layer_tensors.items()}

# %%
import json

LEARNING_RATE = 0.001
NUM_EPOCHS = 300
BATCH_SIZE = 3000
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')
INPUT_DIM = atse_anndata.var.shape[0]
NUM_HIDDEN_LAYERS = 1
HIDDEN_DIMS = [128, 64]
LATENT_DIM = 30
OUTPUT_DIM = INPUT_DIM
PATIENCE = 10
SCHEDULE_STEP_SIZE = 300
SCHEDULE_GAMMA = 0.1

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
    "SCHEDULE_STEP_SIZE":SCHEDULE_STEP_SIZE, 
    "SCHEDULE_GAMMA": SCHEDULE_GAMMA
}

params_file = os.path.join(output_dir, "parameters.json")
with open(params_file, "w") as f:
    json.dump(params, f, indent=4)

# %% [markdown]
# Next, we define our model, Loss function, and helper methods. 

# %%
class Encoder(nn.Module):
    def __init__(self, input_dim, num_hidden_layers, num_hidden_units, latent_dim, dropout_rate = 0.0):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(input_dim, num_hidden_units[0]), nn.ReLU(), nn.Dropout(dropout_rate))
        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden_layers):
            self.hidden_layers.append(nn.Sequential(nn.Linear(num_hidden_units[i], num_hidden_units[i+1]), nn.ReLU(), nn.Dropout(dropout_rate)))
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
    def __init__(self, z_dim, num_hidden_layers, num_hidden_units, output_dim, dropout_rate = 0.0):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(z_dim, num_hidden_units[-1]), nn.ReLU(), nn.Dropout(dropout_rate))
        self.hidden_layers = nn.ModuleList()
        for i in reversed(range(num_hidden_layers)):
            self.hidden_layers.append(nn.Sequential(nn.Linear(num_hidden_units[i+1], num_hidden_units[i]), nn.ReLU(), nn.Dropout(dropout_rate)))
        self.output = nn.Linear(num_hidden_units[0], output_dim)

    def forward(self, x):
        x = self.input(x)
        for layer in self.hidden_layers:
             x = layer(x)
        reconstruction = self.output(x)
        return reconstruction
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
def binomial_loss_function(logits, junction_counts, mean, log_vars, n_cluster_counts, n, k):
    probabilities = torch.sigmoid(logits)
    log_probabilities = torch.log(probabilities + 1e-10)
    log_complement_probabilities = torch.log(1 - probabilities + 1e-10)
    log_likelihood = (
        junction_counts * log_probabilities +
        (n_cluster_counts - junction_counts) * log_complement_probabilities
    )
    log_likelihood = log_likelihood * (float(n)/float(k))
    reconstruction_loss = -log_likelihood.mean()
    qz = Normal(mean, torch.sqrt(torch.exp(log_vars)))
    pz = Normal(0, 1)
    kl_dive = kl_divergence(qz, pz).sum(dim=1).mean()
    total_loss = reconstruction_loss + (kl_dive)
    return total_loss

def construct_input_dataloaders(atse_anndata, batch_size, validation_split = 0.2):
    layer_tensors = {
        layer_key: torch.tensor(atse_anndata.layers[layer_key].toarray() if issparse(atse_anndata.layers[layer_key]) else atse_anndata.layers[layer_key], dtype=torch.float32)
        for layer_key in list(atse_anndata.layers.keys())[:3]
    }
    dataset = AnnDataDataset(layer_tensors)
    total_size = len(dataset)
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    full_dataset = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return full_dataset, train_dataloader, val_dataloader

class VAE(nn.Module):
    def __init__(self, input_dim, num_hidden_layers, num_hidden_units, latent_dim, output_dim, dropout_rate = 0.0):
        super().__init__()
        self.encoder = Encoder(input_dim, num_hidden_layers, num_hidden_units, latent_dim, dropout_rate)
        self.decoder = Decoder(latent_dim, num_hidden_layers, num_hidden_units, output_dim, dropout_rate)
    
    def reparametrize(self, mean, log_vars):
        std = torch.exp(0.5 * log_vars) 
        eps = torch.randn_like(std)
        return mean + eps * std  
    
    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        mean, log_vars = self.encoder(x)
        z = self.reparametrize(mean, log_vars)
        reconstruction = self.decoder(z)
        reconstruction = reconstruction.to(torch.float32)
        mean = mean.to(torch.float32)
        log_vars = log_vars.to(torch.float32)
        return reconstruction, mean, log_vars
    
    def train_model(self, train_dataloader, val_dataloader, num_epochs, learning_rate, patience):
        device = next(self.parameters()).device
        print("Beginning Training")
        model_path = os.path.join(model_dir, "best_model.pth")
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULE_STEP_SIZE, gamma=SCHEDULE_GAMMA)
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        bad_epochs = 0
        epochs_trained = 0
        for epoch in range(num_epochs):
            epochs_trained += 1
            epoch_loss = 0
            self.train()
            for batch in train_dataloader:
                for key in batch:
                    batch[key] = batch[key].to(device)
                optimizer.zero_grad()
                reconstruction, mean, log_vars = self.forward(batch["junc_ratio"])
                loss = binomial_loss_function(reconstruction, batch["cell_by_junction_matrix"], mean, log_vars, batch["cell_by_cluster_matrix"], len(train_dataloader.dataset), len(train_dataloader))
                loss.backward()
                epoch_loss += loss.item()
                optimizer.step()
            train_epoch_loss = epoch_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}; Train Loss = {train_epoch_loss}")
            train_losses.append(train_epoch_loss)
            scheduler.step()
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    for key in batch:
                        batch[key] = batch[key].to(device)
                    reconstruction, mean, log_vars = self.forward(batch["junc_ratio"])
                    val_batch_loss = binomial_loss_function(reconstruction, batch["cell_by_junction_matrix"], mean, log_vars, batch["cell_by_cluster_matrix"], len(val_dataloader.dataset), len(val_dataloader))
                    val_loss += val_batch_loss.item() 
            val_epoch_loss = val_loss / len(val_dataloader)
            val_losses.append(val_epoch_loss)
            print(f"Validation Loss = {val_epoch_loss}")
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                torch.save(self.state_dict(), model_path)
                bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping triggered. Ran out of patience")
                break
        self.load_state_dict(torch.load(model_path))
        with open(params_file, "r+") as f:
            data = json.load(f)
            data["epochs_trained"] = epochs_trained
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()
        return train_losses, val_losses
    def get_latent_rep(self, layer):
        latent_representations = []
        self.encoder.eval()
        layer = layer.to(next(self.parameters()).device)
        with torch.no_grad():
            mean, log_vars = self.encoder(layer)
            z = self.reparametrize(mean, log_vars)
            latent_representations.append(z)
        latent_representations = torch.cat(latent_representations, dim=0)
        latent_representations_np = latent_representations.cpu().numpy()
        return latent_representations_np

# %%
import matplotlib.pyplot as plt

def plot_losses(train_losses, val_losses, output_dir):
    fig, ax = plt.subplots()
    ax.plot(train_losses, label='Training Loss')
    ax.plot(val_losses, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    fig.savefig(os.path.join(output_dir, "example_plot.png"))
    plt.close(fig)

# %% [markdown]
# Our parameters for the VAE. 

# %%
full_dataloader, train_dataloader, val_dataloader = construct_input_dataloaders(atse_anndata, BATCH_SIZE)

# %% [markdown]
# Now, we can train our model!

# %%
model = VAE(INPUT_DIM, NUM_HIDDEN_LAYERS, HIDDEN_DIMS, LATENT_DIM, OUTPUT_DIM)
if USE_CUDA:
    model.cuda()
model.to(device)
train_losses, val_losses = model.train_model(train_dataloader, val_dataloader, NUM_EPOCHS, learning_rate = LEARNING_RATE, patience=PATIENCE)
plot_losses(train_losses, val_losses, output_dir)

# %% [markdown]
# Getting the latent representation. 

# %%
lr = model.get_latent_rep(torch.tensor(atse_anndata.layers['junc_ratio'].toarray(), dtype=torch.float32))

# %%
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

pca_model = PCA(n_components=2)
print(atse_anndata.obs['cell_type_grouped'].value_counts())
pca_latents = pca_model.fit_transform(lr)
cell_types = atse_anndata.obs['cell_type_grouped']
plt.figure(figsize=(8, 6))
scatter = plt.scatter(pca_latents[:, 0], pca_latents[:, 1], 
                      c=cell_types.astype('category').cat.codes, 
                      cmap='tab10', alpha=0.2, s=2)
legend_labels = cell_types.astype('category').cat.categories
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=plt.cm.tab10(i / len(legend_labels)), 
                             markersize=10) for i in range(len(legend_labels))]
plt.legend(legend_handles, legend_labels, title="cell type group", 
           bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel("principal component 1")
plt.ylabel("principal component 2")
plt.title("pca of latent space")
pca_plot_path = os.path.join(output_dir, "pca_latent_space.png")
plt.savefig(pca_plot_path, bbox_inches='tight', dpi=300)
plt.close()
print("Script execution completed.")
