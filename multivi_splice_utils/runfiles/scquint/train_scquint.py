#!/usr/bin/env python
import os
import argparse
import numpy as np
import torch

import mudata as mu
import scanpy as sc
import matplotlib.pyplot as plt
import wandb

# Import your scQuint components as you placed them
# from scquint_vae import run_vae, VAE
from scquint import run_vae, VAE

# ------------------------------
# Defaults (kept minimal)
# ------------------------------
DEFAULT_TRAIN_MDATA = "/path/to/train.h5mu"
DEFAULT_TEST_MDATA  = "/path/to/test.h5mu"
DEFAULT_OUTDIR      = "/path/to/outdir"
DEFAULT_RUN_NAME    = "SCQUINT"

# Model and training knobs (constants here for simplicity)
N_EPOCHS = 300
LR = 1e-2
N_EPOCHS_KL_WARMUP = 20
N_LATENT = 20
N_LAYERS = 1
N_HIDDEN = 128
DROPOUT = 0.25
LINEARITY = "linear"                  # "linear" uses LinearIntronsDecoder
LOSS_INTRONS = "dirichlet-multinomial"
INPUT_TRANSFORM = "log"
USE_CUDA = True

# Data keys
SPLICE_LAYER = "cell_by_junction_matrix"  # junction counts
GROUP_COL = "group_id"                    # intron cluster id
COLOR_KEY = "broad_cell_type"             # used for UMAP coloring if present

parser = argparse.ArgumentParser("Train scQuint on junction counts and output latent UMAPs")
parser.add_argument("--train_mdata_path", type=str, default=DEFAULT_TRAIN_MDATA)
parser.add_argument("--test_mdata_path",  type=str, default=DEFAULT_TEST_MDATA)
parser.add_argument("--outdir",           type=str, default=DEFAULT_OUTDIR)
parser.add_argument("--run_name",         type=str, default=DEFAULT_RUN_NAME)
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)
fig_dir = os.path.join(args.outdir, "figures")
os.makedirs(fig_dir, exist_ok=True)

# ------------------------------
# W&B
# ------------------------------
wandb.init(project="MLCB_SUBMISSION", name=args.run_name, config=dict(
    model="scQuint",
    n_epochs=N_EPOCHS,
    lr=LR,
    n_epochs_kl_warmup=N_EPOCHS_KL_WARMUP,
    n_latent=N_LATENT,
    n_layers=N_LAYERS,
    n_hidden=N_HIDDEN,
    dropout=DROPOUT,
    linearity=LINEARITY,
    loss_introns=LOSS_INTRONS,
    input_transform=INPUT_TRANSFORM,
    splice_layer=SPLICE_LAYER,
    group_col=GROUP_COL,
))

def load_splicing_adata(mdata_path: str):
    m = mu.read_h5mu(mdata_path)  # keep in memory
    ad = m["splicing"].copy()

    X = ad.layers[SPLICE_LAYER]
    X = X.A if hasattr(X, "A") else np.asarray(X)
    ad.X = X.astype(np.float32, copy=False)

    ad.var["intron_group"] = ad.var[GROUP_COL].astype(int)

    # Drop singleton groups to avoid degenerate normalization
    vc = ad.var["intron_group"].value_counts()
    keep = ad.var["intron_group"].isin(vc[vc > 1].index)
    if not bool(np.all(keep.values)):
        ad = ad[:, keep].copy()

    return ad, m

def plot_and_log_umap(latent: np.ndarray, obs, title: str, out_png: str):
    ad_tmp = sc.AnnData(obs=obs.copy())
    ad_tmp.obsm["X_scquint_latent"] = latent
    sc.pp.neighbors(ad_tmp, use_rep="X_scquint_latent")
    sc.tl.umap(ad_tmp)

    plt.figure(figsize=(6, 6))
    color_key = COLOR_KEY if COLOR_KEY in ad_tmp.obs else None
    sc.pl.umap(ad_tmp, color=color_key, frameon=True, show=False, legend_loc=None)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    wandb.log({f"umap/{title}": wandb.Image(out_png)})
    plt.close()

def count_params(model: VAE) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ------------------------------
# TRAIN
# ------------------------------
print(f"Loading TRAIN from {args.train_mdata_path}", flush=True)
ad_tr, m_tr = load_splicing_adata(args.train_mdata_path)

print("Training scQuint…", flush=True)
latent_tr, model = run_vae(
    adata=ad_tr,
    n_epochs=N_EPOCHS,
    use_cuda=USE_CUDA,
    n_latent=N_LATENT,
    n_layers=N_LAYERS,
    dropout_rate=DROPOUT,
    n_hidden=N_HIDDEN,
    lr=LR,
    n_epochs_kl_warmup=N_EPOCHS_KL_WARMUP,
    linearity=LINEARITY,
    loss_introns=LOSS_INTRONS,
    input_transform=INPUT_TRANSFORM,
    regularization_gaussian_std=None,
    sample=True,
    feature_addition=None,
)

n_params = count_params(model)
print(f"Total scQuint parameters: {n_params:,}", flush=True)
wandb.log({"scquint_total_parameters": n_params})

# Train UMAP
train_png = os.path.join(fig_dir, "umap_scquint_train.png")
plot_and_log_umap(latent_tr, ad_tr.obs, title="train", out_png=train_png)

# Save artifacts
model_path = os.path.join(args.outdir, "scquint_model.pt")
lat_tr_path = os.path.join(args.outdir, "scquint_latent_train.npy")
torch.save(model.state_dict(), model_path)
np.save(lat_tr_path, latent_tr)
wandb.log({"model_path": model_path, "latent_train_path": lat_tr_path})

# ------------------------------
# TEST UMAP (latent means only)
# ------------------------------
print(f"Loading TEST from {args.test_mdata_path}", flush=True)
ad_te, m_te = load_splicing_adata(args.test_mdata_path)

device = torch.device("cuda:0") if (USE_CUDA and torch.cuda.is_available()) else torch.device("cpu")
model.to(device).eval()

Zs = []
bs = 2048
X = ad_te.X
n = X.shape[0]
with torch.inference_mode():
    for start in range(0, n, bs):
        stop = min(start + bs, n)
        xb = torch.tensor(X[start:stop], device=device, dtype=torch.float32)
        zb = model.sample_from_posterior_z(xb, give_mean=True)
        Zs.append(zb.detach().cpu().numpy())
latent_te = np.concatenate(Zs, axis=0)

test_png = os.path.join(fig_dir, "umap_scquint_test.png")
plot_and_log_umap(latent_te, ad_te.obs, title="test", out_png=test_png)

lat_te_path = os.path.join(args.outdir, "scquint_latent_test.npy")
np.save(lat_te_path, latent_te)
wandb.log({"latent_test_path": lat_te_path})

wandb.finish()
