#!/usr/bin/env python
import os
import argparse
import numpy as np

import mudata as mu
import scanpy as sc
import scvi
import torch

import wandb
from pytorch_lightning.loggers import WandbLogger

from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

# ------------------------------
# Defaults (mirroring your prior setup as closely as possible)
# ------------------------------
DEFAULT_TRAIN_MDATA = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/072025/train_70_30_20250730_subsetMAX4JUNC.h5mu"
DEFAULT_TEST_MDATA  = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/072025/test_30_70_20250730_subsetMAX4JUNC.h5mu"
DEFAULT_MODEL_DIR   = "/gpfs/commons/home/svaidyanathan/scvi_age/model"
DEFAULT_RUN_NAME    = "SCVI_age_only"

# Hyperparams aligned with your MultiVISplice script where applicable
MAX_EPOCHS = 500
LR = 1e-5
BATCH_SIZE = 256
N_EPOCHS_KL_WARMUP = 100
WEIGHT_DECAY = 1e-3
EARLY_STOPPING_PATIENCE = 50
N_LATENT = 25  # to roughly match your latent size

# Data keys (keep consistent with your previous script)
RNA_LAYER = "length_norm"          # you asked to keep the same arguments you used before
SIZE_FACTOR_KEY = "X_library_size" # consistent with earlier setup
AGE_KEY = "age_numeric"            # used for regression

parser = argparse.ArgumentParser("Train SCVI on RNA only and run age regression on train & test")
parser.add_argument("--train_mdata_path", type=str, default=DEFAULT_TRAIN_MDATA)
parser.add_argument("--test_mdata_path",  type=str, default=DEFAULT_TEST_MDATA)
parser.add_argument("--model_dir",        type=str, default=DEFAULT_MODEL_DIR)
parser.add_argument("--run_name",         type=str, default=DEFAULT_RUN_NAME)
args = parser.parse_args()

os.makedirs(args.model_dir, exist_ok=True)

# ------------------------------
# Weights & Biases
# ------------------------------
wandb.init(project="MLCB_SUBMISSION", name=args.run_name, config=dict(
    model="SCVI",
    max_epochs=MAX_EPOCHS,
    lr=LR,
    batch_size=BATCH_SIZE,
    n_epochs_kl_warmup=N_EPOCHS_KL_WARMUP,
    weight_decay=WEIGHT_DECAY,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    n_latent=N_LATENT,
    rna_layer=RNA_LAYER,
    size_factor_key=SIZE_FACTOR_KEY,
))
wandb_logger = WandbLogger(project="MLCB_SUBMISSION", name=args.run_name, log_model=False)

# ------------------------------
# Load TRAIN
# ------------------------------
# print(f"Loading TRAIN MuData from {args.train_mdata_path}")
# mdata_tr = mu.read_h5mu(args.train_mdata_path, backed = True)
# print(mdata_tr)
# ad_tr = mdata_tr["rna"]
# ad_tr.obs[SIZE_FACTOR_KEY] = mdata_tr.obsm[SIZE_FACTOR_KEY]


# SCVI anndata setup (keeping your keys)
# print("Setting up AnnData for SCVI (TRAIN)…")
# scvi.model.SCVI.setup_anndata(
#     ad_tr,
#     layer=RNA_LAYER,
#     batch_key=None,
#     size_factor_key=SIZE_FACTOR_KEY,
# )

# ------------------------------
# Build & Train SCVI
# ------------------------------
print("Initializing SCVI…")
# model = scvi.model.SCVI(ad_tr, n_latent=N_LATENT)

# Log parameter count
# total_params = sum(p.numel() for p in model.module.parameters())
# print(f"Total SCVI parameters: {total_params:,}")
# wandb.log({"scvi_total_parameters": total_params})

# print("Training SCVI…")
# model.train(
#     max_epochs=MAX_EPOCHS,
#     batch_size=BATCH_SIZE,
#     early_stopping=True,
#     logger=wandb_logger,
# )

# Save model
# print("Saving model…")
# model.save(args.model_dir, overwrite=True)
# wandb.log({"model_saved_to": args.model_dir})



# ------------------------------
# Helper: run age regression on an AnnData using a (possibly queried) model
# ------------------------------
def age_regression_r2(scvi_model, adata, split_name: str):
    assert AGE_KEY in adata.obs, f"{AGE_KEY} not found in .obs"
    Z = scvi_model.get_latent_representation(adata=adata)

    ages = adata.obs[AGE_KEY].astype(float).to_numpy()
    X = StandardScaler().fit_transform(Z)

    X_tr, X_ev, y_tr, y_ev = train_test_split(X, ages, test_size=0.2, random_state=0)
    ridge = RidgeCV(alphas=np.logspace(-2, 3, 20), cv=5).fit(X_tr, y_tr)
    r2 = ridge.score(X_ev, y_ev)
    print(f"[{split_name}] Age regression R²: {r2:.4f}")
    wandb.log({f"{split_name}/age_r2": r2})
    return r2

# ------------------------------
# Evaluate on TRAIN (in-sample latent)
# ------------------------------
# print("Evaluating age regression on TRAIN…")
# age_regression_r2(model, ad_tr, "train")
# del ad_tr, mdata_tr
# torch.cuda.empty_cache()
# ------------------------------
# Evaluate on TEST via query model
# ------------------------------
print(f"\nLoading TEST MuData from {args.test_mdata_path}")
mdata_te = mu.read_h5mu(args.test_mdata_path, backed = True)
ad_te = mdata_te["rna"]
ad_te.obs[SIZE_FACTOR_KEY] = mdata_te.obsm[SIZE_FACTOR_KEY]

print("Loading model")
model = scvi.model.SCVI.load(args.model_dir, ad_te)


print("Preparing TEST AnnData using query model…")

# (Optional) We do not fine-tune; we only extract latents on TEST
print("Evaluating age regression on TEST…")
age_regression_r2(model, ad_te, "test")

print("Done. Finishing W&B.")
wandb.finish()
