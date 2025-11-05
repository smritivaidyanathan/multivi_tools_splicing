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
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, silhouette_score
from sklearn.linear_model import LogisticRegression


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

UMAP_COLOR_KEY = "broad_cell_type"
MIN_GROUP_N = 25  # skip tiny groups for stable R²
AGE_R2_RECORDS = []  # collects per-pair results across splits


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
    model="LinearSCVI",
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

print(f"Loading TRAIN MuData from {args.train_mdata_path}")
mdata_tr = mu.read_h5mu(args.train_mdata_path, backed = "r")  # in memory for training
ad_tr = mdata_tr["rna"]

# If size factors live in obsm on MuData, mirror into obs so SCVI can see them
if SIZE_FACTOR_KEY in mdata_tr.obsm_keys():
    ad_tr.obs[SIZE_FACTOR_KEY] = mdata_tr.obsm[SIZE_FACTOR_KEY]

# ------------------------------
# SCVI anndata setup (TRAIN)
# ------------------------------
print("Setting up AnnData for SCVI (TRAIN)…")
scvi.model.SCVI_Linear.setup_anndata(
    ad_tr,
    layer=RNA_LAYER,
    batch_key=None,
    size_factor_key=SIZE_FACTOR_KEY,
)

# ------------------------------
# Build & Train SCVI
# ------------------------------
print("Initializing SCVI…")
model = scvi.model.SCVI_Linear(ad_tr, n_latent=N_LATENT)

# Log parameter count
total_params = sum(p.numel() for p in model.module.parameters())
print(f"Total SCVI parameters: {total_params:,}")
wandb.log({"scvi_total_parameters": total_params})

print("Training SCVI…")
model.train(
    max_epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    early_stopping=True,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    plan_kwargs={"weight_decay": WEIGHT_DECAY, "lr": LR, "n_epochs_kl_warmup": N_EPOCHS_KL_WARMUP},
    logger=wandb_logger,
)

print("Saving model…")
model.save(args.model_dir, overwrite=True)
wandb.log({"model_saved_to": args.model_dir})

def evaluate_split_scvi(scvi_model, adata, split_name: str):
    # Choose label keys
    umap_color_key = UMAP_COLOR_KEY
    cell_type_classification_key = (
        "medium_cell_type" if "medium_cell_type" in adata.obs.columns else UMAP_COLOR_KEY
    )

    # Latent
    Z = scvi_model.get_latent_representation(adata=adata)

    # ---------- PCA 90% variance ----------
    n_comp_max = min(Z.shape[0], Z.shape[1])
    pca = PCA(n_components=n_comp_max, svd_solver="full").fit(Z)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    pcs_90 = int(np.searchsorted(cum_var, 0.90) + 1)
    wandb.log({
        f"real-{split_name}-expression/pca_n_components_90var": pcs_90,
        f"real-{split_name}-expression/pca_total_dim": Z.shape[1],
        f"real-{split_name}-expression/pca_var90_ratio": pcs_90 / Z.shape[1],
    })

    # ---------- Silhouette (same style as your SpliceVI) ----------
    try:
        labels_broad = adata.obs[umap_color_key].astype(str).values
        sil_broad = silhouette_score(Z, labels_broad)
        wandb.log({f"real-{split_name}-expression/{umap_color_key}-silhouette_score": sil_broad})
    except Exception:
        pass

    try:
        labels_med = adata.obs[cell_type_classification_key].astype(str).values
        sil_med = silhouette_score(Z, labels_med)
        wandb.log({f"real-{split_name}-expression/{cell_type_classification_key}-silhouette_score": sil_med})
    except Exception:
        pass

    # ---------- Cell type classification (LR) ----------
    if cell_type_classification_key in adata.obs:
        y = adata.obs[cell_type_classification_key].astype(str).values
        X_tr, X_ev, y_tr, y_ev = train_test_split(Z, y, test_size=0.2, random_state=0)
        clf = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
        y_pred = clf.predict(X_ev)
        wandb.log({
            f"real-{split_name}-expression/accuracy": accuracy_score(y_ev, y_pred),
            f"real-{split_name}-expression/precision": precision_score(y_ev, y_pred, average="weighted", zero_division=0),
            f"real-{split_name}-expression/recall": recall_score(y_ev, y_pred, average="weighted", zero_division=0),
            f"real-{split_name}-expression/f1_score": f1_score(y_ev, y_pred, average="weighted", zero_division=0),
        })

    # ---------- Age regression only for ages {3, 18, 24} ----------
    if AGE_KEY in adata.obs:
        ages_full = adata.obs[AGE_KEY].astype(float).values
        target_ages = np.array([3.0, 18.0, 24.0], dtype=float)
        mask_age = np.isin(ages_full, target_ages)

        n_kept = int(mask_age.sum())
        wandb.log({f"real-{split_name}-expression/age_n_cells": n_kept})

        if n_kept >= MIN_GROUP_N:
            ages = ages_full[mask_age]
            Z_use = Z[mask_age, :]
            obs_local = adata.obs.iloc[np.where(mask_age)[0]].copy()

            X_latent = StandardScaler().fit_transform(Z_use)
            X_tr, X_ev, y_tr, y_ev = train_test_split(X_latent, ages, test_size=0.2, random_state=0)

            if np.std(y_tr) > 0.0 and np.std(y_ev) > 0.0:
                ridge = RidgeCV(alphas=np.logspace(-2, 3, 20), cv=5).fit(X_tr, y_tr)
                r2_age = ridge.score(X_ev, y_ev)
                wandb.log({f"real-{split_name}-expression/age_r2": r2_age})

            # per (tissue | cell_type)
            if "tissue" in obs_local and cell_type_classification_key in obs_local:
                tissue_series = obs_local["tissue"].astype(str)
                ct_series = obs_local[cell_type_classification_key].astype(str)
                pair = tissue_series + " | " + ct_series
                pair_unique = pair.unique()

                for p in pair_unique:
                    idx = np.where(pair.values == p)[0]
                    if idx.size < MIN_GROUP_N:
                        continue
                    Zg = X_latent[idx]
                    yg = ages[idx]
                    if np.std(yg) == 0.0:
                        continue

                    Ztr, Zev, ytr, yev = train_test_split(Zg, yg, test_size=0.2, random_state=0)
                    if Ztr.shape[0] < 2 or Zev.shape[0] < 2 or np.std(ytr) == 0.0 or np.std(yev) == 0.0:
                        continue

                    try:
                        rg = RidgeCV(alphas=np.logspace(-2, 3, 20), cv=5).fit(Ztr, ytr)
                        r2g = rg.score(Zev, yev)
                    except Exception:
                        continue

                    AGE_R2_RECORDS.append({
                        "dataset": split_name,          # "train" or "test"
                        "space": "expression",          # match your SpliceVI naming
                        "pair": p,                      # "tissue | celltype"
                        "tissue": p.split(" | ", 1)[0],
                        "cell_type": p.split(" | ", 1)[1],
                        "r2": float(r2g),
                        "n": int(idx.size),             # count after age filter
                    })

# ------------------------------
# Evaluate on TRAIN
# ------------------------------
print("Evaluating metrics on TRAIN…")
evaluate_split_scvi(model, ad_tr, "train")

# Free TRAIN
del ad_tr, mdata_tr
torch.cuda.empty_cache()


# ------------------------------
# Evaluate on TEST via loaded model
# ------------------------------
print(f"\nLoading TEST MuData from {args.test_mdata_path}")
mdata_te = mu.read_h5mu(args.test_mdata_path)
ad_te = mdata_te["rna"]
if SIZE_FACTOR_KEY in mdata_te.obsm_keys():
    ad_te.obs[SIZE_FACTOR_KEY] = mdata_te.obsm[SIZE_FACTOR_KEY]

print("Loading model")
model = scvi.model.SCVI_Linear.load(args.model_dir, ad_te)

print("Evaluating metrics on TEST…")
evaluate_split_scvi(model, ad_te, "test")

# ------------------------------
# Save age R² dataframe identical to SpliceVI style
# ------------------------------
if len(AGE_R2_RECORDS) > 0:
    age_df = pd.DataFrame(AGE_R2_RECORDS)
    csv_path = os.path.join(args.model_dir, f"age_r2_by_tissue_celltype_train_test_target_ages_3_18_24_scvi_{args.run_name}.csv")
    age_df.to_csv(csv_path, index=False)
    wandb.log({
        "age_r2/records_csv_path": csv_path,
        "age_r2/n_records": int(len(age_df)),
    })
else:
    print("No age R² per-pair records collected.")

wandb.finish()