#!/usr/bin/env python3
"""Train MULTIVISPLICE with introspected defaults and W&B logging"""
import os
import inspect
import argparse
import wandb
import mudata as mu
import scvi
from pytorch_lightning.loggers import WandbLogger

# ------------------------------
# 0. Default Paths (can be overridden via CLI)
# ------------------------------
DEFAULT_MUDATA_PATH = None  # required
DEFAULT_RUN_DIR = None      # required
DEFAULT_MODEL_SAVE_DIR = None  # required
DEFAULT_FIGURE_DIR = None     # required

# ------------------------------
# 1. Introspect model and train defaults
# ------------------------------
# signature for train()
train_sig = inspect.signature(scvi.model.MULTIVISPLICE.train)
train_defaults = {
    name: param.default
    for name, param in train_sig.parameters.items()
    if name not in ("self", "logger") and param.default is not inspect._empty
}
# signature for __init__
init_sig = inspect.signature(scvi.model.MULTIVISPLICE.__init__)
init_defaults = {
    name: param.default
    for name, param in init_sig.parameters.items()
    if name not in ("self", "adata") and param.default is not inspect._empty
}

# ------------------------------
# 2. Build argparse
# ------------------------------
parser = argparse.ArgumentParser("Train MULTIVISPLICE")
# required paths
parser.add_argument(
    "--mudata_path", type=str, required=True,
    help="Path to input .h5mu"
)
parser.add_argument(
    "--run_dir", type=str, required=True,
    help="Working directory for logs and outputs"
)
parser.add_argument(
    "--model_save_dir", type=str, required=True,
    help="Directory to save trained model"
)
parser.add_argument(
    "--figure_output_dir", type=str, required=True,
    help="Directory to save figures"
)
# model init params
for name, default in init_defaults.items():
    if name == 'n_latent':
        parser.add_argument(
            '--n_latent', type=int, default=None,
            help=f"Dimensionality of latent space (init default={default!r})"
        )
    else:
        arg_type = (lambda x: x.lower() in ("true","1")) if isinstance(default, bool) else type(default)
        parser.add_argument(
            f"--{name}", type=arg_type, default=None,
            help=f"{name} (init default={default!r})"
        )
# training params
for name, default in train_defaults.items():
    arg_type = (lambda x: x.lower() in ("true","1")) if isinstance(default, bool) else type(default)
    parser.add_argument(
        f"--{name}", type=arg_type, default=None,
        help=f"{name} (train default={default!r})"
    )
args = parser.parse_args()

# ------------------------------
# 3. Merge CLI args with defaults
# ------------------------------
# set paths
def _require(arg, name):
    val = getattr(args, name)
    if val is None:
        raise ValueError(f"Missing required argument: --{name}")
    return val

MUDATA_PATH = _require(args, 'mudata_path')
RUN_DIR = _require(args, 'run_dir')
MODEL_SAVE_DIR = _require(args, 'model_save_dir')
FIGURE_OUTPUT_DIR = _require(args, 'figure_output_dir')

# build model init kwargs
nmodel_kwargs = {}
if args.n_latent is not None:
    nmodel_kwargs['n_latent'] = args.n_latent
for name, default in init_defaults.items():
    if name == 'n_latent':
        continue
    val = getattr(args, name)
    if val is not None:
        nmodel_kwargs[name] = val
# build train kwargs
ntrain_kwargs = {
    name: getattr(args, name) if getattr(args, name) is not None else default
    for name, default in train_defaults.items()
}

# ------------------------------
# 4. Initialize W&B
# ------------------------------
full_config = {
    'mudata_path': MUDATA_PATH,
    'run_dir': RUN_DIR,
    'model_save_dir': MODEL_SAVE_DIR,
    'figure_output_dir': FIGURE_OUTPUT_DIR,
    **nmodel_kwargs,
    **ntrain_kwargs,
}
run = wandb.init(
    project='multivisplice',
    config=full_config,
    dir=RUN_DIR
)
wandb_logger = WandbLogger(experiment=run)

# ------------------------------
# 5. Prepare output directories
# ------------------------------
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)
print(f"[INFO] Output dirs: {MODEL_SAVE_DIR}, {FIGURE_OUTPUT_DIR}")

# ------------------------------
# 6. Load data & init model
# ------------------------------
print(f"[INFO] Loading MuData from {MUDATA_PATH}")
mdata = mu.read_h5mu(MUDATA_PATH)

print("[INFO] Initializing model with:", nmodel_kwargs)
model = scvi.model.MULTIVISPLICE(mdata, **nmodel_kwargs)

# ------------------------------
# 7. Train & save
# ------------------------------
print("[INFO] Starting training with:", ntrain_kwargs)
model.train(logger=wandb_logger, **ntrain_kwargs)
print("[INFO] Training complete")

print(f"[INFO] Saving model to {MODEL_SAVE_DIR}")
model.save(MODEL_SAVE_DIR, overwrite=True)

# ------------------------------
# 8. Finish W&B
# ------------------------------
run.finish()
print("[INFO] W&B finished")
