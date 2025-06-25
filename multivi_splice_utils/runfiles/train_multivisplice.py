#!/usr/bin/env python3
"""
train_multivisplice.py: Training script for MULTIVISPLICE model.
Initializes the model, logs to W&B, trains, and saves outputs.
"""
import os
import argparse
import inspect
import wandb
import mudata as mu
import scvi
from pytorch_lightning.loggers import WandbLogger

def get_defaults(sig, exclude):
    return {name: p.default for name, p in sig.parameters.items()
            if name not in exclude and p.default is not inspect._empty}

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train MULTIVISPLICE")
    # paths
    parser.add_argument("--mudata_path", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--model_save_dir", type=str, required=True)
    parser.add_argument("--figure_output_dir", type=str, required=True)

    # inspect defaults
    train_sig = inspect.signature(scvi.model.MULTIVISPLICE.train)
    init_sig = inspect.signature(scvi.model.MULTIVISPLICE.__init__)
    train_defaults = get_defaults(train_sig, exclude=("self","logger"))
    init_defaults = get_defaults(init_sig, exclude=("self","adata"))

    # add init args
    for name, default in init_defaults.items():
        arg_type = int if isinstance(default, int) else (str if isinstance(default, str) else type(default))
        parser.add_argument(f"--{name}", type=arg_type, default=None,
                            help=f"{name} (default={default})")
    # add training args
    for name, default in train_defaults.items():
        arg_type = int if isinstance(default, int) else (str if isinstance(default, str) else type(default))
        parser.add_argument(f"--{name}", type=arg_type, default=None,
                            help=f"{name} (default={default})")
    args = parser.parse_args()

    # prepare dirs
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.figure_output_dir, exist_ok=True)

    # merge kwargs
    model_kwargs = {name: getattr(args, name) for name in init_defaults if getattr(args, name) is not None}
    train_kwargs = {name: getattr(args, name) if getattr(args, name) is not None else default
                    for name, default in train_defaults.items()}

    # W&B init
    run = wandb.init(project="MLCB_SUBMISSION", entity="sv2785-columbia-university",
                     config={**model_kwargs, **train_kwargs, "mudata_path": args.mudata_path})
    wandb_logger = WandbLogger(project=run.project.name, entity=run.entity, log_model=False)

    # load data & setup
    mdata = mu.read_h5mu(args.mudata_path)
    scvi.model.MULTIVISPLICE.setup_mudata(
        mdata,
        batch_key="dataset", size_factor_key="X_library_size",
        rna_layer="length_norm", junc_ratio_layer="junc_ratio",
        atse_counts_layer="cell_by_cluster_matrix",
        junc_counts_layer="cell_by_junction_matrix",
        psi_mask_layer="psi_mask",
        modalities={"rna_layer":"rna","junc_ratio_layer":"splicing"}
    )
    model = scvi.model.MULTIVISPLICE(
        mdata,
        n_genes=(mdata['rna'].var['modality']=='Gene_Expression').sum(),
        n_junctions=(mdata['splicing'].var['modality']=='Splicing').sum(),
        **model_kwargs
    )

    print("Starting training...")
    model.train(logger=wandb_logger, **train_kwargs)

    model.save(args.model_save_dir, overwrite=True)
    print(f"Model saved to {args.model_save_dir}")
    run.finish()