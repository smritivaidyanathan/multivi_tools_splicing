#!/bin/bash
#SBATCH --job-name=spliceVAESweep
#SBATCH --mem=300G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=wandb_agent_%j.out
#SBATCH --error=wandb_agent_%j.err

# === [Optional: Load modules, activate conda env, etc.] ===
# source activate scvi-env

# === [Run the W&B agent for your sweep] ===
wandb agent sv2785-columbia-university/splicing_vae_project/nhjgsl4t