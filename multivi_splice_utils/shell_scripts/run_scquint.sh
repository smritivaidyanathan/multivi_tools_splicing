#!/usr/bin/env bash
#SBATCH --job-name=scquint_umap
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# run_scquint_umap.sh
# Trains scQuint on junction counts and saves UMAPs of the latent space.

# --- Paths and env ---
CONDA_BASE="/gpfs/commons/home/svaidyanathan/miniconda3"
ENV_NAME="scvi-legacy"

TRAIN_MDATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/072025/train_70_30_20250730_subsetMAX4JUNC.h5mu"
TEST_MDATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/072025/test_30_70_20250730_subsetMAX4JUNC.h5mu"

OUTDIR="/gpfs/commons/home/svaidyanathan/scquint/out"
RUN_NAME="SCQUINT_introns_only_$(date +%Y%m%d_%H%M%S)"

# Path to your script
SCRIPT="/gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/multivi_splice_utils/runfiles/scquint/train_scquint.py"

# --- Activate env ---
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Optional: make W&B robust on clusters
export WANDB__SERVICE_WAIT=300

# --- Run ---
python "$SCRIPT" \
  --train_mdata_path "$TRAIN_MDATA_PATH" \
  --test_mdata_path "$TEST_MDATA_PATH" \
  --outdir "$OUTDIR" \
  --run_name "$RUN_NAME"
