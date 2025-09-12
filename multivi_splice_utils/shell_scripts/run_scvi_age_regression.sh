#!/usr/bin/env bash
#SBATCH --job-name=scvi_age
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# run_scvi_age_regression.sh
# Minimal launcher to train SCVI (RNA only) and run age regression on train & test.

# --- Adjust these to your environment ---
CONDA_BASE="/gpfs/commons/home/svaidyanathan/miniconda3"
ENV_NAME="scvi-env"

TRAIN_MDATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/072025/train_70_30_20250730_subsetMAX4JUNC.h5mu"
TEST_MDATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/072025/test_30_70_20250730_subsetMAX4JUNC.h5mu"
MODEL_DIR="/gpfs/commons/home/svaidyanathan/scvi_age/model"


source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

python /gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/multivi_splice_utils/runfiles/scvi_age_regression.py \
  --train_mdata_path "$TRAIN_MDATA_PATH" \
  --test_mdata_path "$TEST_MDATA_PATH" \
  --model_dir "$MODEL_DIR" \
  --run_name "SCVI_age_only_$(date +%Y%m%d_%H%M%S)"
