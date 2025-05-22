#!/usr/bin/env bash
#SBATCH --job-name=ImputationEval
#SBATCH --output=/gpfs/commons/home/svaidyanathan/imputation_eval_runs/slurm_%j.out
#SBATCH --error=/gpfs/commons/home/svaidyanathan/imputation_eval_runs/slurm_%j.err
#SBATCH --mem=300G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# === Configuration ===
BASE_RUN_DIR="/gpfs/commons/home/svaidyanathan/imputation_eval_runs"
SCRIPT_PATH="/gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/multivi_splice_utils/runfiles/imputation_eval.py"

# === Timestamped Output Folder ===
TS=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${BASE_RUN_DIR}/run_${TS}"
mkdir -p "${RUN_DIR}/figures"

# === Redirect stdout/stderr into run dir (overrides default SLURM logs) ===
exec > "${RUN_DIR}/slurm.out" 2> "${RUN_DIR}/slurm.err"

echo "→ Imputation Evaluation Run"
echo "→ Output directory: ${RUN_DIR}"

# === Conda Environment Setup ===
CONDA_BASE="/gpfs/commons/home/svaidyanathan/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate scvi-env

# === Set environment variable for Python script ===
export IMPUTATION_EVAL_OUTDIR="${RUN_DIR}"

# === Run Python script ===
python "${SCRIPT_PATH}"
