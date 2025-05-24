#!/usr/bin/env bash
# submit_parallel_imputation.sh
# Script to submit multiple parallel imputation jobs

BASE_RUN_DIR="/gpfs/commons/home/kisaev/multivi_tools_splicing/results/imputation"
SCRIPT_PATH="/gpfs/commons/home/kisaev/multivi_tools_splicing/multivi_splice_utils/runfiles/imputation_eval_single.py"

# Timestamped batch run
TS=$(date +"%Y%m%d_%H%M%S")
BATCH_RUN_DIR="${BASE_RUN_DIR}/batch_${TS}"
mkdir -p "${BATCH_RUN_DIR}"

echo "→ Submitting parallel imputation jobs to: ${BATCH_RUN_DIR}"

# Create the job template FIRST
cat > "${BATCH_RUN_DIR}/job_template.sh" << 'EOF'
#!/usr/bin/env bash

echo "→ Imputation Evaluation: RNA=${PCT_RNA}%, Splice=${PCT_SPLICE}%"
echo "→ Output directory: ${IMPUTATION_EVAL_OUTDIR}"

# Conda Environment Setup
CONDA_BASE="/gpfs/commons/home/kisaev/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate scvi-env

# Run Python script for this specific condition
python ${SCRIPT_PATH}
EOF

chmod +x "${BATCH_RUN_DIR}/job_template.sh"

# Define the missing percentage pairs
MISSING_PAIRS=(
    "0.1,0.0"   # RNA only
    "0.0,0.1"   # Splice only  
    "0.1,0.1"   # Both low
    "0.0,0.3"   # Splice medium
    "0.3,0.0"   # RNA medium
    "0.3,0.3"   # Both medium
    "0.0,0.5"   # Splice high
    "0.5,0.0"   # RNA high
    "0.5,0.5"   # Both high
)

# Submit one job per missing condition
for pair in "${MISSING_PAIRS[@]}"; do
    pct_rna=$(echo $pair | cut -d',' -f1)
    pct_splice=$(echo $pair | cut -d',' -f2)
    
    job_name="impute_r${pct_rna}_s${pct_splice}"
    job_dir="${BATCH_RUN_DIR}/${job_name}"
    mkdir -p "${job_dir}/figures"
    
    echo "Submitting job: ${job_name}"
    
    sbatch --job-name="${job_name}" \
           --output="${job_dir}/slurm_%j.out" \
           --error="${job_dir}/slurm_%j.err" \
           --mem=200G \
           --partition=gpu \
           --gres=gpu:1 \
           --time=6:00:00 \
           --export=PCT_RNA=${pct_rna},PCT_SPLICE=${pct_splice},IMPUTATION_EVAL_OUTDIR=${job_dir},SCRIPT_PATH=${SCRIPT_PATH} \
           "${BATCH_RUN_DIR}/job_template.sh"
done

echo "→ All jobs submitted. Monitor with: squeue -u $(whoami)"
echo "→ Results will be in: ${BATCH_RUN_DIR}"