#!/usr/bin/env bash
# run_splicevipartialvae_sweep.sh
# Driver script to submit a SpliceVI PartialVAE sweep via a job template + --export pattern

### ─── USER CONFIG ─────────────────────────────────────────────────────────
# Default data & hyperparameters (override in-script or via sbatch --export)
ADATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/TMS_MODELING/DATA_FILES/SIMULATED/simulated_data_2025-03-12.h5ad"
DROPOUT_RATE=0.01
SPLICE_LIKELIHOOD="dirichlet_multinomial"
MAX_EPOCHS=300
LR=1e-5
BATCH_SIZE=256
N_EPOCHS_KL_WARMUP=50
SIMULATED=true

# Conda & script location
CONDA_BASE="/gpfs/commons/home/svaidyanathan/miniconda3"
ENV_NAME="scvi-env"
SCRIPT_PATH="/gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/splicevi_utils/runfiles/splicevipartialvae.py"

# Batch output directory
BASE_RUN_DIR="/gpfs/commons/home/svaidyanathan/splice_vi_partial_vae_sweep"
TS=$(date +"%Y%m%d_%H%M%S")
BATCH_RUN_DIR="$BASE_RUN_DIR/batch_$TS"
mkdir -p "$BATCH_RUN_DIR"

echo "→ Writing job template in $BATCH_RUN_DIR"

# 1) Create a reusable job template that picks up its config via env vars
cat > "$BATCH_RUN_DIR/job_template.sh" << 'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${JOB_DIR}/slurm_%j.out
#SBATCH --error=${JOB_DIR}/slurm_%j.err
#SBATCH --mem=150G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00

# Debug prints
echo "→ Job: $JOB_NAME"
echo "   ADATA_PATH= $ADATA_PATH"
echo "   ENCODER_TYPE= $ENCODER_TYPE"
echo "   POOL_MODE= $POOL_MODE"
echo "   JUNCTION_INCLUSION= $JUNCTION_INCLUSION"
echo "   IMPUTEDENCODER= $IMPUTEDENCODER"

# Activate Conda environment
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Run the Python script with CLI flags built from env vars
python "$SCRIPT_PATH" \
  ${ADATA_PATH:+--adata_path "$ADATA_PATH"} \
  ${MODEL_DIR:+--model_dir "$MODEL_DIR"} \
  ${FIG_DIR:+--fig_dir "$FIG_DIR"} \
  ${DROPOUT_RATE:+--dropout_rate "$DROPOUT_RATE"} \
  ${SPLICE_LIKELIHOOD:+--splice_likelihood "$SPLICE_LIKELIHOOD"} \
  ${MAX_EPOCHS:+--max_epochs "$MAX_EPOCHS"} \
  ${LR:+--lr "$LR"} \
  ${BATCH_SIZE:+--batch_size "$BATCH_SIZE"} \
  ${N_EPOCHS_KL_WARMUP:+--n_epochs_kl_warmup "$N_EPOCHS_KL_WARMUP"} \
  ${SIMULATED:+--simulated} \
  ${ENCODER_TYPE:+--encoder_type "$ENCODER_TYPE"} \
  ${POOL_MODE:+--pool_mode "$POOL_MODE"} \
  ${JUNCTION_INCLUSION:+--junction_inclusion "$JUNCTION_INCLUSION"} \
  ${IMPUTEDENCODER:+--imputedencoder}
EOF

chmod +x "$BATCH_RUN_DIR/job_template.sh"

echo "→ Job template written. Submitting sweep jobs..."

# 2) Define sweep configurations
ENCODER_TYPES=(
  "PartialEncoder"
  "PartialEncoder"
  "PartialEncoderWeightedSum"
  "PartialEncoderWeightedSum"
  "PartialEncoderImpute"
  "PartialEncoderImpute"
)
POOL_MODES=(
  "sum"
  "mean"
  ""
  ""
  ""
  ""
)
JUNCTION_INCLUSIONS=(
  ""
  ""
  "observed_junctions"
  "all_junctions"
  "all_junctions"
  "observed_junctions"
)

# 3) Loop and submit
for i in "${!ENCODER_TYPES[@]}"; do
  ENCODER_TYPE="${ENCODER_TYPES[$i]}"
  POOL_MODE="${POOL_MODES[$i]}"
  JUNCTION_INCLUSION="${JUNCTION_INCLUSIONS[$i]}"
  
  # Only PartialEncoderImpute needs the imputed flag
  if [ "$ENCODER_TYPE" = "PartialEncoderImpute" ]; then
    IMPUTEDENCODER=true
  else
    IMPUTEDENCODER=
  fi
  
  JOB_NAME="sweep_${i}_${ENCODER_TYPE}"
  [ -n "$POOL_MODE" ] && JOB_NAME+="_pool=${POOL_MODE}"
  [ -n "$JUNCTION_INCLUSION" ] && JOB_NAME+="_jinc=${JUNCTION_INCLUSION}"

  JOB_DIR="$BATCH_RUN_DIR/$JOB_NAME"
  mkdir -p "$JOB_DIR/models" "$JOB_DIR/figures"
  MODEL_DIR="$JOB_DIR/models"
  FIG_DIR="$JOB_DIR/figures"

  echo "→ Submitting: $JOB_NAME"
  sbatch \
    --job-name="$JOB_NAME" \
    --output="$JOB_DIR/slurm_%j.out" \
    --error="$JOB_DIR/slurm_%j.err" \
    --mem=150G \
    --partition=gpu \
    --gres=gpu:1 \
    --time=5:00:00 \
    --export=\
JOB_NAME="$JOB_NAME",\
JOB_DIR="$JOB_DIR",\
CONDA_BASE="$CONDA_BASE",\
ENV_NAME="$ENV_NAME",\
SCRIPT_PATH="$SCRIPT_PATH",\
ADATA_PATH="$ADATA_PATH",\
MODEL_DIR="$MODEL_DIR",\
FIG_DIR="$FIG_DIR",\
DROPOUT_RATE="$DROPOUT_RATE",\
SPLICE_LIKELIHOOD="$SPLICE_LIKELIHOOD",\
MAX_EPOCHS="$MAX_EPOCHS",\
LR="$LR",\
BATCH_SIZE="$BATCH_SIZE",\
N_EPOCHS_KL_WARMUP="$N_EPOCHS_KL_WARMUP",\
SIMULATED="$SIMULATED",\
ENCODER_TYPE="$ENCODER_TYPE",\
POOL_MODE="$POOL_MODE",\
JUNCTION_INCLUSION="$JUNCTION_INCLUSION",\
IMPUTEDENCODER="$IMPUTEDENCODER" \
    "$BATCH_RUN_DIR/job_template.sh"
done

echo "→ All sweep jobs submitted. Monitor with: squeue -u \$(whoami)"
echo "→ Logs and outputs in: $BATCH_RUN_DIR"