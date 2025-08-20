#!/usr/bin/env bash
# run_splicevipartialvae_sweep.sh
# Driver script to submit a SpliceVI PartialVAE sweep via a job template + --export pattern

### ─── USER CONFIG ─────────────────────────────────────────────────────────
# Default data & hyperparameters (override in-script or via sbatch --export)
TRAIN_ADATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/072025/train_70_30_ge_splice_combined_20250730_164104.h5mu"
TEST_ADATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/072025/test_30_70_ge_splice_combined_20250730_164104.h5mu"
MASKED_TEST_ADATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/072025/MASKED_0.2_test_30_70_ge_splice_combined_20250730_164104.h5mu"


DROPOUT_RATE=0.01
SPLICE_LIKELIHOOD="dirichlet_multinomial"
MAX_EPOCHS=500
LR=1e-5
BATCH_SIZE=256
N_EPOCHS_KL_WARMUP=50
SIMULATED=true
LATENT_DIM=20
FORWARD_STYLE="scatter"

# Sweep over multiple code dimensions
CODE_DIMS=(32)
NUM_WEIGHT_VECTORS_LIST=(5)



# Conda & script location
CONDA_BASE="/gpfs/commons/home/svaidyanathan/miniconda3"
ENV_NAME="scvi-env"
SCRIPT_PATH="/gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/splicevi_utils/runfiles/splicevipartialvae_realdata.py"

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
#SBATCH --mem=300G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00

# Debug prints
echo "→ Job: $JOB_NAME"
echo "   TRAIN_ADATA_PATH= $TRAIN_ADATA_PATH"
echo "   TEST_ADATA_PATH= $TEST_ADATA_PATH"
echo "   MASKED_TEST_ADATA_PATH= $MASKED_TEST_ADATA_PATH"
echo "   ENCODER_TYPE= $ENCODER_TYPE"
echo "   POOL_MODE= $POOL_MODE"
echo "   CODE_DIM= $CODE_DIM"
echo "   LATENT_DIM= $LATENT_DIM"
echo "   IMPUTEDENCODER= $IMPUTEDENCODER"
echo "   NUM_WEIGHT_VECTORS= $NUM_WEIGHT_VECTORS"
echo "   FORWARD_STYLE= $FORWARD_STYLE"


# Activate Conda environment
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Run the Python script with CLI flags built from env vars
python "$SCRIPT_PATH" \
  ${TRAIN_ADATA_PATH:+--train_adata_path "$TRAIN_ADATA_PATH"} \
  ${TEST_ADATA_PATH:+--test_adata_path "$TEST_ADATA_PATH"} \
  ${MASKED_TEST_ADATA_PATH:+--masked_test_adata_path "$MASKED_TEST_ADATA_PATH"} \
  ${MODEL_DIR:+--model_dir "$MODEL_DIR"} \
  ${FIG_DIR:+--fig_dir "$FIG_DIR"} \
  ${DROPOUT_RATE:+--dropout_rate "$DROPOUT_RATE"} \
  ${SPLICE_LIKELIHOOD:+--splice_likelihood "$SPLICE_LIKELIHOOD"} \
  ${MAX_EPOCHS:+--max_epochs "$MAX_EPOCHS"} \
  ${LR:+--lr "$LR"} \
  ${BATCH_SIZE:+--batch_size "$BATCH_SIZE"} \
  ${N_EPOCHS_KL_WARMUP:+--n_epochs_kl_warmup "$N_EPOCHS_KL_WARMUP"} \
  ${SIMULATED:+--simulated} \
  ${CODE_DIM:+--code_dim "$CODE_DIM"} \
  ${LATENT_DIM:+--latent_dim "$LATENT_DIM"} \
  ${ENCODER_TYPE:+--encoder_type "$ENCODER_TYPE"} \
  ${POOL_MODE:+--pool_mode "$POOL_MODE"} \
  ${FORWARD_STYLE:+--forward_style "$FORWARD_STYLE"} \
  ${IMPUTEDENCODER:+--imputedencoder} \
  ${NUM_WEIGHT_VECTORS:+--num_weight_vectors "$NUM_WEIGHT_VECTORS"}

EOF

chmod +x "$BATCH_RUN_DIR/job_template.sh"

echo "→ Job template written. Submitting sweep jobs..."

# 2) Define sweep configurations
ENCODER_TYPES=(
  "PartialEncoderEDDI"
  # "PartialEncoderEDDI"
  # "PartialEncoderWeightedSumEDDIMultiWeight"
  # "PartialEncoderWeightedSumEDDIMultiWeightATSE"
)
POOL_MODES=(
  "sum"
  # "mean"
  # ""
  # ""
)

# 3) Loop and submit
for CODE_DIM in "${CODE_DIMS[@]}"; do
  for i in "${!ENCODER_TYPES[@]}"; do
    ENCODER_TYPE="${ENCODER_TYPES[$i]}"
    POOL_MODE="${POOL_MODES[$i]}"
    

    # heads are only relevant for MultiWeight encoders
    if [[ "$ENCODER_TYPE" == "PartialEncoderWeightedSumEDDIMultiWeight" || \
          "$ENCODER_TYPE" == "PartialEncoderWeightedSumEDDIMultiWeightATSE" ]]; then
      SWEEP_WEIGHTS=("${NUM_WEIGHT_VECTORS_LIST[@]}")
    else
      SWEEP_WEIGHTS=("")  # run once without flag
    fi

    for NUM_WEIGHT_VECTORS in "${SWEEP_WEIGHTS[@]}"; do
      if [ "$ENCODER_TYPE" = "PartialEncoderImpute" ]; then
        IMPUTEDENCODER=true
      else
        IMPUTEDENCODER=
      fi

      JOB_NAME="sweep_REAL_cd${CODE_DIM}_ld=${LATENT_DIM}_lr=${LR}_${i}_${FORWARD_STYLE}_${ENCODER_TYPE}"
      [ -n "$POOL_MODE" ]            && JOB_NAME+="_pool=${POOL_MODE}"
      [ -n "$NUM_WEIGHT_VECTORS" ]   && JOB_NAME+="_W=${NUM_WEIGHT_VECTORS}"

      JOB_DIR="$BATCH_RUN_DIR/$JOB_NAME"
      mkdir -p "$JOB_DIR/models" "$JOB_DIR/figures"
      MODEL_DIR="$JOB_DIR/models"
      FIG_DIR="$JOB_DIR/figures"

      echo "→ Submitting: $JOB_NAME"
      sbatch \
        --job-name="$JOB_NAME" \
        --output="$JOB_DIR/slurm_%j.out" \
        --error="$JOB_DIR/slurm_%j.err" \
        --mem=300G \
        --partition=gpu \
        --gres=gpu:1 \
        --time=40:00:00 \
        --export=\
JOB_NAME="$JOB_NAME",\
JOB_DIR="$JOB_DIR",\
CONDA_BASE="$CONDA_BASE",\
ENV_NAME="$ENV_NAME",\
SCRIPT_PATH="$SCRIPT_PATH",\
TRAIN_ADATA_PATH="$TRAIN_ADATA_PATH",\
TEST_ADATA_PATH="$TEST_ADATA_PATH",\
MASKED_TEST_ADATA_PATH="$MASKED_TEST_ADATA_PATH",\
MODEL_DIR="$MODEL_DIR",\
FIG_DIR="$FIG_DIR",\
DROPOUT_RATE="$DROPOUT_RATE",\
SPLICE_LIKELIHOOD="$SPLICE_LIKELIHOOD",\
MAX_EPOCHS="$MAX_EPOCHS",\
LR="$LR",\
BATCH_SIZE="$BATCH_SIZE",\
N_EPOCHS_KL_WARMUP="$N_EPOCHS_KL_WARMUP",\
SIMULATED="$SIMULATED",\
CODE_DIM="$CODE_DIM",\
LATENT_DIM="$LATENT_DIM",\
ENCODER_TYPE="$ENCODER_TYPE",\
FORWARD_STYLE="$FORWARD_STYLE",\
POOL_MODE="$POOL_MODE",\
IMPUTEDENCODER="$IMPUTEDENCODER",\
NUM_WEIGHT_VECTORS="$NUM_WEIGHT_VECTORS" \
        "$BATCH_RUN_DIR/job_template.sh"
    done
  done
done


echo "→ All sweep jobs (code_dim ∈ ${CODE_DIMS[*]}) submitted."
echo "→ Monitor with: squeue -u $(whoami)"
echo "→ Logs and outputs in: $BATCH_RUN_DIR"
