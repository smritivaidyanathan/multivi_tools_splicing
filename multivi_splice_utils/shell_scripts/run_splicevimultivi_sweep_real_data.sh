#!/usr/bin/env bash
# run_splicevipartialvae_sweep.sh
# Driver script to submit a SpliceVI PartialVAE sweep via a job template + --export pattern

### ─── USER CONFIG ─────────────────────────────────────────────────────────
# Default data & hyperparameters (override in-script or via sbatch --export)
TRAIN_MDATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/072025/train_70_30_20250730_subsetMAX4JUNC.h5mu"
TEST_MDATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/072025/test_30_70_20250730_subsetMAX4JUNC.h5mu"
MASKED_TEST_MDATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/072025/MASKED_0.2_test_30_70_20250730_subsetMAX4JUNC.h5mu"


DROPOUT_RATE=0.01
SPLICING_LOSS_TYPE="dirichlet_multinomial"   # options: binomial | beta_binomial | dirichlet_multinomial

MAX_EPOCHS=1
LR=1e-5
BATCH_SIZE=256
N_EPOCHS_KL_WARMUP=50
N_LATENT=20                                   # maps to --n_latent
FORWARD_STYLE="scatter"                        # options: per-cell | batched | scatter
MAX_NOBS=-1

# Sweep over multiple code dimensions
CODE_DIMS=(32)
NUM_WEIGHT_VECTORS_LIST=(5)
MODALITY_WEIGHTS="equal"                       # options: equal | cell | universal | concatenate
SPLICING_ARCHITECTURE="partial"                # options: vanilla | partial
EXPRESSION_ARCHITECTURE="vanilla"              # options: vanilla | linear
ENCODER_HIDDEN_DIM=128
H_HIDDEN_DIM=64
ATSE_EMBEDDING_DIMENSION=16
TEMPERATURE_VALUE=-1.0
TEMPERATURE_FIXED=true
POOL_MODE="sum"                                # options: mean | sum (you already sweep this paired with ENCODER_TYPE)

# TRAIN knobs (non-sweep) 
WEIGHT_DECAY=1e-3
EARLY_STOPPING_PATIENCE=50
LR_SCHEDULER_TYPE="plateau"                    # options: plateau | step
LR_FACTOR=0.6
LR_PATIENCE=30
STEP_SIZE=10
GRADIENT_CLIPPING=true
GRADIENT_CLIPPING_MAX_NORM=5.0




# Conda & script location
CONDA_BASE="/gpfs/commons/home/svaidyanathan/miniconda3"
ENV_NAME="scvi-env"
SCRIPT_PATH="/gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/multivi_splice_utils/runfiles/splicevimultivi_realdata.py"

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
echo "   MODALITY_WEIGHTS= $MODALITY_WEIGHTS"
echo "   N_LATENT= $N_LATENT"
echo "   SPLICING_LOSS_TYPE= $SPLICING_LOSS_TYPE"
echo "   SPLICING_ARCHITECTURE= $SPLICING_ARCHITECTURE"
echo "   EXPRESSION_ARCHITECTURE= $EXPRESSION_ARCHITECTURE"
echo "   ENCODER_HIDDEN_DIM= $ENCODER_HIDDEN_DIM"
echo "   H_HIDDEN_DIM= $H_HIDDEN_DIM"
echo "   ATSE_EMBEDDING_DIMENSION= $ATSE_EMBEDDING_DIMENSION"
echo "   TEMPERATURE_VALUE= $TEMPERATURE_VALUE"
echo "   TEMPERATURE_FIXED= $TEMPERATURE_FIXED"
echo "   MODALITY_WEIGHTS= $MODALITY_WEIGHTS"

echo "   MAX_EPOCHS= $MAX_EPOCHS"
echo "   LR= $LR"
echo "   BATCH_SIZE= $BATCH_SIZE"
echo "   WEIGHT_DECAY= $WEIGHT_DECAY"
echo "   EARLY_STOPPING_PATIENCE= $EARLY_STOPPING_PATIENCE"
echo "   LR_SCHEDULER_TYPE= $LR_SCHEDULER_TYPE"
echo "   LR_FACTOR= $LR_FACTOR"
echo "   LR_PATIENCE= $LR_PATIENCE"
echo "   STEP_SIZE= $STEP_SIZE"
echo "   GRADIENT_CLIPPING= $GRADIENT_CLIPPING"
echo "   GRADIENT_CLIPPING_MAX_NORM= $GRADIENT_CLIPPING_MAX_NORM"



# Activate Conda environment
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Run the Python script with CLI flags built from env vars
python "$SCRIPT_PATH" \
  ${TRAIN_MDATA_PATH:+--train_mdata_path "$TRAIN_MDATA_PATH"} \
  ${TEST_MDATA_PATH:+--test_mdata_path "$TEST_MDATA_PATH"} \
  ${MASKED_TEST_MDATA_PATH:+--masked_test_mdata_path "$MASKED_TEST_MDATA_PATH"} \
  ${MODEL_DIR:+--model_dir "$MODEL_DIR"} \
  ${FIG_DIR:+--fig_dir "$FIG_DIR"} \
  ${DROPOUT_RATE:+--dropout_rate "$DROPOUT_RATE"} \
  ${SPLICING_LOSS_TYPE:+--splicing_loss_type "$SPLICING_LOSS_TYPE"} \
  ${MAX_EPOCHS:+--max_epochs "$MAX_EPOCHS"} \
  ${LR:+--lr "$LR"} \
  ${BATCH_SIZE:+--batch_size "$BATCH_SIZE"} \
  ${N_EPOCHS_KL_WARMUP:+--n_epochs_kl_warmup "$N_EPOCHS_KL_WARMUP"} \
  ${CODE_DIM:+--code_dim "$CODE_DIM"} \
  ${N_LATENT:+--n_latent "$N_LATENT"} \
  ${ENCODER_TYPE:+--encoder_type "$ENCODER_TYPE"} \
  ${POOL_MODE:+--pool_mode "$POOL_MODE"} \
  ${FORWARD_STYLE:+--forward_style "$FORWARD_STYLE"} \
  ${MAX_NOBS:+--max_nobs "$MAX_NOBS"} \
  ${NUM_WEIGHT_VECTORS:+--num_weight_vectors "$NUM_WEIGHT_VECTORS"} \
  ${MODALITY_WEIGHTS:+--modality_weights "$MODALITY_WEIGHTS"} \
  ${SPLICING_ARCHITECTURE:+--splicing_architecture "$SPLICING_ARCHITECTURE"} \
  ${EXPRESSION_ARCHITECTURE:+--expression_architecture "$EXPRESSION_ARCHITECTURE"} \
  ${ENCODER_HIDDEN_DIM:+--encoder_hidden_dim "$ENCODER_HIDDEN_DIM"} \
  ${H_HIDDEN_DIM:+--h_hidden_dim "$H_HIDDEN_DIM"} \
  ${ATSE_EMBEDDING_DIMENSION:+--atse_embedding_dimension "$ATSE_EMBEDDING_DIMENSION"} \
  ${TEMPERATURE_VALUE:+--temperature_value "$TEMPERATURE_VALUE"} \
  ${TEMPERATURE_FIXED:+--temperature_fixed "$TEMPERATURE_FIXED"} \
  ${WEIGHT_DECAY:+--weight_decay "$WEIGHT_DECAY"} \
  ${EARLY_STOPPING_PATIENCE:+--early_stopping_patience "$EARLY_STOPPING_PATIENCE"} \
  ${LR_SCHEDULER_TYPE:+--lr_scheduler_type "$LR_SCHEDULER_TYPE"} \
  ${LR_FACTOR:+--lr_factor "$LR_FACTOR"} \
  ${LR_PATIENCE:+--lr_patience "$LR_PATIENCE"} \
  ${STEP_SIZE:+--step_size "$STEP_SIZE"} \
  ${GRADIENT_CLIPPING:+--gradient_clipping "$GRADIENT_CLIPPING"} \
  ${GRADIENT_CLIPPING_MAX_NORM:+--gradient_clipping_max_norm "$GRADIENT_CLIPPING_MAX_NORM"}


EOF

chmod +x "$BATCH_RUN_DIR/job_template.sh"

echo "→ Job template written. Submitting sweep jobs..."

# 2) Define sweep configurations
ENCODER_TYPES=(
  "PartialEncoderWeightedSumEDDIMultiWeight"
  "PartialEncoderEDDI"
  "PartialEncoderEDDIATSE"
  "PartialEncoderEDDI"
  "PartialEncoderEDDIATSE"
  "PartialEncoderWeightedSumEDDIMultiWeightATSE"
)
POOL_MODES=(
  ""
  "sum"
  "sum"
  "mean"
  "mean"
  ""
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

      JOB_NAME="sweep_REAL_cd=${CODE_DIM}_mn=${MAX_NOBS}_ld=${N_LATENT}_lr=${LR}_${i}_${FORWARD_STYLE}_${ENCODER_TYPE}"
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
TRAIN_MDATA_PATH="$TRAIN_MDATA_PATH",\
TEST_MDATA_PATH="$TEST_MDATA_PATH",\
MASKED_TEST_MDATA_PATH="$MASKED_TEST_MDATA_PATH",\
MODEL_DIR="$MODEL_DIR",\
FIG_DIR="$FIG_DIR",\
DROPOUT_RATE="$DROPOUT_RATE",\
MAX_EPOCHS="$MAX_EPOCHS",\
LR="$LR",\
BATCH_SIZE="$BATCH_SIZE",\
N_EPOCHS_KL_WARMUP="$N_EPOCHS_KL_WARMUP",\
CODE_DIM="$CODE_DIM",\
N_LATENT="$N_LATENT",\
ENCODER_TYPE="$ENCODER_TYPE",\
FORWARD_STYLE="$FORWARD_STYLE",\
SPLICING_LOSS_TYPE="$SPLICING_LOSS_TYPE",\
POOL_MODE="$POOL_MODE",\
MAX_NOBS="$MAX_NOBS",\
IMPUTEDENCODER="$IMPUTEDENCODER",\
NUM_WEIGHT_VECTORS="$NUM_WEIGHT_VECTORS",\
MODALITY_WEIGHTS="$MODALITY_WEIGHTS",\
SPLICING_ARCHITECTURE="$SPLICING_ARCHITECTURE",\
EXPRESSION_ARCHITECTURE="$EXPRESSION_ARCHITECTURE",\
ENCODER_HIDDEN_DIM="$ENCODER_HIDDEN_DIM",\
H_HIDDEN_DIM="$H_HIDDEN_DIM",\
ATSE_EMBEDDING_DIMENSION="$ATSE_EMBEDDING_DIMENSION",\
TEMPERATURE_VALUE="$TEMPERATURE_VALUE",\
TEMPERATURE_FIXED="$TEMPERATURE_FIXED",\
WEIGHT_DECAY="$WEIGHT_DECAY",\
EARLY_STOPPING_PATIENCE="$EARLY_STOPPING_PATIENCE",\
LR_SCHEDULER_TYPE="$LR_SCHEDULER_TYPE",\
LR_FACTOR="$LR_FACTOR",\
LR_PATIENCE="$LR_PATIENCE",\
STEP_SIZE="$STEP_SIZE",\
GRADIENT_CLIPPING="$GRADIENT_CLIPPING",\
GRADIENT_CLIPPING_MAX_NORM="$GRADIENT_CLIPPING_MAX_NORM" \
        "$BATCH_RUN_DIR/job_template.sh"
    done
  done
done


echo "→ All sweep jobs (code_dim ∈ ${CODE_DIMS[*]}) submitted."
echo "→ Monitor with: squeue -u $(whoami)"
echo "→ Logs and outputs in: $BATCH_RUN_DIR"
