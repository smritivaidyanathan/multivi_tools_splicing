#!/usr/bin/env bash
#SBATCH --job-name=MultiVI-Splice-Pipeline
#SBATCH --mem=150G
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

### ─── USER EDITABLE CONFIG ────────────────────────────────────────────── ###
# USAGE:
# 1) sbatch multivisplice_pipeline.sh
# 2) Override in-script by uncommenting and editing below
# 3) Override on-the-fly via sbatch --export=VAR=val

# Required data paths
MUDATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/052025/train_70_30_ge_splice_combined_20250513_035938.h5mu"
TEST_MUDATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/052025/test_30_70_ge_splice_combined_20250513_035938.h5mu"

# Model initialization
N_GENES="None"                  # --n_genes
N_JUNCTIONS="None"              # --n_junctions
MODALITY_WEIGHTS="concatenate"  # --modality_weights
MODALITY_PENALTY="Jeffreys"    # --modality_penalty
N_HIDDEN="None"                 # --n_hidden
LATENT_DIM=20                     # --n_latent
N_LAYERS_ENCODER=2                # --n_layers_encoder
N_LAYERS_DECODER=2                # --n_layers_decoder
DROPOUT_RATE=0.1                  # --dropout_rate
GENE_LIKELIHOOD="zinb"          # --gene_likelihood
SPLICING_LOSS_TYPE="dirichlet_multinomial"  # --splicing_loss_type
SPLICING_CONCENTRATION="None"    # --splicing_concentration
SPLICING_ARCHITECTURE="partial" # --splicing_architecture
EXPRESSION_ARCHITECTURE="linear" # --expression_architecture
DISPERSION="gene"               # --dispersion
USE_BATCH_NORM="none"           # --use_batch_norm
USE_LAYER_NORM="both"           # --use_layer_norm
LATENT_DISTRIBUTION="normal"    # --latent_distribution
DEEPLY_INJECT_COVARIATES="false" # --deeply_inject_covariates
ENCODE_COVARIATES="false"       # --encode_covariates
FULLY_PAIRED="false"            # --fully_paired

# Training hyperparameters
MAX_EPOCHS=500                    # --max_epochs
LR=0.000001                       # --lr
ACCELERATOR="auto"              # --accelerator
DEVICES="auto"                  # --devices
TRAIN_SIZE="None"               # --train_size
VALIDATION_SIZE="None"          # --validation_size
SHUFFLE_SET_SPLIT="true"        # --shuffle_set_split
BATCH_SIZE=256                    # --batch_size
WEIGHT_DECAY=1e-6                 # --weight_decay
EPS=1e-08                         # --eps
EARLY_STOPPING="true"           # --early_stopping
EARLY_STOPPING_PATIENCE=10       # --early_stopping_patience
SAVE_BEST="true"                # --save_best
CHECK_VAL_EVERY_N_EPOCH="None"  # --check_val_every_n_epoch
N_STEPS_KL_WARMUP="None"        # --n_steps_kl_warmup
N_EPOCHS_KL_WARMUP=30             # --n_epochs_kl_warmup
ADVERSARIAL_MIXING="true"       # --adversarial_mixing
STEP_SIZE=20                      # --step_size
LR_SCHEDULER_TYPE="step"        # --lr_scheduler_type
REDUCE_LR_ON_PLATEAU="false"    # --reduce_lr_on_plateau
LR_FACTOR=0.5                     # --lr_factor
LR_PATIENCE=30                    # --lr_patience
LR_THRESHOLD=0.0                  # --lr_threshold
LR_SCHEDULER_METRIC="elbo_validation" # --lr_scheduler_metric
GRADIENT_CLIPPING="true"        # --gradient_clipping
GRADIENT_CLIPPING_MAX_NORM=10.0   # --gradient_clipping_max_norm
DATASPLITTER_KWARGS="None"      # --datasplitter_kwargs
PLAN_KWARGS="None"              # --plan_kwargs

# Latent-space evaluation
UMAP_CELL_LABELS="broad_cell_type dataset"  # --umap_cell_labels
CLUSTER_NUMBERS="3 5 10"                    # --cluster_numbers
NEIGHBOR_K=30                                 # --neighbor_k
CELL_TYPE_COLUMN="broad_cell_type"          # --cell_type_column
TOP_N_CELLTYPES=5                             # --top_n_celltypes

### ─── SETUP RUN DIRECTORY ─────────────────────────────────────────────── ###
BASE_NAME="MultiVISplicePipeline"
RUN_ROOT="/gpfs/commons/home/svaidyanathan/multi_vi_splice_runs"
TS=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${RUN_ROOT}/${BASE_NAME}_${TS}_$$"
LATENT_SUB="latent_eval"
WEIGHT_SUB="weights_analysis"
IMPUTE_SUB="imputation_eval"

dirs=(
  "$RUN_DIR/models"
  "$RUN_DIR/figures"
  "$RUN_DIR/$LATENT_SUB/figures"
  "$RUN_DIR/$LATENT_SUB/csv_files"
  "$RUN_DIR/$WEIGHT_SUB/figures"
  "$RUN_DIR/$WEIGHT_SUB/csv_files"
  "$RUN_DIR/$IMPUTE_SUB/figures"
  "$RUN_DIR/$IMPUTE_SUB/csv_files"
)
mkdir -p "${dirs[@]}"
echo "[INFO] Directories created."

### ─── LOGGING & ENVIRONMENT ─────────────────────────────────────────────── ###
exec >"$RUN_DIR/slurm.out" 2>"$RUN_DIR/slurm.err"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate scvi-env

### ─── TRAINING ─────────────────────────────────────────────────────────── ###
args_train=(
  --mudata_path       "$MUDATA_PATH"
  --run_dir           "$RUN_DIR"
  --model_save_dir    "$RUN_DIR/models"
  --figure_output_dir "$RUN_DIR/figures"
)
# init flags
[ -n "$N_GENES" ]            && args_train+=( --n_genes "$N_GENES" )
[ -n "$N_JUNCTIONS" ]        && args_train+=( --n_junctions "$N_JUNCTIONS" )
[ -n "$MODALITY_WEIGHTS" ]   && args_train+=( --modality_weights "$MODALITY_WEIGHTS" )
[ -n "$MODALITY_PENALTY" ]   && args_train+=( --modality_penalty "$MODALITY_PENALTY" )
[ -n "$N_HIDDEN" ]           && args_train+=( --n_hidden "$N_HIDDEN" )
[ -n "$LATENT_DIM" ]         && args_train+=( --n_latent "$LATENT_DIM" )
[ -n "$N_LAYERS_ENCODER" ]   && args_train+=( --n_layers_encoder "$N_LAYERS_ENCODER" )
[ -n "$N_LAYERS_DECODER" ]   && args_train+=( --n_layers_decoder "$N_LAYERS_DECODER" )
[ -n "$DROPOUT_RATE" ]       && args_train+=( --dropout_rate "$DROPOUT_RATE" )
[ -n "$GENE_LIKELIHOOD" ]    && args_train+=( --gene_likelihood "$GENE_LIKELIHOOD" )
[ -n "$SPLICING_LOSS_TYPE" ] && args_train+=( --splicing_loss_type "$SPLICING_LOSS_TYPE" )
[ -n "$SPLICING_CONCENTRATION" ] && args_train+=( --splicing_concentration "$SPLICING_CONCENTRATION" )
[ -n "$SPLICING_ARCHITECTURE" ] && args_train+=( --splicing_architecture "$SPLICING_ARCHITECTURE" )
[ -n "$EXPRESSION_ARCHITECTURE" ] && args_train+=( --expression_architecture "$EXPRESSION_ARCHITECTURE" )
# training flags
[ -n "$MAX_EPOCHS" ]          && args_train+=( --max_epochs "$MAX_EPOCHS" )
[ -n "$LR" ]                  && args_train+=( --lr "$LR" )
[ -n "$ACCELERATOR" ]         && args_train+=( --accelerator "$ACCELERATOR" )
[ -n "$DEVICES" ]             && args_train+=( --devices "$DEVICES" )
[ -n "$TRAIN_SIZE" ]          && args_train+=( --train_size "$TRAIN_SIZE" )
[ -n "$VALIDATION_SIZE" ]     && args_train+=( --validation_size "$VALIDATION_SIZE" )
[ -n "$SHUFFLE_SET_SPLIT" ]   && args_train+=( --shuffle_set_split "$SHUFFLE_SET_SPLIT" )
[ -n "$BATCH_SIZE" ]          && args_train+=( --batch_size "$BATCH_SIZE" )
[ -n "$WEIGHT_DECAY" ]        && args_train+=( --weight_decay "$WEIGHT_DECAY" )
[ -n "$EPS" ]                 && args_train+=( --eps "$EPS" )
[ -n "$EARLY_STOPPING" ]      && args_train+=( --early_stopping "$EARLY_STOPPING" )
[ -n "$EARLY_STOPPING_PATIENCE" ] && args_train+=( --early_stopping_patience "$EARLY_STOPPING_PATIENCE" )
[ -n "$SAVE_BEST" ]           && args_train+=( --save_best "$SAVE_BEST" )
[ -n "$CHECK_VAL_EVERY_N_EPOCH" ] && args_train+=( --check_val_every_n_epoch "$CHECK_VAL_EVERY_N_EPOCH" )
[ -n "$N_STEPS_KL_WARMUP" ]   && args_train+=( --n_steps_kl_warmup "$N_STEPS_KL_WARMUP" )
[ -n "$N_EPOCHS_KL_WARMUP" ]  && args_train+=( --n_epochs_kl_warmup "$N_EPOCHS_KL_WARMUP" )
[ -n "$ADVERSARIAL_MIXING" ]  && args_train+=( --adversarial_mixing "$ADVERSARIAL_MIXING" )
[ -n "$STEP_SIZE" ]           && args_train+=( --step_size "$STEP_SIZE" )
[ -n "$LR_SCHEDULER_TYPE" ]   && args_train+=( --lr_scheduler_type "$LR_SCHEDULER_TYPE" )
[ -n="$REDUCE_LR_ON_PLATEAU" ]&& args_train+=( --reduce_lr_on_plateau "$REDUCE_LR_ON_PLATEAU" )
[ -n="$LR_FACTOR" ]           && args_train+=( --lr_factor "$LR_FACTOR" )
[ -n="$LR_PATIENCE" ]         && args_train+=( --lr_patience "$LR_PATIENCE" )
[ -n="$LR_THRESHOLD" ]        && args_train+=( --lr_threshold "$LR_THRESHOLD" )
[ -n="$LR_SCHEDULER_METRIC" ] && args_train+=( --lr_scheduler_metric "$LR_SCHEDULER_METRIC" )
[ -n="$GRADIENT_CLIPPING" ]   && args_train+=( --gradient_clipping "$GRADIENT_CLIPPING" )
[ -n="$GRADIENT_CLIPPING_MAX_NORM" ] && args_train+=( --gradient_clipping_max_norm "$GRADIENT_CLIPPING_MAX_NORM" )
[ -n="$DATASPLITTER_KWARGS" ] && args_train+=( --datasplitter_kwargs "$DATASPLITTER_KWARGS" )
[ -n="$PLAN_KWARGS" ]         && args_train+=( --plan_kwargs "$PLAN_KWARGS" )

echo "=== [TRAINING] Starting ==="
python /gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/multivi_splice_utils/runfiles/train_multivisplice.py "${args_train[@]}"

### ─── LATENT-SPACE EVALUATION ───────────────────────────────────────────── ###
args_latent=(
  --mudata_path "$MUDATA_PATH"
  --model_path   "$RUN_DIR/models"
  --out_dir      "$RUN_DIR/$LATENT_SUB"
)
[ -n "$CLUSTER_NUMBERS" ]     && read -r -a cl <<< "$CLUSTER_NUMBERS" && args_latent+=( --cluster_numbers "${cl[@]}" )
[ -n "$NEIGHBOR_K" ]          && args_latent+=( --neighbor_k "$NEIGHBOR_K" )
[ -n "$CELL_TYPE_COLUMN" ]    && args_latent+=( --cell_type_column "$CELL_TYPE_COLUMN" )
[ -n="$TOP_N_CELLTYPES" ]     && args_latent+=( --top_n_celltypes "$TOP_N_CELLTYPES" )
[ -n="$UMAP_CELL_LABELS" ]    && read -r -a ul <<< "$UMAP_CELL_LABELS" && args_latent+=( --umap_cell_labels "${ul[@]}" )

echo "=== [LATENT EVAL] Starting ==="
python /gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/multivi_splice_utils/runfiles/latentspace_multivisplice.py "${args_latent[@]}"

### ─── WEIGHT ANALYSIS ─────────────────────────────────────────────────── ###
args_weight=(
  --mudata_path "$MUDATA_PATH"
  --model_path   "$RUN_DIR/models"
  --out_dir      "$RUN_DIR/$WEIGHT_SUB"
)
[ -n="$UMAP_CELL_LABELS" ]    && read -r -a ul2 <<< "$UMAP_CELL_LABELS" && args_weight+=( --umap_labels "${ul2[@]}" )

echo "=== [WEIGHT ANALYSIS] Starting ==="
python /gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/multivi_splice_utils/runfiles/weights_multivisplice.py "${args_weight[@]}"

### ─── IMPUTATION EVALUATION ───────────────────────────────────────────── ###
args_imp=(
  --test_mudata_path "$TEST_MUDATA_PATH"
  --model_path       "$RUN_DIR/models"
  --out_dir          "$RUN_DIR/$IMPUTE_SUB"
)

echo "=== [IMPUTATION EVAL] Starting ==="
python /gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/multivi_splice_utils/runfiles/imputation_multivisplice.py "${args_imp[@]}"

echo "=== Pipeline complete. Outputs in ${RUN_DIR} ==="
