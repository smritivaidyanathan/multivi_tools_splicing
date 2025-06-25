#!/usr/bin/env bash
#SBATCH --job-name=MultiVI-Splice-Training
#SBATCH --mem=150G
#SBATCH --partition=gpu
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1

### ─── USER EDITABLE CONFIG ────────────────────────────────────────────── ###
# ───────────────────────────────────────────────────────────────────────────
# USAGE:
# 1. Default run (uses all the defaults from the Python script):
#  sbatch multivisplice_train.sh
# 2. Override in-script parameters:
#      Uncomment and edit variables under "# Optional hyperparams"
# 3. Override on-the-fly:
#      sbatch --export=LATENT_DIM=64,LR=5e-4 multivisplice_train.sh
#    (exported vars take precedence over in-script defaults)
# ───────────────────────────────────────────────────────────────────────────

# Required
#MUDATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/052025/aligned__ge_splice_combined_20250513_035938.h5mu"
# Test on subset of data
# MUDATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/052025/SUBSETTOP5CELLSTYPES_aligned__ge_splice_combined_20250513_035938.h5mu"
MUDATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/052025/SUBSETTOP5CELLSTYPES_aligned__ge_splice_combined_20250513_035938.h5mu"
# Optional hyperparams for MODEL INIT (uncomment to override; defaults in parentheses):
# N_GENES="None"                 # --n_genes (default: None, inferred from data)
# N_JUNCTIONS="None"             # --n_junctions (default: None, inferred from data)
MODALITY_WEIGHTS="concatenate"       # --modality_weights (default: "equal")
# MODALITY_PENALTY="Jeffreys"    # --modality_penalty (default: "Jeffreys")
# N_HIDDEN="None"                # --n_hidden (default: None = √n_junctions)
LATENT_DIM=20                     # --n_latent (default: None = √n_hidden)
# N_LAYERS_ENCODER=2               # --n_layers_encoder (default: 2)
# N_LAYERS_DECODER=2               # --n_layers_decoder (default: 2)
# DROPOUT_RATE=0.1                 # --dropout_rate (default: 0.1)
# GENE_LIKELIHOOD="zinb"         # --gene_likelihood (default: "zinb") 
SPLICING_LOSS_TYPE="dirichlet_multinomial" # --splicing_loss_type (default: "beta_binomial")
# SPLICING_CONCENTRATION="None"   # --splicing_concentration (default: None)
SPLICING_ARCHITECTURE="partial"   # --splicing_architecture (default: "vanilla")
EXPRESSION_ARCHITECTURE="linear"   # --expression_architecture (default: "vanilla")
# DISPERSION="gene"              # --dispersion (default: "gene")
# USE_BATCH_NORM="none"          # --use_batch_norm (default: "none")
# USE_LAYER_NORM="both"          # --use_layer_norm (default: "both")
# LATENT_DISTRIBUTION="normal"   # --latent_distribution (default: "normal")
# DEEPLY_INJECT_COVARIATES="false" # --deeply_inject_covariates (default: false)
# ENCODE_COVARIATES="false"      # --encode_covariates (default: false)
# FULLY_PAIRED="false"           # --fully_paired (default: false)

# Optional hyperparams for TRAINING (uncomment to override; defaults in parentheses):
MAX_EPOCHS=500                   # --max_epochs (default: 500)
LR=0.000001                        # --lr (default: 1e-4)
# ACCELERATOR="auto"             # --accelerator (default: "auto")
# DEVICES="auto"                 # --devices (default: "auto")
# TRAIN_SIZE="None"              # --train_size (default: None)
# VALIDATION_SIZE="None"         # --validation_size (default: None)
# SHUFFLE_SET_SPLIT="true"       # --shuffle_set_split (default: true)
BATCH_SIZE=256                   # --batch_size (default: 128)
WEIGHT_DECAY=1e-6                # --weight_decay (default: 1e-3)
# EPS=1e-08                        # --eps (default: 1e-08)
# EARLY_STOPPING="true"          # --early_stopping (default: true)
EARLY_STOPPING_PATIENCE=10      #--early_stopping_patience (default: 50)
# SAVE_BEST="true"               # --save_best (default: true)
# CHECK_VAL_EVERY_N_EPOCH="None" # --check_val_every_n_epoch (default: None)
# N_STEPS_KL_WARMUP="None"       # --n_steps_kl_warmup (default: None)
N_EPOCHS_KL_WARMUP=30            # --n_epochs_kl_warmup (default: 50)
#ADVERSARIAL_MIXING="true"      # --adversarial_mixing (default: true)
STEP_SIZE=20                    # --step_size (default: 10)
LR_SCHEDULER_TYPE="step"      # --lr_scheduler_type (default: "plateau")
#REDUCE_LR_ON_PLATEAU="false"     # --reduce_lr_on_plateau (default: false)
LR_FACTOR=0.5                    # --lr_factor (default: 0.6)
#LR_PATIENCE=30                   # --lr_patience (default: 30)
#LR_THRESHOLD=0.0                 # --lr_threshold (default: 0.0)
#LR_SCHEDULER_METRIC="elbo_validation"  # --lr_scheduler_metric (default: "elbo_validation")
#GRADIENT_CLIPPING="false"  # --gradient_clipping (default: true)
#GRADIENT_CLIPPING_MAX_NORM=10.0  # --gradient_clipping_max_norm (default: 5.0)
# DATASPLITTER_KWARGS="None"     # --datasplitter_kwargs (default: None)
# PLAN_KWARGS="None"             # --plan_kwargs (default: None)

# UMAP labels (uncomment to override; default: broad_cell_type)
UMAP_CELL_LABELS="broad_cell_type dataset"
# ───────────────────────────────────────────────────────────────────────────

# Build unique run folder
BASE_NAME="MultiVISpliceTraining"
RUN_ROOT="/gpfs/commons/home/svaidyanathan/multi_vi_splice_runs"
TS=$(date +"%Y%m%d_%H%M%S")
RUN_ID="${TS}_job${SLURM_JOB_ID:-manual}"
RUN_DIR="${RUN_ROOT}/${BASE_NAME}_${RUN_ID}"
mkdir -p "${RUN_DIR}/models" "${RUN_DIR}/figures"

# Redirect stdout/stderr into run directory
if [ -n "$SLURM_JOB_ID" ]; then
  exec > "${RUN_DIR}/slurm_${SLURM_JOB_ID}.out" 2> "${RUN_DIR}/slurm_${SLURM_JOB_ID}.err"
else
  exec > "${RUN_DIR}/slurm_manual.out" 2> "${RUN_DIR}/slurm_manual.err"
fi

echo "→ Run dir: $RUN_DIR"
echo "   models  → ${RUN_DIR}/models"
echo "   figures → ${RUN_DIR}/figures"

# Initialize Conda
CONDA_BASE="/gpfs/commons/home/svaidyanathan/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate scvi-env

# Build args array
args=(
  --mudata_path       "$MUDATA_PATH"
  --model_save_dir    "${RUN_DIR}/models"
  --figure_output_dir "${RUN_DIR}/figures"
)

# MODEL INIT flags
[ -n "$N_GENES" ]            && args+=( --n_genes "$N_GENES" )
[ -n "$N_JUNCTIONS" ]        && args+=( --n_junctions "$N_JUNCTIONS" )
[ -n "$MODALITY_WEIGHTS" ]   && args+=( --modality_weights "$MODALITY_WEIGHTS" )
[ -n "$MODALITY_PENALTY" ]   && args+=( --modality_penalty "$MODALITY_PENALTY" )
[ -n "$N_HIDDEN" ]           && args+=( --n_hidden "$N_HIDDEN" )
[ -n "$LATENT_DIM" ]         && args+=( --n_latent "$LATENT_DIM" )
[ -n "$N_LAYERS_ENCODER" ]   && args+=( --n_layers_encoder "$N_LAYERS_ENCODER" )
[ -n "$N_LAYERS_DECODER" ]   && args+=( --n_layers_decoder "$N_LAYERS_DECODER" )
[ -n "$DROPOUT_RATE" ]       && args+=( --dropout_rate "$DROPOUT_RATE" )
[ -n "$GENE_LIKELIHOOD" ]    && args+=( --gene_likelihood "$GENE_LIKELIHOOD" )
[ -n "$SPLICING_LOSS_TYPE" ] && args+=( --splicing_loss_type "$SPLICING_LOSS_TYPE" )
[ -n "$SPLICING_CONCENTRATION" ] && args+=( --splicing_concentration "$SPLICING_CONCENTRATION" )
[ -n "$SPLICING_ARCHITECTURE" ] && args+=( --splicing_architecture "$SPLICING_ARCHITECTURE" )
[ -n "$EXPRESSION_ARCHITECTURE" ] && args+=( --expression_architecture "$EXPRESSION_ARCHITECTURE" )
[ -n "$DISPERSION" ]         && args+=( --dispersion "$DISPERSION" )
[ -n "$USE_BATCH_NORM" ]     && args+=( --use_batch_norm "$USE_BATCH_NORM" )
[ -n "$USE_LAYER_NORM" ]     && args+=( --use_layer_norm "$USE_LAYER_NORM" )
[ -n "$LATENT_DISTRIBUTION" ]&& args+=( --latent_distribution "$LATENT_DISTRIBUTION" )
[ -n "$DEEPLY_INJECT_COVARIATES" ] && args+=( --deeply_inject_covariates "$DEEPLY_INJECT_COVARIATES" )
[ -n "$ENCODE_COVARIATES" ]  && args+=( --encode_covariates "$ENCODE_COVARIATES" )
[ -n "$FULLY_PAIRED" ]       && args+=( --fully_paired "$FULLY_PAIRED" )

# TRAINING flags
[ -n "$MAX_EPOCHS" ]          && args+=( --max_epochs "$MAX_EPOCHS" )
[ -n "$LR" ]                  && args+=( --lr "$LR" )
[ -n "$ACCELERATOR" ]         && args+=( --accelerator "$ACCELERATOR" )
[ -n "$DEVICES" ]             && args+=( --devices "$DEVICES" )
[ -n "$TRAIN_SIZE" ]          && args+=( --train_size "$TRAIN_SIZE" )
[ -n "$VALIDATION_SIZE" ]     && args+=( --validation_size "$VALIDATION_SIZE" )
[ -n "$SHUFFLE_SET_SPLIT" ]   && args+=( --shuffle_set_split "$SHUFFLE_SET_SPLIT" )
[ -n "$BATCH_SIZE" ]          && args+=( --batch_size "$BATCH_SIZE" )
[ -n "$WEIGHT_DECAY" ]        && args+=( --weight_decay "$WEIGHT_DECAY" )
[ -n "$EPS" ]                 && args+=( --eps "$EPS" )
[ -n "$EARLY_STOPPING" ]      && args+=( --early_stopping "$EARLY_STOPPING" )
[ -n "$SAVE_BEST" ]           && args+=( --save_best "$SAVE_BEST" )
[ -n "$CHECK_VAL_EVERY_N_EPOCH" ] && args+=( --check_val_every_n_epoch "$CHECK_VAL_EVERY_N_EPOCH" )
[ -n "$N_STEPS_KL_WARMUP" ]   && args+=( --n_steps_kl_warmup "$N_STEPS_KL_WARMUP" )
[ -n "$N_EPOCHS_KL_WARMUP" ]  && args+=( --n_epochs_kl_warmup "$N_EPOCHS_KL_WARMUP" )
[ -n "$ADVERSARIAL_MIXING" ]  && args+=( --adversarial_mixing "$ADVERSARIAL_MIXING" )
[ -n "$STEP_SIZE" ]           && args+=( --step_size "$STEP_SIZE" )
[ -n "$LR_SCHEDULER_TYPE" ]   && args+=( --lr_scheduler_type "$LR_SCHEDULER_TYPE" )
[ -n "$REDUCE_LR_ON_PLATEAU" ]&& args+=( --reduce_lr_on_plateau "$REDUCE_LR_ON_PLATEAU" )
[ -n "$LR_FACTOR" ]           && args+=( --lr_factor "$LR_FACTOR" )
[ -n "$EARLY_STOPPING_PATIENCE" ] && args+=( --early_stopping_patience "$EARLY_STOPPING_PATIENCE" )
[ -n "$LR_PATIENCE" ]         && args+=( --lr_patience "$LR_PATIENCE" )
[ -n "$LR_THRESHOLD" ]        && args+=( --lr_threshold "$LR_THRESHOLD" )
[ -n "$LR_SCHEDULER_METRIC" ] && args+=( --lr_scheduler_metric "$LR_SCHEDULER_METRIC" )
[ -n "$GRADIENT_CLIPPING" ] && args+=( --gradient_clipping "$GRADIENT_CLIPPING" )
[ -n "$GRADIENT_CLIPPING_MAX_NORM" ] && args+=( --gradient_clipping_max_norm "$GRADIENT_CLIPPING_MAX_NORM" )
[ -n "$DATASPLITTER_KWARGS" ] && args+=( --datasplitter_kwargs "$DATASPLITTER_KWARGS" )
[ -n "$PLAN_KWARGS" ]         && args+=( --plan_kwargs "$PLAN_KWARGS" )

# UMAP labels
if [ -n "$UMAP_CELL_LABELS" ]; then
  # split on spaces and pass each as its own arg
  read -r -a labels <<< "$UMAP_CELL_LABELS"
  args+=( --umap_cell_label "${labels[@]}" )
fi

# Launch the pipeline
python /gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/multivi_splice_utils/runfiles/multivirun.py "${args[@]}"
