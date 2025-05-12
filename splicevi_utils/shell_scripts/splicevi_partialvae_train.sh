#!/usr/bin/env bash
#SBATCH --job-name=Splice_VI_PartialVAE_Training
#SBATCH --mem=100G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

### ─── USER EDITABLE CONFIG ────────────────────────────────────────────── ###
# USAGE:
# 1. Default run (uses all defaults):
#      sbatch run_splicevi_partialvae.sh
# 2. Override in-script:
#      Uncomment and edit variables under "# Optional hyperparams"
# 3. Override on-the-fly:
#      sbatch --export=LR=5e-4,latent_dim=20 run_splicevi_partialvae.sh
#    (exported vars take precedence over in-script defaults)
# ───────────────────────────────────────────────────────────────────────────

# Required
ADATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/TMS_MODELING/DATA_FILES/SIMULATED/simulated_data_2025-03-27.h5ad"

# Optional hyperparams for MODEL INIT (uncomment to override; defaults in parentheses):
#CODE_DIM=128                # --code_dim (default: 16)
# H_HIDDEN_DIM=64            # --h_hidden_dim (default: 64)
#ENCODER_HIDDEN_DIM=32       # --encoder_hidden_dim (default: 128)
#LATENT_DIM=10                # --latent_dim (default: 10)
DROPOUT_RATE=0.01           # --dropout_rate (default: 0.0)
# LEARN_CONCENTRATION=true   # --learn_concentration (default: true)
SPLICE_LIKELIHOOD="binomial" # --splice_likelihood (default: "beta_binomial")

# Optional hyperparams for TRAINING (uncomment to override; defaults in parentheses):
MAX_EPOCHS=100         # --max_epochs (default: 500)
LR=1e-4                      # --lr (default: 1e-4)
# ACCELERATOR="auto"       # --accelerator (default: "auto")
# DEVICES="auto"           # --devices (default: "auto")
# TRAIN_SIZE="None"        # --train_size (default: None)
# VALIDATION_SIZE="None"   # --validation_size (default: None)
# SHUFFLE_SET_SPLIT="true" # --shuffle_set_split (default: true)
BATCH_SIZE=128               # --batch_size (default: 128)
WEIGHT_DECAY=0         # --weight_decay (default: 1e-3)
# EPS=1e-08                  # --eps (default: 1e-8)
# EARLY_STOPPING="true"    # --earlbut ty_stopping (default: true)
# SAVE_BEST="true"         # --save_best (default: true)
# CHECK_VAL_EVERY_N_EPOCH="None" # --check_val_every_n_epoch (default: None)
# N_STEPS_KL_WARMUP="None" # --n_steps_kl_warmup (default: None)
#N_EPOCHS_KL_WARMUP=0    # --n_epochs_kl_warmup (default: 50)
# REDUCE_LR_ON_PLATEAU=""   # if set to any non-empty value, will turn on LR scheduling
# LR_FACTOR=0.6             # --lr_factor (default: 0.6)
# LR_PATIENCE=30            # --lr_patience (default: 30)
# LR_THRESHOLD=0.0          # --lr_threshold (default: 0.0)
# LR_MIN=0.0                # --lr_min (default: 0.0)
# DATASPLITTER_KWARGS="None" # --datasplitter_kwargs (default: None)
# PLAN_KWARGS="None"       # --plan_kwargs (default: None)

# UMAP colors (uncomment to override; default: cell_type_grouped)
# UMAP_COLORS="cell_type_grouped sex mouse.id"
### ────────────────────────────────────────────────────────────────────── ###

# Build unique run folder
BASE_NAME="SpliceVI_PartialVAE_Training"
RUN_ROOT="/gpfs/commons/home/svaidyanathan/splice_vi_partial_vae_runs"
TS=$(date +"%Y%m%d_%H%M%S")
RUN_ID="${TS}_job${SLURM_JOB_ID:-manual}"
RUN_DIR="${RUN_ROOT}/${BASE_NAME}_${RUN_ID}"
mkdir -p "${RUN_DIR}/models" "${RUN_DIR}/figures"

# Redirect stdout/stderr
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
  --adata_path   "$ADATA_PATH"
  --model_dir    "${RUN_DIR}/models"
  --fig_dir      "${RUN_DIR}/figures"
)

# MODEL INIT flags
[ -n "$CODE_DIM" ]            && args+=( --code_dim "$CODE_DIM" )
[ -n "$H_HIDDEN_DIM" ]        && args+=( --h_hidden_dim "$H_HIDDEN_DIM" )
[ -n "$ENCODER_HIDDEN_DIM" ]  && args+=( --encoder_hidden_dim "$ENCODER_HIDDEN_DIM" )
[ -n "$LATENT_DIM" ]          && args+=( --latent_dim "$LATENT_DIM" )
[ -n "$DROPOUT_RATE" ]        && args+=( --dropout_rate "$DROPOUT_RATE" )
[ -n "$LEARN_CONCENTRATION" ] && args+=( --learn_concentration "$LEARN_CONCENTRATION" )
[ -n "$SPLICE_LIKELIHOOD" ]    && args+=( --splice_likelihood "$SPLICE_LIKELIHOOD" )

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
[ -n "$REDUCE_LR_ON_PLATEAU" ] && args+=( --reduce_lr_on_plateau )
[ -n "$LR_FACTOR" ]            && args+=( --lr_factor "$LR_FACTOR" )
[ -n "$LR_PATIENCE" ]          && args+=( --lr_patience "$LR_PATIENCE" )
[ -n "$LR_THRESHOLD" ]         && args+=( --lr_threshold "$LR_THRESHOLD" )
[ -n "$LR_MIN" ]               && args+=( --lr_min "$LR_MIN" )
[ -n "$DATASPLITTER_KWARGS" ] && args+=( --datasplitter_kwargs "$DATASPLITTER_KWARGS" )
[ -n "$PLAN_KWARGS" ]         && args+=( --plan_kwargs "$PLAN_KWARGS" )

# UMAP labels
if [ -n "$UMAP_CELL_LABELS" ]; then
  # split on spaces and pass each as its own arg
  read -r -a labels <<< "$UMAP_CELL_LABELS"
  args+=( --umap_cell_label "${labels[@]}" )
fi

# Launch the pipeline
python /gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/splicevi_utils/runfiles/splicevipartialvae.py "${args[@]}"
