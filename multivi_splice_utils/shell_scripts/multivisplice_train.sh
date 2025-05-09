#!/usr/bin/env bash
#SBATCH --job-name=MultiVI-Splice-Training
#SBATCH --mem=100G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

### ─── USER EDITABLE CONFIG ────────────────────────────────────────────── ###

# ──────────────────────────────────────────────────────────────────────────────
# USAGE:
#
# 1. Default run (uses all the defaults you’ve set in the USER EDITABLE CONFIG):
#     sbatch run_multivi_splice.sh
#
# 2. To override any parameter in the script itself:
#     – Uncomment and edit its variable under “# Optional hyperparams”.
#
# 3. To override on‐the‐fly (no file edits needed):
#     sbatch --export=LATENT_DIM=64,LR=5e-4 run_multivi_splice.sh
#
#    (any vars you export will take precedence over the commented‐in defaults)
# ──────────────────────────────────────────────────────────────────────────────


# Required
MUDATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/052025/mouse_foundation_data_20250502_155802_ge_splice_combined.h5mu"

# Optional hyperparams (uncomment to override)
LATENT_DIM=30              # --latent_dim (default: model’s internal)
# LR=1e-4                    # --lr (default: 1e-4)
# MAX_EPOCHS=500             # --max_epochs (default: 500)
# ACCELERATOR="auto"         # --accelerator (default: "auto")
# DEVICES="auto"             # --devices (default: "auto")
# TRAIN_SIZE=""              # --train_size (default: None)
# VALIDATION_SIZE=""         # --validation_size (default: None)
# SHUFFLE_SET_SPLIT="true"   # --shuffle_set_split (default: true)
# BATCH_SIZE=128             # --batch_size (default: 128)
# WEIGHT_DECAY=1e-3          # --weight_decay (default: 1e-3)
# EPS=1e-08                  # --eps (default: 1e-08)
# EARLY_STOPPING="true"      # --early_stopping (default: true)
# SAVE_BEST="true"           # --save_best (default: true)
# CHECK_VAL_EVERY_N_EPOCH="" # --check_val_every_n_epoch (default: None)
# N_STEPS_KL_WARMUP=""       # --n_steps_kl_warmup (default: None)
# N_EPOCHS_KL_WARMUP=50      # --n_epochs_kl_warmup (default: 50)
# ADVERSARIAL_MIXING="true"  # --adversarial_mixing (default: true)
# DATASPLITTER_KWARGS=""     # --datasplitter_kwargs (default: None)
# PLAN_KWARGS=""             # --plan_kwargs (default: None)

# UMAP labels (uncomment to override)
# UMAP_CELL_LABELS="broad_cell_type dataset batch"  # --umap_cell_label
### ────────────────────────────────────────────────────────────────────── ###

# build a unique run folder inside RUN_ROOT, which we assume already exists
BASE_NAME="MultiVISpliceTraining"
RUN_ROOT="/gpfs/commons/home/svaidyanathan/multi_vi_splice_runs"
TS=$(date +"%Y%m%d_%H%M%S")
RUN_ID="${TS}_job${SLURM_JOB_ID:-manual}"
RUN_DIR="${RUN_ROOT}/${BASE_NAME}_${RUN_ID}"
mkdir -p "${RUN_DIR}/models" "${RUN_DIR}/figures"

# redirect Slurm stdout/stderr into run directory
if [ -n "$SLURM_JOB_ID" ]; then
  exec > "${RUN_DIR}/slurm_${SLURM_JOB_ID}.out" 2> "${RUN_DIR}/slurm_${SLURM_JOB_ID}.err"
else
  exec > "${RUN_DIR}/slurm_manual.out" 2> "${RUN_DIR}/slurm_manual.err"
fi

echo "→ Run dir: $RUN_DIR"
echo "   models  → ${RUN_DIR}/models"
echo "   figures → ${RUN_DIR}/figures"

# ─── Initialize Conda ─────────────────────────────────────────────────────
CONDA_BASE="/gpfs/commons/home/svaidyanathan/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate scvi-env
# ──────────────────────────────────────────────────────────────────────────

# Build an array of CLI args
args=(
  --mudata_path       "$MUDATA_PATH"
  --model_save_dir    "${RUN_DIR}/models"
  --figure_output_dir "${RUN_DIR}/figures"
)

# Only add the flag if its variable is non-empty
[ -n "$LATENT_DIM" ]      && args+=( --latent_dim       "$LATENT_DIM" )
[ -n "$LR" ]              && args+=( --lr               "$LR" )
[ -n "$MAX_EPOCHS" ]      && args+=( --max_epochs       "$MAX_EPOCHS" )
[ -n "$ACCELERATOR" ]     && args+=( --accelerator      "$ACCELERATOR" )
[ -n "$DEVICES" ]         && args+=( --devices          "$DEVICES" )
[ -n "$TRAIN_SIZE" ]      && args+=( --train_size       "$TRAIN_SIZE" )
[ -n "$VALIDATION_SIZE" ] && args+=( --validation_size  "$VALIDATION_SIZE" )
[ -n "$SHUFFLE_SET_SPLIT" ] && args+=( --shuffle_set_split "$SHUFFLE_SET_SPLIT" )
[ -n "$BATCH_SIZE" ]      && args+=( --batch_size      "$BATCH_SIZE" )
[ -n "$WEIGHT_DECAY" ]    && args+=( --weight_decay    "$WEIGHT_DECAY" )
[ -n "$EPS" ]             && args+=( --eps              "$EPS" )
[ -n "$EARLY_STOPPING" ]  && args+=( --early_stopping   "$EARLY_STOPPING" )
[ -n "$SAVE_BEST" ]       && args+=( --save_best        "$SAVE_BEST" )
[ -n "$CHECK_VAL_EVERY_N_EPOCH" ] && args+=( --check_val_every_n_epoch "$CHECK_VAL_EVERY_N_EPOCH" )
[ -n "$N_STEPS_KL_WARMUP" ] && args+=( --n_steps_kl_warmup "$N_STEPS_KL_WARMUP" )
[ -n "$N_EPOCHS_KL_WARMUP" ] && args+=( --n_epochs_kl_warmup "$N_EPOCHS_KL_WARMUP" )
[ -n "$ADVERSARIAL_MIXING" ] && args+=( --adversarial_mixing "$ADVERSARIAL_MIXING" )
[ -n "$DATASPLITTER_KWARGS" ] && args+=( --datasplitter_kwargs "$DATASPLITTER_KWARGS" )
[ -n "$PLAN_KWARGS" ]       && args+=( --plan_kwargs      "$PLAN_KWARGS" )
[ -n "$UMAP_CELL_LABELS" ]  && args+=( --umap_cell_label "$UMAP_CELL_LABELS" )

# Now call python with exactly the flags you set above
python /gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/multivi_splice_utils/runfiles/multivirun.py "${args[@]}"