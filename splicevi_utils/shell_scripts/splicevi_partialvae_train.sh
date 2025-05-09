#!/usr/bin/env bash
#SBATCH --job-name=Splice_VI_PartialVAE_Training
#SBATCH --mem=100G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

### ─── USER EDITABLE CONFIG ────────────────────────────────────────────── ###
# USAGE:
# 1. Default run:
#      sbatch run_splicevi_pipeline.sh
# 2. Override in-script:
#      Uncomment and edit variables below.
# 3. Override on-the-fly:
#      sbatch --export=LR=5e-4,latent_dim=64 splicevi_partialvae_train.sh

# Required
ADATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/TMS_MODELING/DATA_FILES/SIMULATED/simulated_data_2025-03-27.h5ad"

# Model init params (uncomment to override)
# code_dim=512            # --code_dim
# h_hidden_dim=256        # --h_hidden_dim
# encoder_hidden_dim=128  # --encoder_hidden_dim
# latent_dim=30           # --latent_dim
# decoder_hidden_dim=256  # --decoder_hidden_dim
# dropout_rate=0.1        # --dropout_rate
# learn_concentration=false # --learn_concentration

# Training params (uncomment to override)
# max_epochs=500                # --max_epochs
# lr=1e-4                       # --lr
# accelerator="auto"          # --accelerator
# devices="auto"              # --devices
# train_size=""               # --train_size
# validation_size=""          # --validation_size
# shuffle_set_split="true"    # --shuffle_set_split
# batch_size=128                # --batch_size
# weight_decay=1e-3             # --weight_decay
# eps=1e-8                      # --eps
# early_stopping="true"       # --early_stopping
# save_best="true"            # --save_best
# check_val_every_n_epoch=""  # --check_val_every_n_epoch
# n_steps_kl_warmup=""        # --n_steps_kl_warmup
# n_epochs_kl_warmup=50        # --n_epochs_kl_warmup
# datasplitter_kwargs=""      # --datasplitter_kwargs
# plan_kwargs=""              # --plan_kwargs

# UMAP colors (uncomment to override)
# UMAP_COLORS="cell_type_grouped sex mouse.id"
### ────────────────────────────────────────────────────────────────────── ###

# build unique run folder
BASE_NAME="SpliceVI_PartialVAE_Training"
RUN_ROOT="/gpfs/commons/home/svaidyanathan/splice_vi_partial_vae_runs"
TS=$(date +"%Y%m%d_%H%M%S")
RUN_ID="${TS}_job${SLURM_JOB_ID:-manual}"
RUN_DIR="${RUN_ROOT}/${BASE_NAME}_${RUN_ID}"
mkdir -p "${RUN_DIR}/models" "${RUN_DIR}/figures"

# redirect stdout/stderr
if [ -n "$SLURM_JOB_ID" ]; then
  exec > "${RUN_DIR}/slurm_${SLURM_JOB_ID}.out" 2> "${RUN_DIR}/slurm_${SLURM_JOB_ID}.err"
else
  exec > "${RUN_DIR}/slurm_manual.out" 2> "${RUN_DIR}/slurm_manual.err"
fi

echo "→ Run dir: $RUN_DIR"
echo "   models  → ${RUN_DIR}/models"
echo "   figures → ${RUN_DIR}/figures"

# activate conda env
CONDA_BASE="/gpfs/commons/home/svaidyanathan/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate scvi-env

# build args array
args=(
  --adata_path       "$ADATA_PATH"
  --model_dir        "${RUN_DIR}/models"
  --fig_dir          "${RUN_DIR}/figures"
)

# model init flags
[ -n "$code_dim" ]         && args+=( --code_dim         "$code_dim" )
[ -n "$h_hidden_dim" ]     && args+=( --h_hidden_dim     "$h_hidden_dim" )
[ -n "$encoder_hidden_dim" ] && args+=( --encoder_hidden_dim "$encoder_hidden_dim" )
[ -n "$latent_dim" ]       && args+=( --latent_dim       "$latent_dim" )
[ -n "$decoder_hidden_dim" ] && args+=( --decoder_hidden_dim "$decoder_hidden_dim" )
[ -n "$dropout_rate" ]     && args+=( --dropout_rate     "$dropout_rate" )
[ -n "$learn_concentration" ] && args+=( --learn_concentration "$learn_concentration" )

# training flags
[ -n "$max_epochs" ]            && args+=( --max_epochs            "$max_epochs" )
[ -n "$lr" ]                    && args+=( --lr                    "$lr" )
[ -n "$accelerator" ]           && args+=( --accelerator           "$accelerator" )
[ -n "$devices" ]               && args+=( --devices               "$devices" )
[ -n "$train_size" ]            && args+=( --train_size            "$train_size" )
[ -n "$validation_size" ]       && args+=( --validation_size       "$validation_size" )
[ -n "$shuffle_set_split" ]     && args+=( --shuffle_set_split     "$shuffle_set_split" )
[ -n "$batch_size" ]            && args+=( --batch_size            "$batch_size" )
[ -n "$weight_decay" ]          && args+=( --weight_decay          "$weight_decay" )
[ -n "$eps" ]                   && args+=( --eps                   "$eps" )
[ -n "$early_stopping" ]        && args+=( --early_stopping        "$early_stopping" )
[ -n "$save_best" ]             && args+=( --save_best             "$save_best" )
[ -n "$check_val_every_n_epoch" ] && args+=( --check_val_every_n_epoch "$check_val_every_n_epoch" )
[ -n "$n_steps_kl_warmup" ]     && args+=( --n_steps_kl_warmup     "$n_steps_kl_warmup" )
[ -n "$n_epochs_kl_warmup" ]    && args+=( --n_epochs_kl_warmup    "$n_epochs_kl_warmup" )
[ -n "$datasplitter_kwargs" ]   && args+=( --datasplitter_kwargs   "$datasplitter_kwargs" )
[ -n "$plan_kwargs" ]           && args+=( --plan_kwargs           "$plan_kwargs" )

# UMAP colors
[ -n "$UMAP_COLORS" ] && args+=( --umap_colors "$UMAP_COLORS" )

# run pipeline
python splicevi_pipeline.py "${args[@]}"
