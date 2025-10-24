#!/usr/bin/env bash
#SBATCH --job-name=Subcluster_Analysis
#SBATCH --output=logs/subcluster_%j.out
#SBATCH --error=logs/subcluster_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1            
#SBATCH --cpus-per-task=8         
#SBATCH --mem=200G               
#SBATCH --time=24:00:00

set -euo pipefail

# 1) Conda environment
CONDA_BASE="/gpfs/commons/home/svaidyanathan/miniconda3"
ENV_NAME="scvi-env"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# 2) Paths
PY="/gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/multivi_splice_utils/runfiles/subcluster_diff_splicing_analysis.py"

MODEL_DIR="/gpfs/commons/home/svaidyanathan/splice_vi_partial_vae_sweep/batch_20251020_102953/mouse_trainandtest_REAL_cd=32_mn=50000_ld=25_lr=1e-5_0_scatter_PartialEncoderEDDI_pool=sum/models"
MUDATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/072025/train_70_30_ge_splice_combined_20250730_164104.h5mu"
BASE_OUTDIR="/gpfs/commons/home/svaidyanathan/analysis/subcluster_outputs"

# 3) Analysis knobs
LEIDEN_RES=1.2
PREFERRED_CTYPE="medium_cell_type"
FALLBACK_CTYPE="broad_cell_type"
TARGET_CELLTYPE="Cortical excitatory neuron"
# Space separated list. Leave empty to ignore tissue filter.
TARGET_TISSUES=()

DE_DELTA=0.25        # for differential expression
DS_DELTA=0.10        # for differential splicing PSI
FDR=0.05
BATCH_SIZE_POST=512
RUN_TSNE=1           # set 0 to skip tSNE

# Layers if you need to override defaults
X_LAYER="junc_ratio"
JUNC_COUNTS_LAYER="cell_by_junction_matrix"
CLUST_COUNTS_LAYER="cell_by_cluster_matrix"
MASK_LAYER="psi_mask"

# 4) Make a logs folder if missing
mkdir -p logs

# 5) Build command
CMD=(python "$PY"
  --model_dir "$MODEL_DIR"
  --mudata_path "$MUDATA_PATH"
  --base_outdir "$BASE_OUTDIR"
  --leiden_resolution "$LEIDEN_RES"
  --preferred_celltype_col "$PREFERRED_CTYPE"
  --fallback_celltype_col "$FALLBACK_CTYPE"
  --target_celltype "$TARGET_CELLTYPE"
  --de_delta "$DE_DELTA"
  --ds_delta "$DS_DELTA"
  --fdr "$FDR"
  --batch_size_post "$BATCH_SIZE_POST"
  --x_layer "$X_LAYER"
  --junction_counts_layer "$JUNC_COUNTS_LAYER"
  --cluster_counts_layer "$CLUST_COUNTS_LAYER"
  --mask_layer "$MASK_LAYER"
)

# Append tissues if provided
if [[ ${#TARGET_TISSUES[@]} -gt 0 ]]; then
  CMD+=(--target_tissues "${TARGET_TISSUES[@]}")
fi

# Optional tSNE switch
if [[ "$RUN_TSNE" == "1" ]]; then
  CMD+=(--run_tsne)
fi

# 6) Echo and run
echo "Running: ${CMD[*]}"
"${CMD[@]}"
