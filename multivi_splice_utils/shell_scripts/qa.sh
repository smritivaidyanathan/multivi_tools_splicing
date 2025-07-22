#!/usr/bin/env bash
#
#SBATCH --job-name=silhouette
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=300G
#SBATCH --time=04:00:00
#SBATCH --output=logs/silhouette_%j.out
#SBATCH --error=logs/silhouette_%j.err

echo "Running silhouette analysis on $(hostname) at $(date)"
python /gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/multivi_splice_utils/runfiles/quickdifferentialanalysis.py
echo "Done at $(date)"
