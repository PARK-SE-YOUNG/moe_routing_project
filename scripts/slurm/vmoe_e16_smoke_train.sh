#!/bin/bash
#SBATCH --job-name=vmoe_baseline_eval
#SBATCH --output=logs/vmoe_baseline_eval_%j.out
#SBATCH --error=logs/vmoe_baseline_eval_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

cd ${VMOE_ROOT:-$PWD}

export PYTHONPATH="${VIT_JAX_ROOT}:${PYTHONPATH:-}"
export TFDS_DATA_DIR="${TFDS_DATA_DIR:-/path/to/tfds}"
export TFDS_MANUAL_DIR="${TFDS_MANUAL_DIR:-/path/to/imagenet/manual}"

CONFIG="vmoe/configs/vmoe_paper/vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012_e16_smoke_train.py"
WORKDIR="${WORKDIR:-/path/to/workdir/vmoe_baseline_eval}"

python -m vmoe.train.main \
  --config="${CONFIG}" \
  --workdir="${WORKDIR}"