#!/bin/bash
#SBATCH --job-name=vmoe_adapter_smoke
#SBATCH --output=logs/vmoe_adapter_smoke_%j.out
#SBATCH --error=logs/vmoe_adapter_smoke_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

cd ${VMOE_ROOT:-$PWD}

export PYTHONPATH="${VIT_JAX_ROOT}:${PYTHONPATH:-}"

python tiny_trainer_step_smoke.py
python checkpoint_restore_smoke.py