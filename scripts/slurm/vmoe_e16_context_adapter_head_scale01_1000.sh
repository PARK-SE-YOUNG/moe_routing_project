#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/workspace/moe_routing_project
cd "${REPO_ROOT}"

source .venv/bin/activate
source scripts/env_runpod_vmoe.sh

export VMOE_USE_WANDB="${VMOE_USE_WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-yonsei2026dl10-yonsei-university}"
export WANDB_PROJECT="${WANDB_PROJECT:-vmoe-baseline}"
export WANDB_NAME="${WANDB_NAME:-e16-context-adapter-head-scale01-1000steps-debugmetrics}"

python -m vmoe.train.main \
  --config=vmoe/configs/vmoe_paper/vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012_e16_context_adapter_head_scale01_1000.py \
  --workdir=/workspace/moe_routing_project/logs/e16_context_adapter_head_scale01_1000
