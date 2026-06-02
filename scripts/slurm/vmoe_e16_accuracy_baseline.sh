#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/workspace/moe_routing_project
cd "${REPO_ROOT}"

: "${VIT_JAX_ROOT:=/workspace/vision_transformer}"

export PYTHONPATH="${REPO_ROOT}:${VIT_JAX_ROOT}:${PYTHONPATH:-}"

# JAX CUDA runtime libraries for RunPod reconnect sessions.
export CUDA_PIP_LIBS=$(python - <<'PY'
import site
from pathlib import Path

paths = []
for sp in site.getsitepackages():
    nvidia_root = Path(sp) / "nvidia"
    if nvidia_root.exists():
        for libdir in nvidia_root.glob("*/lib"):
            if libdir.exists():
                paths.append(str(libdir))
print(":".join(paths))
PY
)

export LD_LIBRARY_PATH="${CUDA_PIP_LIBS}:/usr/local/cuda-12.8/targets/x86_64-linux/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

export VMOE_USE_WANDB="${VMOE_USE_WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-yonsei2026dl10-yonsei-university}"
export WANDB_PROJECT="${WANDB_PROJECT:-vmoe-baseline}"
export WANDB_NAME="${WANDB_NAME:-e16-accuracy-baseline-1000steps}"

python -m vmoe.train.main \
  --config=vmoe/configs/vmoe_paper/vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012_e16_accuracy_baseline.py \
  --workdir=/workspace/moe_routing_project/logs/e16_accuracy_baseline
