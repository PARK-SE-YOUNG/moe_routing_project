#!/usr/bin/env bash

export VIT_JAX_ROOT=/workspace/vision_transformer
export PYTHONPATH="/workspace/moe_routing_project:${VIT_JAX_ROOT}:${PYTHONPATH:-}"

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

export VMOE_USE_WANDB=1
export WANDB_MODE=online
export WANDB_ENTITY=yonsei2026dl10-yonsei-university
export WANDB_PROJECT=vmoe-baseline
