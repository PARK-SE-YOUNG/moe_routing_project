"""Lightweight GPU statistics helpers for V-MoE experiments."""

from __future__ import annotations

from typing import Dict


def get_gpu_stats(prefix: str = "gpu") -> Dict[str, float]:
  """Returns lightweight aggregate GPU stats.

  This helper is intentionally optional. If NVML is unavailable, it returns
  an empty dict instead of failing the training run.
  """
  try:
    import pynvml  # pylint: disable=import-outside-toplevel
  except Exception:
    return {}

  try:
    pynvml.nvmlInit()
    count = pynvml.nvmlDeviceGetCount()
    if count <= 0:
      return {}

    utils = []
    mem_used = []
    mem_total = []

    for i in range(count):
      handle = pynvml.nvmlDeviceGetHandleByIndex(i)
      util = pynvml.nvmlDeviceGetUtilizationRates(handle)
      mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

      utils.append(float(util.gpu))
      mem_used.append(float(mem.used) / (1024.0 ** 2))
      mem_total.append(float(mem.total) / (1024.0 ** 2))

    used_mean = sum(mem_used) / count
    total_mean = sum(mem_total) / count

    return {
        f"{prefix}/utilization_mean": sum(utils) / count,
        f"{prefix}/memory_used_mb_mean": used_mean,
        f"{prefix}/memory_total_mb_mean": total_mean,
        f"{prefix}/memory_used_ratio_mean": used_mean / total_mean if total_mean else 0.0,
        f"{prefix}/num_gpus": float(count),
    }
  except Exception:
    return {}
