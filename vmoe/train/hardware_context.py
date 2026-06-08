"""Hardware/context feature helpers for router fine-tuning."""

from __future__ import annotations

from typing import Mapping

import jax
import jax.numpy as jnp
import ml_collections

from vmoe.train import gpu_stats


# Fixed feature order consumed by RouterAdapter/RL policies. Values are
# normalized so they are numerically comparable with token-derived statistics.
ROUTING_CONTEXT_FEATURES = (
    "gpu_utilization_ratio",
    "gpu_memory_used_ratio",
    "gpu_memory_total_gb_norm",
    "num_gpus_norm",
    "num_expert_partitions_norm",
    "experts_per_partition_norm",
    "batch_size_per_gpu_norm",
    "step_fraction",
)


def _get(config: ml_collections.ConfigDict, path: str, default):
  value = config
  for key in path.split("."):
    if not hasattr(value, "get") or key not in value:
      return default
    value = value[key]
  return value


def make_routing_context(
    config: ml_collections.ConfigDict,
    *,
    step: int,
    train_steps: int,
    batch_size: int,
) -> Mapping[str, jax.Array]:
  """Builds the hardware/context vector passed to the router.

  The returned dict has a stable set of scalar JAX arrays. The same keys are
  passed every step, so pjit compiles once while the scalar values can change.
  """
  stats = gpu_stats.get_gpu_stats(prefix="gpu")

  num_gpus = stats.get("gpu/num_gpus", float(jax.device_count()))
  memory_total_mb = stats.get("gpu/memory_total_mb_mean", 0.0)
  num_experts = float(_get(config, "model.encoder.moe.num_experts", 1))
  num_expert_partitions = float(config.get("num_expert_partitions", num_gpus))
  experts_per_partition = (
      num_experts / num_expert_partitions if num_expert_partitions else 0.0)
  batch_size_per_gpu = float(batch_size) / num_gpus if num_gpus else 0.0
  step_fraction = float(step) / float(max(train_steps, 1))

  values = {
      "gpu_utilization_ratio": stats.get("gpu/utilization_mean", 0.0) / 100.0,
      "gpu_memory_used_ratio": stats.get("gpu/memory_used_ratio_mean", 0.0),
      "gpu_memory_total_gb_norm": (memory_total_mb / 1024.0) / 100.0,
      "num_gpus_norm": num_gpus / 16.0,
      "num_expert_partitions_norm": num_expert_partitions / 16.0,
      "experts_per_partition_norm": experts_per_partition / 16.0,
      "batch_size_per_gpu_norm": batch_size_per_gpu / 256.0,
      "step_fraction": step_fraction,
  }

  return {
      name: jnp.asarray(values.get(name, 0.0), dtype=jnp.float32)
      for name in ROUTING_CONTEXT_FEATURES
  }
