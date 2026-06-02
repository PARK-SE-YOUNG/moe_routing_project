"""Minimal WandB wrapper for CLU MetricWriter.

This wrapper forwards metrics to the original CLU writer and additionally logs
scalar metrics to Weights & Biases when enabled.

This is intentionally lightweight for RunPod smoke experiments.
"""

from __future__ import annotations
from vmoe.train import gpu_stats

from typing import Any, Mapping, Optional

from absl import logging


class WandBMetricWriter:
  """Wraps a CLU metric writer and mirrors scalar metrics to WandB."""

  def __init__(
      self,
      base_writer,
      *,
      project: str,
      entity: Optional[str] = None,
      name: Optional[str] = None,
      config: Optional[Mapping[str, Any]] = None,
      enabled: bool = True,
  ):
    self._base_writer = base_writer
    self._enabled = enabled
    self._wandb = None

    if enabled:
      try:
        import wandb  # pylint: disable=import-outside-toplevel
        self._wandb = wandb
        wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=dict(config or {}),
        )
        logging.info("WandB initialized: project=%s entity=%s name=%s",
                     project, entity, name)
      except Exception as e:  # pylint: disable=broad-exception-caught
        self._enabled = False
        self._wandb = None
        logging.warning("WandB disabled because initialization failed: %s", e)

  def write_scalars(self, step: int, scalars: Mapping[str, Any]):
    self._base_writer.write_scalars(step, scalars)

    if not self._enabled or self._wandb is None:
      return

    safe_scalars = {}
    for k, v in scalars.items():
      try:
        # Convert JAX/NumPy scalar-like values to Python floats when possible.
        if hasattr(v, "item"):
          v = v.item()
        if isinstance(v, (int, float, bool)):
          safe_scalars[k] = v
      except Exception:  # pylint: disable=broad-exception-caught
        continue

    if safe_scalars:
      scalars = dict(scalars)
      safe_scalars.update(gpu_stats.get_gpu_stats())
      self._wandb.log(safe_scalars, step=int(step))

  def write_images(self, step: int, images):
    if hasattr(self._base_writer, "write_images"):
      self._base_writer.write_images(step, images)

  def write_texts(self, step: int, texts):
    if hasattr(self._base_writer, "write_texts"):
      self._base_writer.write_texts(step, texts)

  def write_hparams(self, hparams):
    if hasattr(self._base_writer, "write_hparams"):
      self._base_writer.write_hparams(hparams)

  def flush(self):
    self._base_writer.flush()
    if self._enabled and self._wandb is not None:
      self._wandb.finish()

  def close(self):
    if hasattr(self._base_writer, "close"):
      self._base_writer.close()
    if self._enabled and self._wandb is not None:
      self._wandb.finish()
