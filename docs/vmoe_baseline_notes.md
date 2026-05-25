

# V-MoE Baseline Notes

## 0. Purpose

This document tracks the baseline setup plan for reproducing Google Research V-MoE evaluation and preparing RL-router adapter integration.

Current local PyTorch scaffold is only a toy prototype.

The next goal is:

1. reproduce official V-MoE checkpoint evaluation
2. identify routing/logging call paths
3. insert router adapter with minimal modification
4. prepare RL fine-tuning hooks

Official repository:
https://github.com/google-research/vmoe

Primary objective:
Build an RL-router fine-tuning baseline on top of the official V-MoE codebase instead of implementing a new Sparse MoE model from scratch.

---

## 1. Repository Setup

### Repository

```bash
git clone https://github.com/google-research/vmoe.git

cd vmoe

git checkout -b rl-router-baseline
```

### Current Local Branches

- main
- rl-router-baseline
- inspect-vmoe-baseline

Recommended usage:

- main:
  pristine upstream

- inspect-vmoe-baseline:
  code reading / architecture tracing

- rl-router-baseline:
  actual implementation branch

---

## 2. Install Notes

### Python Environment

```bash
python -m venv .venv
```

Windows activation:

```bash
.venv\Scripts\activate
```

### Dependency Notes

Observed repository structure:

- requirements.txt
- vmoe/
- vmoe/nn/
- vmoe/train/
- vmoe/evaluate/
- vmoe/configs/

### CUDA / JAX Notes

Execution server not finalized yet.

Current phase:
- repository analysis
- baseline anatomy tracing
- adapter insertion planning

Deferred until server setup:
- actual ImageNet evaluation
- multi-GPU runs
- latency benchmarking
- throughput benchmarking

---

## 3. Baseline Objectives

Target baseline goals:

1. restore official V-MoE checkpoint
2. reproduce ImageNet validation evaluation
3. verify multi-GPU inference/evaluation
4. measure latency and throughput
5. log routing statistics
6. insert router adapter
7. support RL fine-tuning later

Current focus:
baseline analysis and hook preparation only.

---

## 4. Expected Minimal Modification Policy

Important principle:

DO NOT:
- implement a new Sparse MoE model
- rewrite dispatcher/moe.py
- build a toy PyTorch baseline
- implement PPO/SAC/REINFORCE immediately

Instead:
- modify official V-MoE minimally
- patch router only
- preserve checkpoint compatibility
- preserve existing dispatcher logic

---

## 5. Repository Structure Findings

Important directories:

- vmoe/nn/
- vmoe/train/
- vmoe/evaluate/
- vmoe/configs/
- vmoe/moe.py

Important files:

- vmoe/nn/routing.py
- vmoe/nn/vit_moe.py
- vmoe/train/trainer.py
- vmoe/evaluate/evaluator.py

---

## 6. routing.py Findings

### Router Classes

- NoisyTopExpertsPerItemRouter
- NoisyTopItemsPerExpertRouter

### Router Logits Path

Current routing structure:

```text
inputs
 -> nn.Dense(num_experts)
 -> gates_logits
 -> softmax
 -> dispatcher
```

### Router Logits Computation

Current implementation pattern:

```python
gates_logits = nn.Dense(
    features=num_experts,
    use_bias=False,
)(inputs)
```

### Adapter Insertion Candidate

Most likely insertion point:

Immediately after gates_logits computation.

Planned modification:

```python
gates_logits_original = Dense(inputs)

delta_logits = RouterAdapter(inputs)

gates_logits_new =
    gates_logits_original + delta_logits
```

### Existing Metrics

Observed existing routing metrics:

- auxiliary_loss
- importance_loss
- gshard_loss
- load_auxiliary_loss

### Dispatcher Path

```text
routing.py
 -> vmoe.moe.get_top_experts_per_item_dispatcher
```

### Important Observation

Dispatcher logic already exists and appears reusable.

Current recommendation:
avoid modifying dispatcher/moe.py initially.

---

## 7. vit_moe.py Findings

### Router Creation Path

```python
MlpMoeBlock.create_router()
```

### Router Instantiation

Observed structure:

```python
router_cls = router_kwargs.pop(
    'name',
    'NoisyTopExpertsPerItemRouter'
)
```

This suggests router class replacement is already supported by config.

### Router Invocation

```python
dispatcher, metrics = self.create_router()(inputs)
```

### Token Representation x

Current router input:

```python
inputs
```

This corresponds to token representation x.

### Metrics Propagation

Current metrics flow:

```text
routing.py metrics
 -> encoderblock metrics
 -> encoder metrics
 -> trainer.py
```

### Sparse MoE Execution

```python
vmoe.moe.sparse_moe_spmd(...)
```

### routing_context Extension Candidate

Current structure:

```python
router(inputs)
```

Possible future extension:

```python
router(
    inputs,
    routing_context={
        "hardware_state": h,
        "expert_load": l,
    }
)
```

Candidate insertion point:

```python
MlpMoeBlock.__call__()
```

---

## 8. trainer.py Findings

### train_step Path

```text
train_step
 -> compute_grads_and_metrics
 -> state.apply_fn
 -> logits, metrics
```

### Existing Loss Structure

Current implementation:

```python
total_loss =
    main_loss
    + auxiliary_loss
```

### RL Loss Hook Candidate

Possible future extension:

```python
total_loss =
    main_loss
    + auxiliary_loss
    + rl_loss
```

### Metrics Propagation

Observed structure:

```python
logits, metrics = state.apply_fn(...)
```

This means routing metrics can propagate into trainer.py.

### Logging Path

Observed logging structure:

```python
progress_hook(
    scalar_metrics=...
)
```

### Candidate RL Metrics

Possible future RL-related metrics:

- selected_expert
- action_logprob
- router_entropy
- KL_to_original_router
- expert_count_mean
- expert_count_std
- overflow_ratio
- dropped_token_ratio
- latency
- throughput
- classification_loss

---

## 9. evaluator.py Findings

### Eval Step Path

```text
make_eval_step_pjit()
 -> evaluate_step()
 -> evaluate_dataset()
```

### Existing Metrics

Observed metrics:

- prec@1
- loss
- duration_secs
- compile_secs
- step_flops_per_device
- step_seconds_per_device

### JAX Sync

Observed synchronization:

```python
block_until_ready()
```

already used after eval_step_pjit.

### Latency / Throughput Hook Candidate

Possible future metrics:

```python
duration_secs = t1 - t0

throughput =
    num_examples / duration_secs

latency_per_image =
    duration_secs / num_examples
```

---

## 10. Adapter Design Candidate

### Goal

Add router adapter without replacing the original router.

### Planned Structure

```python
gates_logits_new =
    gates_logits_original
    + Delta_theta(x)
```

### Initialization Strategy

Recommended:
- zero-init final layer

Goal:

```text
Delta_theta(x) ≈ 0
```

at initialization.

This helps preserve baseline checkpoint behavior initially.

---

## 11. Freeze Strategy Candidate

Initial PoC freeze plan:

Frozen:
- patch embedding
- transformer attention blocks
- MoE experts
- classifier head
- original router

Trainable:
- router adapter only

Possible config pattern:

```python
config.optimizer.trainable_pattern =
    'Adapter|RouterAdapter|RLRouter'
```

---

## 12. Dispatcher Modification Policy

Current recommendation:

DO NOT modify dispatcher/moe.py initially.

Reason:
- existing dispatch/combine logic already works
- safer checkpoint compatibility
- lower risk baseline reproduction
- router-only modification is simpler

---

## 13. RL Hook Candidate Summary

Potential RL-related hook locations:

### routing.py

Potential future additions:
- action logprob
- entropy
- KL divergence
- expert statistics

### vit_moe.py

Potential future additions:
- routing_context propagation
- metric propagation

### trainer.py

Potential future additions:
- RL loss aggregation
- RL metric logging

### evaluator.py

Potential future additions:
- latency benchmarking
- throughput benchmarking
- evaluation profiling

---

## 14. Current Phase Status

Current completed work:

- official repository cloned
- branch structure prepared
- routing call path identified
- trainer/evaluator hooks identified
- adapter insertion point identified
- RL extension path identified

Current pending work:

- official checkpoint restore
- ImageNet evaluation reproduction
- actual latency benchmarking
- routing statistics logging
- adapter smoke test
- freeze configuration
- multi-GPU evaluation

---

## 15. Immediate Next Steps

Planned next steps:

1. inspect vmoe/configs/
2. identify official evaluation config
3. identify checkpoint restore path
4. identify optimizer freeze configuration
5. prepare smoke-test config
6. prepare router adapter patch plan
7. prepare baseline evaluation scripts

## Config Findings

### Baseline Candidate Configs

- vmoe_b16_imagenet21k_randaug_strong_ft_ilsvrc2012.py
- vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012.py

### Checkpoint Restore Path

config.initialization.prefix

Example:
gs://vmoe_checkpoints/...

### Router Config Path

config.encoder.moe.router

### Dispatcher Config Path

config.encoder.moe.router.dispatcher

### Important Dispatcher Parameters

- capacity_factor
- partition_spec
- batch_priority

### Important Observation

Router appears fully config-driven.
Possible future router replacement through config.

## optimizer.py Findings

### Freeze Support

optimizer.py already supports:

- trainable_pattern
- frozen_pattern

### Freeze Mechanism

The optimizer can zero out gradients based on regex matching.

### Adapter-only Training Strategy

Use trainable_pattern so that only adapter parameters are trainable.

Possible config:

```python
config.optimizer.trainable_pattern = 'RouterAdapter|Adapter|RLRouter'

## routing.py Detailed Findings

### Existing Dense Parameter Name

Observed router dense layer:

```python
gates_logits = nn.Dense(
    features=num_experts,
    use_bias=False,
    dtype=dtype,
    name="dense",
)(inputs)
```

Important implication:

- existing checkpoint parameter path likely depends on:
  Router/dense

Recommendation:
DO NOT rename existing dense layer.

### Recommended Adapter Patch

```python
gates_logits = nn.Dense(
    ...,
    name="dense",
)(inputs)

delta_logits = RouterAdapter(
    name="RouterAdapter",
)(inputs)

gates_logits =
    gates_logits + delta_logits
```

### Checkpoint Compatibility Strategy

Keep original dense parameters unchanged.

Only initialize:
- RouterAdapter/*

### Optimizer Compatibility

Possible future config:

```python
config.optimizer.trainable_pattern =
    "RouterAdapter"
```

### RL Metric Candidate Location

Immediately after:

```python
gates_softmax = jax.nn.softmax(gates_logits)
```

Possible future metrics:
- entropy
- KL divergence
- selected expert
- action logprob

## Adapter Smoke Test Results

### Config Load

Passed.

- adapter.hidden_dim = 64
- optimizer.trainable_pattern = RouterAdapter

### Model Initialization

Passed on CPU with dummy input:

- input shape: (8, 384, 384, 3)
- logits shape: (8, 1000)

### RouterAdapter Parameters

Detected in:

- Encoder/encoderblock_5/Moe/Router/RouterAdapter
- Encoder/encoderblock_7/Moe/Router/RouterAdapter

### Zero Initialization

Passed.

RouterAdapter/fc2 parameters are initialized to zero.

### Forward Pass

Passed.

Metrics returned:

- encoderblock_5
- encoderblock_7
- auxiliary_loss

### Adapter-only Trainable Check

Passed.

- trainable_count = 8
- frozen_count = 138

Only RouterAdapter parameters matched trainable_pattern.