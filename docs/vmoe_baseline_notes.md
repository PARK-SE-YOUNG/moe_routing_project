

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

## Router Metrics Smoke Test Results

### Added Metrics

The router now returns additional routing statistics:

- router_entropy
- router_confidence
- expert_usage_min
- expert_usage_max
- expert_usage_std

### Verified Output

Forward smoke test returned metrics from encoderblock_5:

```text
auxiliary_loss
gshard_loss
importance_loss
router_entropy
router_confidence
expert_usage_min
expert_usage_max
expert_usage_std
```

## Evaluation Latency / Throughput Metrics

### evaluator.py Update

Added evaluation system metrics:

- duration_secs
- images_per_second
- latency_per_image

### Metric Formulas

```python
duration_secs = t1 - t0
images_per_second = num_examples / duration_secs
latency_per_image = duration_secs / num_examples
```

## Baseline Eval Command Candidates

### Current Status

Server and dataset path are not finalized yet.

Execution is deferred until GPU/TPU server is available.

### Target Config

```text
vmoe.configs.vmoe_paper.vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012
```
## Adapter-only Optimizer Update Smoke Test

### Result

Passed.

### Key Output

- changed_count = 4
- adapter_changed_count = 4
- non_adapter_changed_count = 0

### Interpretation

Only RouterAdapter parameters were updated by the optimizer.

Backbone, experts, classifier head, and original router parameters remained unchanged.

### Note

The first optimizer step produced no parameter change because the warmup learning-rate schedule starts at zero. Running two update steps confirmed adapter-only updates.

## Overflow / Dropped Token Metric Status

Current router path:

- NoisyTopExpertsPerItemRouter
- vmoe.moe.get_top_experts_per_item_dispatcher

Observation:

- The current TopExpertsPerItem dispatcher path returns dispatcher only.
- It does not directly return overflow or dropped-token metrics.
- No change was made to vmoe/moe.py to avoid modifying dispatcher/expert execution logic.

Current proxy metrics:

- expert_usage_min
- expert_usage_max
- expert_usage_std
- router_entropy
- router_confidence

Decision:

Overflow / dropped-token ratio is deferred until server-side baseline evaluation or until dispatcher-level instrumentation is explicitly needed.

## Server-side Execution Plan

### Current Status

Server, GPU count, dataset path, and log path are not finalized yet.

Local ImageNet TFDS check:

```text
data_dir = C:\Users\janis\tensorflow_datasets\imagenet2012\5.1.0
splits = {}
```

### Interpretation

```text
TFDS ImageNet builder exists locally.
Actual ImageNet train/validation data is not prepared locally.
Local ImageNet subset evaluation is not available yet.
Dummy dataset evaluation is intentionally not used.
```

### Baseline Eval Script

```text
scripts/slurm/vmoe_baseline_eval.sh
```

Expected purpose:

```text
Load official V-MoE checkpoint.
Run ImageNet validation evaluation.
Log top-1 / top-5 accuracy and validation loss.
Log latency / throughput metrics.
Log routing statistics.
```

Required placeholders to fill later:

```text
VMOE_ROOT
VIT_JAX_ROOT
TFDS_DATA_DIR
TFDS_MANUAL_DIR
WORKDIR
GPU count
branch / commit hash
```

### Adapter Smoke Script

```text
scripts/slurm/vmoe_router_adapter_smoke.sh
```

Expected purpose:

```text
Run trainer-based adapter-only smoke test.
Run checkpoint restore smoke test.
Confirm RouterAdapter params are preserved after checkpoint restore.
Confirm non-adapter params remain frozen.
```

## Config Split Status

### Baseline Config

```text
vmoe/configs/vmoe_paper/vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012.py
```

Purpose:

```text
Official baseline reproduction.
No RouterAdapter.
```

### Adapter Smoke Config

```text
vmoe/configs/vmoe_paper/vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012_router_adapter_smoke.py
```

Purpose:

```text
RouterAdapter enabled.
Adapter-only trainable.
Used for local smoke tests and future fine-tuning.
```

## Trainer-based Adapter Step Smoke Test

### Result

Passed.

### Key Output

```text
adapter_changed_count = 4
non_adapter_changed_count = 0
```

### Interpretation

```text
trainer.train_step() successfully runs with the RouterAdapter config.
The RL loss hook is included in total_loss.
Config-based optimizer freeze is applied.
Only RouterAdapter parameters are updated.
Backbone, experts, classifier head, and original router remain frozen.
```

## Checkpoint Restore Smoke Test

### Result

Passed.

### Key Output

```text
restore complete
num_restored_params = 146
num_restored_adapter_params = 8
Checkpoint restore smoke test passed.
```

### Interpretation

```text
Official V-MoE checkpoint parameters can be restored into the adapter-enabled parameter tree.
RouterAdapter parameters are not found in the checkpoint and are preserved from initialization.
This is expected and desired for adapter fine-tuning.
```

## routing_context Scaffold Status

### Result

Passed.

### Current Structure

```text
router(inputs, routing_context=None)
RouterAdapter(inputs, routing_context=None)
```

### Interpretation

```text
Current smoke test still uses Delta_theta(x).
The call path is now backward-compatible and can later be extended to Delta_theta(x, h, l).
hardware_state h and expert_load l are not implemented yet.
```

## Top-5 Accuracy Metric Status

### Result

Implemented.

### Added Metric

```text
prec@5
```

### evaluator.py Update

```text
EvalState now tracks sum_correct_top5.
evaluate_step computes top-5 predictions using jax.lax.top_k.
callback_fn writes {dataset}/prec@5.
```

### Note

```text
Only syntax and static wiring were verified locally.
Full validation requires ImageNet data and server/GPU environment.
```

## Deferred Until Server / Dataset Ready

```text
Full ImageNet validation accuracy.
Actual top-1 / top-5 baseline numbers.
Real latency / throughput measurement.
Single-GPU runtime validation.
Multi-GPU runtime validation.
Slurm execution with real paths.
```
## WandB E=16 Smoke Upload Confirmed

Date: 2026-06-02

Entity:
- yonsei2026dl10-yonsei-university

Project:
- vmoe-baseline

Run:
- e16-4gpu-smoke-wandb-test
- https://wandb.ai/yonsei2026dl10-yonsei-university/vmoe-baseline/runs/qh44hjkp

Confirmed metrics:
- test/prec@1 = 0.02759
- test/prec@5 = 0.07007
- test/loss = 6.90529
- test/images_per_second = 43.39827
- test/latency_per_image = 0.02304
- steps_per_sec = 0.09576
- flops = 58719521996800.0

Conclusion:
- WandB API key and .netrc setup work.
- Correct entity is yonsei2026dl10-yonsei-university.
- V-MoE E=16 smoke metrics are successfully uploaded to WandB.

## GPU Stat Scaffold Confirmed

Date: 2026-06-02

Run:
- e16-4gpu-smoke-gpustat-test
- https://wandb.ai/yonsei2026dl10-yonsei-university/vmoe-baseline/runs/7vg7iogi

Confirmed GPU metrics:
- gpu/memory_total_mb_mean = 32623
- gpu/memory_used_mb_mean = 499
- gpu/memory_used_ratio_mean = 0.0153
- gpu/num_gpus = 4
- gpu/utilization_mean = 0

Conclusion:
- GPU stat helper vmoe/train/gpu_stats.py works.
- WandBMetricWriter successfully appends GPU stats to logged scalar metrics.
- Current GPU utilization is sampled only at metric write time, so utilization may be 0 after eval/training completes.

## Token Representation Summary Scaffold Confirmed

Date: 2026-06-02

Run:
- e16-4gpu-smoke-tokenstat-test
- TODO: paste WandB run URL

Confirmed token summary metrics:
- router_input/token_l2_mean
- router_input/token_l2_std
- router_input/token_abs_mean
- router_input/token_abs_max
- router_input/token_num_groups
- router_input/token_num_tokens
- router_input/token_hidden_dim

Conclusion:
- Router input token representation summary is logged without changing routing decisions.
- This provides the first router-input scaffold signal for later RouterAdapter/RL-router policy inputs.

## Token Representation Summary Scaffold Confirmed

Date: 2026-06-02

Run:
- e16-4gpu-smoke-tokenstat-test
- https://wandb.ai/yonsei2026dl10-yonsei-university/vmoe-baseline/runs/ty4rtfau

Confirmed token summary metrics:
- router_input/token_l2_mean
- router_input/token_l2_std
- router_input/token_abs_mean
- router_input/token_abs_max
- router_input/token_num_groups
- router_input/token_num_tokens
- router_input/token_hidden_dim

Also retained:
- test/prec@1
- test/prec@5
- test/loss
- gpu/memory_used_mb_mean
- gpu/num_gpus
- router_entropy
- router_confidence
- expert_usage_min/max/std
- selected_log_prob

Conclusion:
- Router input token representation summary is logged without changing routing decisions.
- This provides the first router-input scaffold signal for later RouterAdapter/RL-router policy inputs.

## Router Context Metric Scaffold Confirmed

Date: 2026-06-02

Run:
- e16-4gpu-smoke-routercontext-test
- https://wandb.ai/yonsei2026dl10-yonsei-university/vmoe-baseline/runs/zn34annx

Confirmed router context metrics:
- router_context/expert_usage_min
- router_context/expert_usage_max
- router_context/expert_usage_std
- router_context/router_entropy
- router_context/router_confidence
- router_context/selected_log_prob
- router_context/router_kl_to_original

Also retained:
- test/prec@1
- test/prec@5
- test/loss
- test/images_per_second
- test/latency_per_image
- gpu/memory_used_ratio_mean
- router_input/token_l2_mean
- router_input/token_l2_std
- router_input/token_abs_mean
- router_input/token_abs_max

Conclusion:
- Existing expert/router statistics are now mirrored under router_context/*.
- This makes future RouterAdapter/RL-router input candidates explicit while preserving original metric names.

## Context-aware RouterAdapter Smoke Confirmed

Date: 2026-06-02

Run:
- e16-4gpu-smoke-context-adapter-test
- https://wandb.ai/yonsei2026dl10-yonsei-university/vmoe-baseline/runs/bgwntei7

Change:
- RouterAdapter now supports use_context_features=True.
- It derives lightweight token/context features from router inputs.
- Context features are broadcast and concatenated to token representations before the adapter MLP.
- Original router logits remain unchanged; adapter produces delta logits.

Observed metrics:
- gpu/memory_total_mb_mean = 32623
- gpu/memory_used_mb_mean = 25088.1875
- gpu/memory_used_ratio_mean = 0.76903
- gpu/num_gpus = 4
- gpu/utilization_mean = 100
- steps_per_sec = 2.38151
- test/images_per_second = 4598.57324
- test/compile_secs = 5.90231
- test/duration_secs = 0.89071
- flops = 7616026214400.0

Expected adapter input:
- hidden_dim 512 + context_dim 6 = 518
- RouterAdapter/fc1/kernel expected input dimension: 518

Conclusion:
- Context-aware RouterAdapter scaffold is executable on E=16 / 4GPU smoke setting.
- This completes the first executable NN-input scaffold using token-derived context.

## Final Baseline Scaffold Smoke Confirmed

Date: 2026-06-02

Run:
- e16-4gpu-smoke-final-baseline-scaffold-test
- TODO: paste WandB run URL

Confirmed:
- E=16 / 4GPU / 4 experts per GPU smoke execution
- WandB online logging
- accuracy/loss metrics
- latency/throughput metrics
- GPU stats
- token representation stats
- router_context stats
- context-aware RouterAdapter scaffold
- checkpoint save disabled via config flag for smoke runs

Conclusion:
- Baseline scaffold is ready for handoff and further experiments.

## Final Baseline Scaffold Smoke Confirmed

Date: 2026-06-02

Run:
- e16-4gpu-smoke-final-baseline-scaffold-test
- TODO: https://wandb.ai/yonsei2026dl10-yonsei-university/vmoe-baseline/runs/djmujg53

Confirmed:
- E=16 / 4GPU / 4 experts per GPU smoke execution
- WandB online logging
- accuracy/loss metrics
- latency/throughput metrics
- GPU stats
- token representation stats
- router_context stats
- context-aware RouterAdapter scaffold
- checkpoint save disabled via config flag for smoke runs

Conclusion:
- Baseline scaffold is ready for handoff and further experiments.

## E=16 Accuracy Baseline 1000-step Run Confirmed

Date: 2026-06-02

Run:
- e16-accuracy-baseline-1000steps-tfdsfix
- https://wandb.ai/yonsei2026dl10-yonsei-university/vmoe-baseline/runs/mcy0t5r7

Setting:
- Official router baseline
- Adapter disabled
- E=16
- num_expert_partitions=4
- 4 GPUs
- 4 experts per GPU
- train split: train
- validation split: validation
- batch_size: 64
- train_steps: 1000
- checkpoint save disabled via config flag

Observed summary:
- flops = 315456782336000.0
- steps_per_sec = 2.00309
- test/compile_secs = 6.20133
- test/duration_secs = 10.8306
- test/images_per_second = 4616.54883
- gpu/memory_total_mb_mean = 32623
- gpu/memory_used_mb_mean = 7792.25
- gpu/memory_used_ratio_mean = 0.23886
- gpu/num_gpus = 4
- gpu/utilization_mean = 100

Accuracy metrics:
- test/prec@1 = 0.20763999223709104
- test/prec@5 = 0.4370200037956238
- test/loss = 6.7397541999816895

Conclusion:
- E=16 official-router baseline accuracy run completed successfully.
- TFDS path issue was fixed by explicitly setting data_dir and manual_dir to /workspace/imagenet/tfds and /workspace/imagenet/raw.

## E=16 Context-aware RouterAdapter 1000-step Matched Run Confirmed

Date: 2026-06-02

Run:
- e16-context-adapter-1000steps
- https://wandb.ai/yonsei2026dl10-yonsei-university/vmoe-baseline/runs/7ihgttm4

Setting:
- E=16
- 4 GPUs
- 4 experts per GPU
- Adapter ON
- use_context_features=True
- train_steps = 1000
- batch_size = 64
- train split: train
- validation split: validation

Final metrics:
- test/prec@1 = 0.21977999806404114
- test/prec@5 = 0.44579997658729553
- test/loss = 6.739166736602783
- test/images_per_second = 4639.58544921875
- test/latency_per_image = 0.00021553647820837796
- steps_per_sec = 2.011138111197917
- gpu/memory_used_ratio_mean = 0.7686658952272937
- gpu/utilization_mean = 100

Comparison against E=16 official-router baseline:
- Baseline Top-1 = 0.20763999223709104
- Context-adapter Top-1 = 0.21977999806404114
- Top-1 delta = +0.01214000582695010

- Baseline Top-5 = 0.4370200037956238
- Context-adapter Top-5 = 0.44579997658729553
- Top-5 delta = +0.00877997279167173

- Baseline latency/image = 0.00021661202481482175
- Context-adapter latency/image = 0.00021553647820837796

Conclusion:
- Context-aware RouterAdapter achieved a small accuracy improvement over the E=16 official-router baseline in the matched 1000-step run.
- Latency and throughput were effectively unchanged.
- GPU memory usage was higher and requires repeated controlled runs before attributing the increase solely to the adapter.

## E16 routing metrics smoke

- Run name: e16-routing-metrics-smoke
- W&B URL: https://wandb.ai/yonsei2026dl10-yonsei-university/vmoe-baseline/runs/at8oma46
- Purpose:
  - Verify routing/* metric namespace.
  - Verify expert token count logging.
  - Verify synced eval timing after block_until_ready().
- Summary:
  - images/s: 4655.18945
  - test/duration_secs: 0.87988
  - steps/s: 2.33283
  - GPU utilization mean: 100
  - GPU memory used ratio mean: 0.76911
- Notes:
  - routing/overflow_ratio and routing/dropped_token_ratio are currently placeholder 0.0.
  - Actual overflow/dropped token metrics require dispatcher-level inspection.
