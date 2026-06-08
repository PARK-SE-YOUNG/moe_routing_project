# Agent Handoff Notes

이 repo는 V-MoE fixed top-k routing을 token별 variable-k routing으로 확장하는 실험용 branch 상태다. 다음 agent는 이 파일만 먼저 읽고, 필요한 세부사항은 코드와 로그에서 확인하면 된다.

## Repo / Environment

작업 repo:

```bash
cd /workspace/moe_routing_project
```

RunPod에서 실험/테스트를 돌릴 때는 repo 내부 virtualenv를 먼저 켠 뒤 환경 스크립트를 source한다.

```bash
source .venv/bin/activate
source scripts/env_runpod_vmoe.sh
```

순서가 중요하다. `scripts/env_runpod_vmoe.sh`는 현재 활성화된 Python의 `site-packages/nvidia/*/lib`를 찾아 `LD_LIBRARY_PATH`에 추가한다. `.venv` 활성화 전에 source하면 system Python 기준으로 CUDA path를 잡아서 JAX가 CPU로 fallback할 수 있다.

GPU 확인:

```bash
which python
python -c "import jax; print(jax.devices())"
nvidia-smi
```

정상적으로는 2개 GPU가 보여야 한다.

```text
[CudaDevice(id=0), CudaDevice(id=1)]
```

W&B는 `scripts/env_runpod_vmoe.sh`에서 기본 online 설정을 넣는다.

```text
WANDB_ENTITY=yonsei2026dl10-yonsei-university
WANDB_PROJECT=vmoe-baseline
WANDB_MODE=online
```

W&B login은 이미 되어 있었지만, 새 pod에서는 아래를 먼저 확인한다.

```bash
wandb login
```

## Current Implementation State

구현 방향은 `plan_shift.md`의 V1 이후, PPO 방식으로 바뀐 상태다.

핵심 구현:

- `vmoe/nn/routing.py`
  - `VariableKSubsetPolicyAdapter` 추가.
  - original router가 top-M 후보를 만들고, top-1은 항상 force-select.
  - top-2..top-M subset을 categorical action으로 선택한다.
  - PPO용 action, logprob, value, selected/accepted q mass, miss, k distribution metric을 반환한다.
- `vmoe/moe.py`
  - zero/masked gate 후보가 dispatch와 capacity를 소비하지 않도록 처리.
  - `num_selected_experts=M`과 capacity 기준 `capacity_num_selected_experts=K` 분리.
- `vmoe/train/cleanrl_ppo.py`
  - CleanRL PPO loss 구성요소를 vendoring한 파일.
  - clipped policy loss, clipped value loss, entropy, approx KL, clipfrac 계산.
- `vmoe/train/trainer.py`
  - PPO enabled일 때 collect/update 2단계로 adapter만 학습.
  - reward: `accepted_q_mass * miss_reward_weight + usage_entropy * usage_entropy_weight + latency_reward * latency_weight`.
  - latency reward: `-((wall_time_per_step_secs / latency_baseline_secs) - 1.0)`.
  - `latency_weight > 0`인데 `latency_baseline_secs <= 0`이면 fail fast.
  - train logging은 compact allowlist로 줄여둠.
- `vmoe/train/periodic_actions.py`
  - progress scalar logging을 끌 수 있도록 `write_progress_metrics=False` 지원.
- `vmoe/train/hardware_context.py`
  - GPU/W&B 환경 메타 기록용 helper.

중요한 config:

- `vmoe/configs/vmoe_paper/vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012_variable_k_v1_ft_1000.py`
  - 현재 주 실험 config.
  - 이름은 `ft_1000`이지만 CLI override로 `--config.train_steps=10000` 등 사용 가능.
  - PPO default: `usage_entropy_weight=0.01`, `latency_weight=0.01`, `latency_baseline_secs=0.052`.
  - optimizer trainable pattern은 `VariableKSubsetPolicyAdapter`; pretrained router/expert/head는 freeze 의도.
- 추가 baseline/config 파일:
  - `..._original_topk_v1.py`
  - `..._zero_padded_topm_v1.py`
  - `..._full_topm_v1.py`
  - `..._variable_k_v1.py`
  - `..._variable_k_v1_smoke.py`
  - `..._variable_k_ppo.py`

주의: 현재 worktree에는 여러 수정/신규 파일이 uncommitted 상태다. 기존 사용자 변경을 함부로 revert하지 말 것.

## Running Experiments

기본 10k PPO fine-tuning run:

```bash
cd /workspace/moe_routing_project
source .venv/bin/activate
source scripts/env_runpod_vmoe.sh

RUN_NAME=variable_k_ppo_ft10000_$(date +%Y%m%d_%H%M%S)
WORKDIR=/workspace/moe_routing_project/logs/${RUN_NAME}
mkdir -p "${WORKDIR}"

VMOE_USE_WANDB=1 \
WANDB_MODE=online \
WANDB_PROJECT=vmoe-baseline \
WANDB_ENTITY=yonsei2026dl10-yonsei-university \
WANDB_RUN_NAME="${RUN_NAME}" \
VMOE_JAX_LOG_COMPILES=0 \
VMOE_LOG_PARAMETER_OVERVIEW=0 \
VMOE_SUPPRESS_INIT_MAPPING_WARNINGS=1 \
VMOE_SUPPRESS_CAPACITY_FACTOR_WARNINGS=1 \
python -m vmoe.train.main \
  --config=vmoe/configs/vmoe_paper/vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012_variable_k_v1_ft_1000.py \
  --config.train_steps=10000 \
  --config.profile.first_profile=10000000 \
  --workdir="${WORKDIR}" \
  2>&1 | tee "${WORKDIR}/console.log"
```

tmux로 백그라운드 실행:

```bash
tmux new-session -d -s vmoe_ft10000 "cd /workspace/moe_routing_project && source .venv/bin/activate && source scripts/env_runpod_vmoe.sh && RUN_NAME=variable_k_ppo_ft10000_\$(date +%Y%m%d_%H%M%S) && WORKDIR=/workspace/moe_routing_project/logs/\${RUN_NAME} && mkdir -p \${WORKDIR} && VMOE_USE_WANDB=1 WANDB_MODE=online WANDB_PROJECT=vmoe-baseline WANDB_ENTITY=yonsei2026dl10-yonsei-university WANDB_RUN_NAME=\${RUN_NAME} VMOE_JAX_LOG_COMPILES=0 VMOE_LOG_PARAMETER_OVERVIEW=0 VMOE_SUPPRESS_INIT_MAPPING_WARNINGS=1 VMOE_SUPPRESS_CAPACITY_FACTOR_WARNINGS=1 python -m vmoe.train.main --config=vmoe/configs/vmoe_paper/vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012_variable_k_v1_ft_1000.py --config.train_steps=10000 --config.profile.first_profile=10000000 --workdir=\${WORKDIR} > \${WORKDIR}/console.log 2>&1"
```

실험 변형은 CLI override로 충분하다. 예를 들어 entropy term 제거, latency weight 10배:

```bash
--config.ppo.usage_entropy_weight=0.0 \
--config.ppo.latency_weight=0.1
```

단, 2개 32GB GPU에서 같은 모델 2개를 동시에 띄우면 첫 run이 이미 GPU당 약 24-25GB를 잡는다. 두 번째 run은 compile/restore 시점에 OOM 가능성이 크다. 동시에 돌릴 때는 `nvidia-smi`와 console log를 반드시 확인한다.

## Current / Recent Runs

현재 문서 작성 시점에 돌렸던 주요 run:

- `variable_k_ppo_ft10000_20260608_133809`
  - W&B: `https://wandb.ai/yonsei2026dl10-yonsei-university/vmoe-baseline/runs/g2owchrn`
  - workdir: `/workspace/moe_routing_project/logs/variable_k_ppo_ft10000_20260608_133809`
  - tmux: `vmoe_ft10000_1338`
  - 2026-06-08 14:15 UTC 근처 확인 시 `7500/10000` 진행 중.
  - 5000 step eval: `test/prec@1=0.75968`, `test/prec@5=0.92178`, `val/prec@1=0.78903`, `val/prec@5=0.93732`.
  - routing은 aggregate `avg_selected_k`가 약 4.78, block 5는 약 4.56, block 7은 거의 5.0으로 full top-M 쪽 collapse가 보임.
- `variable_k_ppo_ft10000_noent_latx10_20260608_140450`
  - W&B: `https://wandb.ai/yonsei2026dl10-yonsei-university/vmoe-baseline/runs/219ew8cr`
  - entropy 0, latency weight 0.1 실험.
  - 사용자 요청으로 곧바로 cancel함. W&B sync 완료, tmux 세션 제거됨.

RunPod GPU 할당을 끊으면 tmux와 process는 사라진다. 현재 config는 checkpoint 저장을 사실상 꺼둔 상태라 중간 state 복구는 기대하지 말고, W&B/console log 기록만 남는다고 보면 된다.

## Logs / Metrics

train logging은 너무 많아서 줄여둔 상태다. 핵심적으로 봐야 할 것:

- `train/main_loss`: supervised classification loss.
- `train/rl_loss`: PPO adapter update loss. PPO path에서는 내부 `total_loss`와 같은 값이었으므로 `total_loss`는 train log에서 제거.
- `train/ppo/*`: policy/value update 진단.
- `train/rl_loss/miss_reward_component`: accepted q mass 기반 reward component.
- `train/rl_loss/usage_entropy_reward_component`: expert usage entropy reward component.
- `train/rl_loss/latency_reward_component`: fixed latency baseline 대비 reward component.
- `train/routing/variable_k/avg_selected_k`: MoE block 전체 평균 selected K.
- `train/routing/variable_k/block_5/*`, `block_7/*`: MoE layer별 routing metric. 현재 MoE layer는 encoder block 5와 7.
- `selected_k_ratio/k_1..k_5`: 선택된 K 분포.
- `accepted_q_mass`: capacity 적용 후 살아남은 original top-M probability mass.
- `importance_weighted_miss`: `1 - accepted_q_mass`에 가까운 miss 지표.
- `policy_entropy`: categorical subset policy entropy. usage entropy와 다르다.

제거/비활성화한 train-side logging:

- `train/total_loss`
- `train/ppo_update_time_secs`
- train-side `steps_per_sec`
- train-side FLOPs/compile cost

eval 쪽 `test/compile_secs`, `test/step_flops_per_device` 등은 아직 남아 있을 수 있다.

## Quick Status Commands

tmux/process:

```bash
tmux ls
ps -ef | rg 'vmoe.train.main|variable_k_ppo'
```

latest train metric만 보기:

```bash
perl -ne 'if (/\[(\d+)\] train\//) { $s=$1; ($avg)=/train\/routing\/variable_k\/avg_selected_k=([^, ]+)/; ($b5)=/train\/routing\/variable_k\/block_5\/avg_selected_k=([^, ]+)/; ($b7)=/train\/routing\/variable_k\/block_7\/avg_selected_k=([^, ]+)/; } END { print "latest_train_step=$s avg_k=$avg block5_avg_k=$b5 block7_avg_k=$b7\n" }' logs/<RUN_NAME>/console.log
```

latest eval accuracy:

```bash
perl -ne 'if (/\[(\d+)\].*test\/prec\@1=([^, ]+).*test\/prec\@5=([^, ]+)/) { $ts=$1; $t1=$2; $t5=$3; } if (/\[(\d+)\].*val\/prec\@1=([^, ]+).*val\/prec\@5=([^, ]+)/) { $vs=$1; $v1=$2; $v5=$3; } END { print "latest_test_step=$ts test_prec1=$t1 test_prec5=$t5 latest_val_step=$vs val_prec1=$v1 val_prec5=$v5\n" }' logs/<RUN_NAME>/console.log
```

## Tests

환경 설정 후:

```bash
python vmoe/moe_test.py
python vmoe/nn/routing_test.py
python vmoe/train/cleanrl_ppo_test.py
```

전체/추가 smoke는 시간이 더 걸릴 수 있다.

```bash
python vmoe/nn/vit_moe_test.py
```

`vit_moe_test.py`는 `/workspace/vision_transformer`가 필요하고, 이 경로는 `scripts/env_runpod_vmoe.sh`가 `PYTHONPATH`에 추가한다.

## Known Caveats / Next Things To Check

- 현재 PPO reward가 sparsity를 강하게 유도하지 못해 block 7이 거의 항상 K=5를 고르는 경향이 있다.
- latency reward는 fixed baseline 대비 scalar reward로만 들어가며 router input feature로는 넣지 않는다.
- latency baseline `0.052`는 기존 short run에서 얻은 값이다. hardware/batch/config가 바뀌면 original top-k short run으로 다시 측정해야 한다.
- `usage_entropy`는 expert usage distribution entropy이고, PPO `policy_entropy`와 다르다.
- layer별 expert embedding은 각 MoE router module의 별도 parameter scope 때문에 layer마다 분리된다. global id `layer_id * E + expert_id`를 직접 만들지는 않는다.
- 사용자 관심사는 최종 inference latency contribution이다. training 중 `ppo_update_time_secs`, train throughput/FLOPs는 현재 중요하지 않아 train log에서 뺐다.
