# WANDB_GUIDE.md

Last Updated: 2026-05-30

---

# Purpose

본 문서는 V-MoE RL Router 프로젝트에서 WandB(Weights & Biases)를 사용하는 방법을 정리한다.

WandB의 목적은 실험 결과를 한 곳에 저장하고, Baseline / Router Adapter / RL Router 실험을 비교 가능하게 만드는 것이다.

---

# 1. WandB Account Structure

## Shared Account

공용 계정

* [yonsei2026dl10@gmail.com](mailto:yonsei2026dl10@gmail.com)

이 계정은 RunPod 및 WandB 실험 관리를 위해 사용한다.

---

## Recommended Team

공식 팀 공간

* yonsei-vmoe-team

가능하면 모든 실험은 이 Team 아래에서 관리한다.

---

## Existing Organization / Workspace

현재 WandB에는 다음 Workspace도 존재한다.

* yonsei2026dl10-yonsei-university-org

이는 WandB 가입 과정에서 자동 생성된 Organization / Workspace 성격의 공간이다.

혼선을 줄이기 위해 실험 로그는 가능하면 다음 공간으로 통일한다.

* Team: yonsei-vmoe-team
* Project: vmoe-baseline

---

# 2. Project Structure

권장 Project 구조

## vmoe-baseline

목적

* Official V-MoE checkpoint baseline reproduction
* ImageNet validation accuracy
* Latency / throughput benchmark
* Routing statistics logging

---

## vmoe-router-adapter

목적

* RouterAdapter fine-tuning experiments
* Adapter-only training comparison
* Delta logits analysis

---

## vmoe-rl-router

목적

* PPO
* REINFORCE
* SAC
* finite-horizon RL router experiments
* reward / policy / KL tracking

---

# 3. Login

RunPod 서버에서 실행

```bash
wandb login
```

공용 계정의 API Key를 사용한다.

주의

* API Key는 GitHub에 올리지 않는다.
* API Key는 코드에 하드코딩하지 않는다.
* API Key는 팀 내부 보안 채널에서만 공유한다.

---

# 4. Recommended Environment Variables

RunPod에서 실험 실행 전 설정

```bash
export WANDB_ENTITY=yonsei-vmoe-team
export WANDB_PROJECT=vmoe-baseline
```

Router Adapter 실험 시

```bash
export WANDB_PROJECT=vmoe-router-adapter
```

RL Router 실험 시

```bash
export WANDB_PROJECT=vmoe-rl-router
```

---

# 5. Recommended Run Naming

Run 이름은 다음 형식을 권장한다.

```text
<stage>-<model>-<gpu_count>gpu-<date>
```

예시

```text
baseline-vmoe-s32-4gpu-20260530
adapter-smoke-vmoe-s32-1gpu-20260530
rl-router-ppo-vmoe-s32-4gpu-20260601
```

---

# 6. Recommended Tags

공통 Tag

* vmoe
* imagenet
* baseline
* router
* adapter
* rl-router

GPU 관련 Tag

* 1gpu
* 4gpu
* rtx-pro-4500

실험 단계 Tag

* smoke
* baseline
* adapter
* rl
* latency
* throughput

---

# 7. Metrics to Log

## Classification Metrics

* eval/prec@1
* eval/prec@5
* eval/loss

목적

* Official baseline reproduction
* Router Adapter 성능 비교
* RL Router 성능 비교

---

## Performance Metrics

* system/duration_secs
* system/images_per_second
* system/latency_per_image
* system/step_time

목적

* latency benchmark
* throughput benchmark
* multi-GPU scaling 비교

---

## Routing Metrics

* router/entropy
* router/confidence
* router/expert_usage_min
* router/expert_usage_max
* router/expert_usage_std
* router/selected_log_prob
* router/kl_to_original

목적

* expert load balancing 확인
* RL router policy 변화 확인
* 기존 router 대비 KL 변화 확인

---

## RL Metrics

* rl/reward
* rl/policy_loss
* rl/value_loss
* rl/entropy_bonus
* rl/kl_penalty
* rl/total_loss

목적

* finite-horizon RL objective 추적
* PPO / REINFORCE / SAC 비교

---

## Hardware Metrics

가능하면 기록

* gpu/utilization
* gpu/memory_used
* gpu/power_usage
* gpu/temperature

목적

* hardware-aware routing 실험
* GPU state를 routing_context로 연결하기 위한 분석

---

# 8. Current Usage Plan

## Phase 1: Baseline

Project

* vmoe-baseline

기록 대상

* Top-1 Accuracy
* Top-5 Accuracy
* Validation Loss
* Latency
* Throughput
* Router Entropy
* Expert Usage

---

## Phase 2: Router Adapter

Project

* vmoe-router-adapter

기록 대상

* Adapter-only train loss
* Adapter parameter update
* KL-to-original-router
* selected log-prob
* expert usage 변화

---

## Phase 3: RL Router

Project

* vmoe-rl-router

기록 대상

* Reward
* RL loss
* Policy entropy
* KL penalty
* Latency reward
* Expert load balancing reward

---

# 9. Minimal WandB Code Pattern

Python 예시

```python
import os
import wandb

wandb.init(
    entity=os.environ.get("WANDB_ENTITY", "yonsei-vmoe-team"),
    project=os.environ.get("WANDB_PROJECT", "vmoe-baseline"),
    name=os.environ.get("WANDB_RUN_NAME", None),
    tags=os.environ.get("WANDB_TAGS", "").split(",") if os.environ.get("WANDB_TAGS") else [],
)

wandb.log({
    "eval/prec@1": 0.0,
    "eval/prec@5": 0.0,
    "eval/loss": 0.0,
    "system/images_per_second": 0.0,
    "system/latency_per_image": 0.0,
    "router/entropy": 0.0,
})
```

---

# 10. Important Rules

## Do Not

* API Key를 GitHub에 올리지 않는다.
* 개인 WandB project에 실험을 흩뜨리지 않는다.
* Run 이름 없이 실험하지 않는다.
* baseline / adapter / RL 실험을 같은 project에 무분별하게 섞지 않는다.

---

# 11. Recommended Team Workflow

1. 실험 시작 전 WandB Project 선택
2. Run 이름과 Tag 지정
3. 실험 config 기록
4. 주요 metric logging
5. 실험 종료 후 결과 summary 작성
6. GitHub issue 또는 Notion에 WandB 링크 첨부

---

# 12. Current Status

완료

* 공용 WandB 계정 생성
* Team 생성
* vmoe-baseline project 생성
* wandb package 설치 확인

진행 예정

* baseline eval metric logging
* latency / throughput logging
* routing metric logging
* RL metric logging

---

# 13. Summary

WandB는 본 프로젝트에서 단순한 로그 저장소가 아니라, Baseline / Adapter / RL Router 실험을 비교하는 실험 관리 시스템이다.

특히 향후 PPO, REINFORCE, SAC 실험에서는 reward curve, policy loss, KL penalty, router entropy, latency reward를 비교해야 하므로 WandB 사용은 필수에 가깝다.
