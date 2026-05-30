# TEAM_ONBOARDING.md

# V-MoE RL Router Project

Official google-research/vmoe 기반 RL Router 연구 프로젝트

---

# 1. Project Goal

본 프로젝트의 최종 목표는 V-MoE(Vision Mixture-of-Experts)의 Router를 Reinforcement Learning 기반으로 Fine-tuning하여,

* Classification Accuracy 유지 또는 향상
* Latency 감소
* Throughput 향상
* Expert Load Balancing 개선

을 동시에 달성하는 것이다.

본 저장소는 공식 google-research/vmoe를 기반으로 한다.

---

# 2. Team Roles

## Baseline Owner

담당 업무

* 공식 V-MoE baseline 재현
* Checkpoint restore 검증
* ImageNet evaluation
* Latency / Throughput 측정
* Router statistics 추출

---

## RL Owner

담당 업무

* RL objective 설계
* RL reward 설계
* PPO / REINFORCE 등 RL 알고리즘 연구
* Router fine-tuning 정책 설계

---

## Experiment Owner

담당 업무

* Hyperparameter 관리
* WandB 실험 관리
* Reproducibility 검증

---

## Presentation Owner

담당 업무

* 실험 결과 정리
* 논문 및 발표자료 제작

---

# 3. Repository Structure

Branch

* main
* rl-router-baseline-scaffold

현재 작업 브랜치

rl-router-baseline-scaffold

---

# 4. Environment Setup

Python

3.10 권장

필수 패키지

* JAX
* Flax
* Optax
* TensorFlow CPU
* TensorFlow Datasets
* vit_jax

---

# 5. vit_jax Setup

Vision Transformer repository clone

git clone https://github.com/google-research/vision_transformer.git

환경 변수 설정

Windows

set PYTHONPATH=C:\vision_transformer;%PYTHONPATH%

PowerShell

$env:PYTHONPATH="C:\vision_transformer;$env:PYTHONPATH"

Linux

export PYTHONPATH=/workspace/vision_transformer:$PYTHONPATH

---

# 6. Baseline Status

완료

* Official V-MoE clone
* Router call path 분석
* Checkpoint initialization 분석
* Adapter insertion
* RL hook scaffold
* Routing metrics
* Latency metrics
* Throughput metrics

진행 예정

* Multi-GPU baseline reproduction
* ImageNet validation
* Real latency benchmark

---

# 7. Router Adapter

추가 위치

vmoe/nn/routing.py

구조

logits_new = logits_original + Δθ(x)

특징

* Adapter-only trainable
* Backbone frozen
* Experts frozen
* Original router frozen

---

# 8. Smoke Tests

## Forward Pass

성공

---

## Tiny Adapter Train Loop

성공

검증 내용

* Gradient 계산
* Optimizer update
* Adapter parameter update

---

## Freeze Policy

성공

검증 내용

* Adapter parameter 변경
* Non-adapter parameter 유지

---

# 9. Checkpoint Compatibility

상태

검증 완료

결과

* Official checkpoint restore 가능
* Adapter parameter 유지
* Initialization path 확인

---

# 10. RL Scaffold

추가 완료

Metrics

* router entropy
* router confidence
* expert usage
* selected log-prob
* KL-to-original-router

Loss Hook

trainer.py 내 RL loss 연결 가능 구조 확보

---

# 11. Evaluation Metrics

Classification

* Top-1 Accuracy
* Top-5 Accuracy

Performance

* Latency
* Throughput

Routing

* Expert Usage
* Expert Load Std
* Entropy
* Confidence

---

# 12. ImageNet

사용 데이터

ILSVRC2012

필요 파일

* ILSVRC2012_img_train.tar
* ILSVRC2012_img_val.tar
* ILSVRC2012_devkit_t12.tar.gz

TFDS Dataset

imagenet2012

---

# 13. WandB

Project

vmoe-baseline

권장 향후 프로젝트

* vmoe-baseline
* vmoe-router-adapter
* vmoe-rl-router

---

# 14. Current Project Status

RL-router Infrastructure

95%

Baseline Integration

95%

Baseline Reproduction

55%

ImageNet Access

완료

ImageNet Download

완료

RunPod Validation

완료

다음 단계

1. RunPod RTX PRO 4500 x4 확보
2. 500GB~1TB Storage 확보
3. ImageNet Upload
4. TFDS Build
5. Baseline Accuracy Reproduction
6. Latency Benchmark
7. Throughput Benchmark
8. RL Router Experiments

# 15. Infrastructure Accounts

## GitHub

Repository

https://github.com/PARK-SE-YOUNG/moe_routing_project

Working Branch

rl-router-baseline-scaffold

---

## RunPod

공용 계정

yonsei2026dl10@gmail.com

용도

* GPU Server
* Baseline Reproduction
* Multi-GPU Experiments

사용 규칙

* 사용 전 팀 공유
* 사용 후 반드시 STOP
* 장시간 미사용 시 TERMINATE

---

## WandB

공용 계정

yonsei2026dl10@gmail.com

Team

yonsei-vmoe-team

Primary Project

vmoe-baseline

용도

* Accuracy Tracking
* Latency Tracking
* Throughput Tracking
* Routing Metrics
* RL Metrics

# 16. RunPod Validation Status

검증 완료

* SSH Connection
* VSCode Remote SSH
* JAX GPU Environment
* Flax
* Optax
* TensorFlow
* TFDS
* vit_jax

결과

JAX GPU

CudaDevice(id=0)

GPU

RTX PRO 4500

상태

Validation Passed

## Checkpoint Restore Smoke

성공

검증 내용

* Official checkpoint restore
* RouterAdapter compatibility
* Initialization path verification

결과

Checkpoint restore smoke test passed.

---

## Tiny Trainer Step Smoke

성공

검증 내용

* train_step 실행
* gradient 계산
* optimizer update
* adapter-only update

결과

adapter_changed_count = 4

non_adapter_changed_count = 0