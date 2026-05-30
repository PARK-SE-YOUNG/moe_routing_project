# V-MoE RL Router Project Status

Last Updated: 2026-05-30

---

# Project Overview

Official google-research/vmoe 기반 RL Router 연구 프로젝트

목표:

* Classification Accuracy 유지 또는 향상
* Latency 감소
* Throughput 향상
* Expert Load Balancing 개선

---

# Current Branch

Repository

https://github.com/PARK-SE-YOUNG/moe_routing_project

Working Branch

rl-router-baseline-scaffold

---

# Overall Progress

| 범주                    | 진행률    |
| --------------------- | ------ |
| RL-router 실험 인프라 구축   | 95%    |
| 공식 V-MoE Integration  | 95%    |
| RL-ready Scaffold 구축  | 97%    |
| Baseline Reproduction | 55%    |
| 전체 프로젝트               | 60~65% |

---

# Completed Milestones

## Environment

* Official V-MoE Clone
* JAX GPU Environment
* Flax / Optax
* TensorFlow / TFDS
* vit_jax Integration
* RunPod Validation

## Router Adapter

* RouterAdapter 구현
* Delta Logits Injection
* Adapter-only Training
* Zero-init Final Layer
* Checkpoint Compatibility 유지

## Metrics

* Latency Metric
* Throughput Metric
* Top-5 Accuracy
* Router Entropy
* Router Confidence
* Expert Usage
* Selected Log Prob
* KL-to-Original Router

## RL Scaffold

* RL Loss Hook
* Routing Context Scaffold
* Finite Horizon RL 연결 구조

## Smoke Tests

* model.init
* forward pass
* tiny adapter train loop
* trainer.train_step
* checkpoint_restore_smoke

모두 성공

---

# RunPod Validation Result

GPU

RTX PRO 4500

결과

* SSH 연결 성공
* VSCode Remote 연결 성공
* JAX GPU 인식 성공

JAX Output

CudaDevice(id=0)

Checkpoint Restore

Passed

Tiny Trainer Step

Passed

Adapter-only Update

Passed

---

# Current Status

## 완료

* Official checkpoint restore
* RouterAdapter compatibility
* Adapter-only optimizer
* Tiny train loop
* RL metric scaffold
* RL loss scaffold
* RunPod validation
* ImageNet Access
* ImageNet Download

## 진행 중

* Multi-GPU Environment
* ImageNet TFDS Build
* Baseline Accuracy Reproduction

## 미착수

* PPO
* REINFORCE
* SAC
* Hardware-aware Routing

---

# Next Milestone

## Phase 1

* RunPod RTX PRO 4500 x4 확보
* 500GB~1TB Storage 확보
* ImageNet Upload
* TFDS Build

## Phase 2

* Official Baseline Accuracy Reproduction
* Latency Benchmark
* Throughput Benchmark

## Phase 3

* RL Router Fine-Tuning
* PPO Experiments
* REINFORCE Experiments
* Hardware-aware Routing

---

# Infrastructure

## GitHub

Repository

moe_routing_project

Branch

rl-router-baseline-scaffold

## RunPod

공용 계정 사용

용도

* Multi-GPU Training
* Baseline Reproduction
* RL Experiments

## WandB

Team

yonsei-vmoe-team

Project

vmoe-baseline

용도

* Accuracy Tracking
* Latency Tracking
* Throughput Tracking
* Routing Metrics
* RL Metrics
