# RUNPOD_GUIDE.md

Last Updated: 2026-05-30

---

# Purpose

본 문서는 V-MoE RL Router 프로젝트의 RunPod 환경 구축 및 운영 가이드이다.

목표:

* Multi-GPU V-MoE Baseline Reproduction
* ImageNet Evaluation
* Latency / Throughput Benchmark
* RL Router Experiments

---

# Recommended Infrastructure

## GPU

권장

RTX PRO 4500 x4

## Storage

권장

500GB ~ 1TB Network Volume

이유

* ImageNet Raw : 약 145GB
* Extracted Dataset : 약 150~170GB
* TFDS Build : 약 100~150GB
* Checkpoints / Logs 필요

---

# Required Accounts

## GitHub

Repository

https://github.com/PARK-SE-YOUNG/moe_routing_project

Working Branch

rl-router-baseline-scaffold

---

## RunPod

공용 계정 사용

계정

[yonsei2026dl10@gmail.com](mailto:yonsei2026dl10@gmail.com)

용도

* GPU Server
* Multi-GPU Training
* Baseline Evaluation

---

## WandB

공용 계정 사용

계정

[yonsei2026dl10@gmail.com](mailto:yonsei2026dl10@gmail.com)

Team

yonsei-vmoe-team

Project

vmoe-baseline

---

# SSH Setup

## Generate SSH Key

Windows PowerShell

```bash
ssh-keygen -t ed25519 -C "yonsei2026dl10@gmail.com"
```

---

## Register Public Key

GitHub

Settings
→ SSH and GPG Keys
→ New SSH Key

등록 대상

```text
id_ed25519.pub
```

---

## RunPod SSH Config

파일 위치

```text
C:\Users\<USER>\.ssh\config
```

예시

```ssh
Host runpod-vmoe
    HostName ssh.runpod.io
    User <RUNPOD_USER>
    IdentityFile C:/Users/<USER>/.ssh/id_ed25519
    IdentitiesOnly yes
```

---

## Verify

```bash
ssh runpod-vmoe
```

성공 시 RunPod Banner 출력

---

# VSCode Remote SSH

필수 Extension

Remote - SSH

연결

```text
Ctrl + Shift + P
Remote-SSH: Connect to Host
runpod-vmoe
```

주의

* Linux 선택
* 최초 접속 시 .vscode-server 설치
* 수 분 소요 가능

---

# Repository Setup

```bash
cd /workspace

git clone -b rl-router-baseline-scaffold --single-branch \
https://github.com/PARK-SE-YOUNG/moe_routing_project.git
```

---

# Python Environment

```bash
cd /workspace/moe_routing_project

python -m venv .venv

source .venv/bin/activate
```

---

# JAX GPU Installation

기본 CUDA Driver 확인

```bash
nvidia-smi
```

---

설치

```bash
pip install --upgrade pip

pip install "jax[cuda12]" \
-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

---

검증

```bash
python -c "import jax; print(jax.devices())"
```

기대 결과

```text
[CudaDevice(id=0)]
```

---

# Vision Transformer Setup

```bash
cd /workspace

git clone https://github.com/google-research/vision_transformer.git
```

환경 변수

```bash
export PYTHONPATH=/workspace/vision_transformer:$PYTHONPATH
```

검증

```bash
python -c "import vit_jax"
```

---

# Dependency Validation

```bash
python -c "
import jax
import flax
import optax
import tensorflow
import tensorflow_datasets
print('deps OK')
"
```

---

# Smoke Tests

## Checkpoint Restore

```bash
python checkpoint_restore_smoke.py
```

Expected

```text
Checkpoint restore smoke test passed.
```

---

## Tiny Trainer Step

```bash
python tiny_trainer_step_smoke.py
```

Expected

```text
adapter_changed_count > 0
non_adapter_changed_count = 0
```

의미

* Adapter만 업데이트
* Backbone Freeze
* Experts Freeze
* Original Router Freeze

---

# ImageNet

필수 파일

* ILSVRC2012_img_train.tar
* ILSVRC2012_img_val.tar
* ILSVRC2012_devkit_t12.tar.gz

권장 위치

```text
/data/imagenet/raw
```

---

# TFDS

환경 변수

```bash
export TFDS_DATA_DIR=/data/tensorflow_datasets
export TFDS_MANUAL_DIR=/data/imagenet/raw
```

---

# Baseline Reproduction

목표 Config

```text
vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012.py
```

측정 대상

## Accuracy

* Top-1
* Top-5

## Performance

* Latency
* Throughput

## Routing

* Entropy
* Confidence
* Expert Usage
* Overflow Rate

---

# RunPod Usage Policy

사용 전

팀 채널 공지

예시

사용자: 박세영
목적: Baseline Reproduction
예상 사용시간: 4시간

---

사용 후

반드시

STOP

또는

TERMINATE

확인

---

# Current Status

완료

* RunPod Validation
* JAX GPU Validation
* Checkpoint Restore
* Tiny Trainer Step
* ImageNet Download

다음 단계

1. RTX PRO 4500 x4 확보
2. ImageNet Upload
3. TFDS Build
4. Baseline Accuracy Reproduction
5. Latency Benchmark
6. Throughput Benchmark
7. RL Router Experiments
