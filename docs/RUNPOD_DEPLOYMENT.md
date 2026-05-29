# RUNPOD_DEPLOYMENT.md

# RunPod Deployment Guide for V-MoE Baseline

## 1. Target Server

### Recommended GPU

* RTX PRO 4500 x4

### Minimum Storage

* 500GB

### Reason

* ImageNet raw files: 약 145GB
* TFDS build output: 약 150~200GB
* Checkpoints, logs, repository 저장 공간 필요

---

## 2. Required Accounts

준비 완료 항목

* RunPod Account
* GitHub SSH Key
* WandB Account
* ImageNet Access

---

## 3. SSH Access

### Local SSH Config

파일 위치:

```text
C:\Users\<USER>\.ssh\config
```

예시:

```ssh
Host runpod-vmoe
    HostName <RUNPOD_IP>
    User root
    Port 22
    IdentityFile ~/.ssh/id_ed25519
```

### Connect

```bash
ssh runpod-vmoe
```

### VSCode

* Remote SSH Extension 설치
* Remote-SSH: Connect to Host
* runpod-vmoe 선택

---

## 4. Clone Repositories

### V-MoE Project

```bash
cd /workspace

git clone git@github.com:PARK-SE-YOUNG/moe_routing_project.git

cd moe_routing_project

git checkout rl-router-baseline-scaffold
```

### Vision Transformer

```bash
cd /workspace

git clone https://github.com/google-research/vision_transformer.git
```

### PYTHONPATH

```bash
export PYTHONPATH=/workspace/vision_transformer:$PYTHONPATH
```

---

## 5. Python Environment

```bash
cd /workspace/moe_routing_project

python -m venv .venv

source .venv/bin/activate

pip install --upgrade pip

pip install -r requirements.txt

pip install wandb
```

---

## 6. Verify GPU

### NVIDIA

```bash
nvidia-smi
```

### JAX

```bash
python -c "import jax; print(jax.devices())"
```

Expected:

* 4 GPUs visible

---

## 7. WandB Login

```bash
wandb login
```

사용 계정:

* [yonsei2026dl10@gmail.com](mailto:yonsei2026dl10@gmail.com)

API Key 사용

---

## 8. ImageNet Raw Files

필수 파일

* ILSVRC2012_img_train.tar
* ILSVRC2012_img_val.tar
* ILSVRC2012_devkit_t12.tar.gz

### Recommended Server Path

```text
/data/imagenet/raw
```

예상 구조

```text
/data/imagenet/raw
├── ILSVRC2012_img_train.tar
├── ILSVRC2012_img_val.tar
├── ILSVRC2012_devkit_t12.tar.gz
```

---

## 9. TFDS Build

### Environment Variables

```bash
export TFDS_DATA_DIR=/data/tensorflow_datasets

export TFDS_MANUAL_DIR=/data/imagenet/raw
```

### Verification

```bash
python -c "import tensorflow_datasets as tfds; b=tfds.builder('imagenet2012'); print(b.info.splits)"
```

Expected Output

```text
train
validation
```

---

## 10. Baseline Evaluation

### Target Config

```text
vmoe/configs/vmoe_paper/vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012.py
```

### Run Script

```bash
bash scripts/slurm/vmoe_baseline_eval.sh
```

### Expected Metrics

Classification

* prec@1
* prec@5
* loss

Performance

* duration_secs
* images_per_second
* latency_per_image

Routing

* router_entropy
* expert_usage_min
* expert_usage_max
* expert_usage_std

---

## 11. Adapter Smoke Test

### Run

```bash
bash scripts/slurm/vmoe_router_adapter_smoke.sh
```

### Expected Result

* adapter_changed_count > 0
* non_adapter_changed_count = 0
* checkpoint restore smoke test passed

---

## 12. Important Notes

### Do Not

* Implement a new MoE
* Use dummy datasets
* Rewrite dispatcher/moe.py
* Run PPO/SAC before baseline reproduction

### First Priority

Official V-MoE checkpoint + ImageNet validation reproduction

---

## 13. Deployment Checklist

### Before Starting Server

* ImageNet downloaded
* GitHub SSH verified
* WandB verified
* RunPod credit available

### After Server Creation

* SSH connection successful
* GPU detected
* Repository cloned
* Environment installed
* TFDS build completed
* Baseline evaluation completed

---

## 14. Current Project Goal

1. Official V-MoE Baseline Reproduction
2. Multi-GPU Evaluation
3. Accuracy Measurement
4. Latency Measurement
5. Throughput Measurement
6. Router Statistics Extraction
7. RL Router Fine-Tuning
