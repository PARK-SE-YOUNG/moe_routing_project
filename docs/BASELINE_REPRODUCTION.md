# BASELINE_REPRODUCTION.md

# Official V-MoE Baseline Reproduction Guide

## 1. Purpose

본 문서의 목적은 공식 Google Research V-MoE 모델의 ImageNet-1K Validation 성능을 재현하는 것이다.

본 프로젝트에서 Baseline Reproduction은 모든 후속 실험의 기준점(Baseline)으로 사용된다.

RL Router Adapter, Latency Optimization, Routing Policy 개선 실험은 반드시 Baseline 재현 이후 수행한다.

---

## 2. Target Model

Model:

* V-MoE S/32 Last2

Config:

* vmoe/configs/vmoe_paper/vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012.py

Official Checkpoint:

* gs://vmoe_checkpoints/vmoe_s32_last2_ilsvrc2012_randaug_light1

---

## 3. Success Criteria

Baseline Reproduction 성공 기준

### Checkpoint

* Official checkpoint restore 성공

### Evaluation

* ImageNet validation evaluation 완료

### Metrics

다음 지표 출력 확인

* prec@1
* prec@5
* loss

### Performance

다음 지표 출력 확인

* duration_secs
* images_per_second
* latency_per_image

---

## 4. Required Resources

### Hardware

권장

* RTX PRO 4500 x4

최소

* Multi-GPU 환경

### Storage

권장

* 500GB 이상

---

## 5. Required Dataset

### ImageNet

필수 파일

* ILSVRC2012_img_train.tar
* ILSVRC2012_img_val.tar
* ILSVRC2012_devkit_t12.tar.gz

### TFDS Dataset

Expected Dataset

* imagenet2012

---

## 6. Dataset Verification

### Verify TFDS

실행

```bash
python -c "import tensorflow_datasets as tfds; b=tfds.builder('imagenet2012'); print(b.info.splits)"
```

Expected Output

```text
train
validation
```

성공 시 TFDS Dataset 구축 완료.

---

## 7. Environment Verification

### Verify GPU

```bash
nvidia-smi
```

Expected

* 4 GPU visible

### Verify JAX

```bash
python -c "import jax; print(jax.devices())"
```

Expected

* 4 GPU devices detected

---

## 8. Checkpoint Restore Verification

Checkpoint Restore Smoke Test

```bash
python checkpoint_restore_smoke.py
```

Expected

```text
restore complete
num_restored_params > 0
num_restored_adapter_params > 0
```

---

## 9. Adapter Compatibility Verification

Adapter Smoke Test

```bash
bash scripts/slurm/vmoe_router_adapter_smoke.sh
```

Expected

```text
adapter_changed_count > 0
non_adapter_changed_count = 0
```

Interpretation

* Adapter parameters updated
* Backbone parameters frozen
* Original Router parameters frozen

---

## 10. Baseline Evaluation

### Run

```bash
bash scripts/slurm/vmoe_baseline_eval.sh
```

### Expected Outputs

Classification Metrics

* prec@1
* prec@5
* loss

Performance Metrics

* duration_secs
* images_per_second

Routing Metrics

* router_entropy
* router_confidence
* expert_usage_min
* expert_usage_max
* expert_usage_std

---

## 11. Logging

### WandB

Project

* vmoe-baseline

Log Categories

Classification

* Top1 Accuracy
* Top5 Accuracy
* Loss

Performance

* Latency
* Throughput

Routing

* Entropy
* Confidence
* Expert Usage

---

## 12. Baseline Reproduction Checklist

### Dataset

* [ ] ImageNet downloaded
* [ ] TFDS build completed
* [ ] Dataset verification completed

### Environment

* [ ] GPU detected
* [ ] JAX detected
* [ ] WandB login successful

### Model

* [ ] Checkpoint restored
* [ ] Adapter compatibility verified

### Evaluation

* [ ] Evaluation completed
* [ ] Top1 accuracy recorded
* [ ] Top5 accuracy recorded
* [ ] Latency recorded
* [ ] Throughput recorded

---

## 13. Expected Deliverables

Baseline Report

포함 항목

### Accuracy

* Top1 Accuracy
* Top5 Accuracy

### Latency

* Average Latency
* Images Per Second

### Routing Statistics

* Router Entropy
* Router Confidence
* Expert Usage Distribution

### System Information

* GPU Type
* GPU Count
* Batch Size
* Evaluation Time

---

## 14. Next Step After Reproduction

Baseline 재현 완료 후 수행.

### Router Adapter Experiments

* Adapter Fine-Tuning

### RL Router Experiments

* RL Reward Design
* RL Fine-Tuning
* Routing Optimization

### Multi-GPU Scaling Analysis

* Latency Scaling
* Throughput Scaling
* Expert Load Balancing Analysis

---

## 15. Important Principle

프로젝트의 최우선 목표는 다음과 같다.

1. Official V-MoE Baseline Reproduction
2. ImageNet Validation Accuracy Verification
3. Latency / Throughput Measurement
4. Routing Statistics Extraction

위 4개가 완료되기 전에는 PPO, SAC, REINFORCE 등의 RL 알고리즘 실험을 시작하지 않는다.
