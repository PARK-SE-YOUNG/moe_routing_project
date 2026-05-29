# RUNPOD_DEPLOYMENT.md

# RunPod Deployment Guide for V-MoE Baseline

## 1. Target Server

Recommended GPU:

```text
RTX PRO 4500 x4

Minimum storage:

500GB

Reason:

ImageNet raw files: ~145GB
TFDS build output: ~150-200GB
Checkpoints / logs / repo: additional space
2. Required Accounts
RunPod account: prepared
GitHub SSH key: prepared
WandB account: prepared
ImageNet access: approved
3. SSH Access

Local SSH config template:

Host runpod-vmoe
    HostName <RUNPOD_IP>
    User root
    Port <RUNPOD_PORT>
    IdentityFile ~/.ssh/id_ed25519

Connect:

ssh runpod-vmoe

VSCode:

Remote-SSH: Connect to Host -> runpod-vmoe
4. Clone Repositories
cd /workspace

git clone git@github.com:PARK-SE-YOUNG/moe_routing_project.git
cd moe_routing_project
git checkout rl-router-baseline-scaffold

cd /workspace
git clone https://github.com/google-research/vision_transformer.git

Set PYTHONPATH:

export PYTHONPATH=/workspace/vision_transformer:$PYTHONPATH
5. Python Environment
cd /workspace/moe_routing_project

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install wandb
6. Verify GPU
nvidia-smi
python -c "import jax; print(jax.devices())"

Expected:

4 GPUs visible
7. WandB Login
wandb login

Use the API key from the shared WandB account.

8. ImageNet Raw Files

Required files:

ILSVRC2012_img_train.tar
ILSVRC2012_img_val.tar
ILSVRC2012_devkit_t12.tar.gz

Recommended server path:

/data/imagenet/raw

Expected structure:

/data/imagenet/raw
├── ILSVRC2012_img_train.tar
├── ILSVRC2012_img_val.tar
├── ILSVRC2012_devkit_t12.tar.gz
9. TFDS Build Path

Recommended:

export TFDS_DATA_DIR=/data/tensorflow_datasets
export TFDS_MANUAL_DIR=/data/imagenet/raw

After building, verify:

python -c "import tensorflow_datasets as tfds; b=tfds.builder('imagenet2012'); print(b.info.splits)"

Expected:

train
validation
10. Baseline Evaluation

Target config:

vmoe/configs/vmoe_paper/vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012.py

Run script:

bash scripts/slurm/vmoe_baseline_eval.sh

Expected metrics:

prec@1
prec@5
loss
duration_secs
images_per_second
latency_per_image
router_entropy
expert_usage_min/max/std
11. Adapter Smoke Test
bash scripts/slurm/vmoe_router_adapter_smoke.sh

Expected:

adapter_changed_count > 0
non_adapter_changed_count = 0
Checkpoint restore smoke test passed.
12. Important Notes

Do not:

- implement a new MoE
- use dummy dataset
- rewrite dispatcher/moe.py
- run PPO/SAC before baseline reproduction

First priority:

Official V-MoE checkpoint + ImageNet validation reproduction