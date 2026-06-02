
## E=16 정식 Baseline Accuracy Run 결과

Run:
- e16-accuracy-baseline-1000steps-tfdsfix
- https://wandb.ai/yonsei2026dl10-yonsei-university/vmoe-baseline/runs/mcy0t5r7

Setting:
- Official router baseline
- Adapter disabled
- E=16
- 4 GPUs
- 4 experts per GPU
- train_steps = 1000
- batch_size = 64
- train split = train
- validation split = validation

Observed:
- test/images_per_second = 4616.54883
- steps_per_sec = 2.00309
- gpu/memory_used_ratio_mean = 0.23886
- gpu/utilization_mean = 100

Accuracy:
- test/prec@1 = TODO
- test/prec@5 = TODO
- test/loss = TODO
