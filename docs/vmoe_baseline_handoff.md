
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
- test/prec@1 = 0.20763999223709104
- test/prec@5 = 0.4370200037956238
- test/loss = 6.7397541999816895

## Note on E=16 Accuracy Baseline Router Metrics

In the E=16 official-router accuracy baseline run:

- Run: e16-accuracy-baseline-1000steps-tfdsfix
- URL: https://wandb.ai/yonsei2026dl10-yonsei-university/vmoe-baseline/runs/mcy0t5r7

The following metrics were successfully recorded:
- test/prec@1
- test/prec@5
- test/loss
- test/images_per_second
- test/latency_per_image
- steps_per_sec
- gpu/memory_used_ratio_mean
- gpu/utilization_mean

However, router/expert metrics were not present in this run summary or history:
- router_entropy
- router_confidence
- selected_log_prob
- expert_usage_min/max/std
- router_context/*

Interpretation:
- This run should be treated as the formal E=16 accuracy/latency baseline.
- A separate matched run is needed if router/expert statistics must be compared under the same 1000-step official-router baseline setting.
