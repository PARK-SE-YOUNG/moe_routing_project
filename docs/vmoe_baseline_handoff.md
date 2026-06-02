
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

## E=16 Context-aware RouterAdapter 1000-step Matched Run 결과 확정

Run:
- e16-context-adapter-1000steps
- https://wandb.ai/yonsei2026dl10-yonsei-university/vmoe-baseline/runs/7ihgttm4

Setting:
- E=16
- 4 GPUs
- 4 experts per GPU
- Adapter ON
- use_context_features=True
- train_steps = 1000
- batch_size = 64
- train split = train
- validation split = validation

Final metrics:
- test/prec@1 = 0.21977999806404114
- test/prec@5 = 0.44579997658729553
- test/loss = 6.739166736602783
- test/images_per_second = 4639.58544921875
- test/latency_per_image = 0.00021553647820837796
- steps_per_sec = 2.011138111197917
- gpu/memory_used_ratio_mean = 0.7686658952272937
- gpu/utilization_mean = 100

Comparison with E=16 official-router baseline:

| Run | Adapter | Context | Top-1 | Top-5 | Loss | Images/s | Latency/Image | Steps/s | GPU Mem Ratio |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| e16-accuracy-baseline-1000steps-tfdsfix | OFF | OFF | 0.2076399922 | 0.4370200038 | 6.7397542000 | 4616.548828 | 0.0002166120 | 2.003093 | 0.238858 |
| e16-context-adapter-1000steps | ON | token-derived | 0.2197799981 | 0.4457999766 | 6.7391667366 | 4639.585449 | 0.0002155365 | 2.011138 | 0.768666 |

Delta versus E=16 official-router baseline:
- Top-1: +0.0121400058, approximately +1.21 percentage points
- Top-5: +0.0087799728, approximately +0.88 percentage points
- Loss: -0.0005874634
- Images/s: +23.036621
- Latency/Image: -0.0000010755 sec/image
- Steps/s: +0.008045
- GPU memory ratio: +0.529808

Interpretation:
- The context-aware RouterAdapter run slightly improved Top-1 and Top-5 accuracy compared with the E=16 official-router baseline under the same 1000-step setting.
- Throughput and latency were nearly unchanged; if anything, the context-adapter run was slightly faster in this measurement.
- GPU memory usage was much higher in the context-adapter run. This may reflect JAX/XLA memory allocation behavior as well as adapter/context overhead, so memory overhead should be interpreted cautiously and confirmed with repeated controlled runs.
- This matched run provides the first direct comparison between the official E=16 router baseline and the token-derived context-aware RouterAdapter scaffold.
