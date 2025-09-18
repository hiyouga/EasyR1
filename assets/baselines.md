# Baselines

Environment: [hiyouga/verl:ngc-th2.7.1-cu12.6-vllm0.10.0](https://hub.docker.com/layers/hiyouga/verl/ngc-th2.7.1-cu12.6-vllm0.10.0/images/sha256-cfc8c1ce3ea52dee0444f3e58e900d0b1d3b6b315deaf5f58c44b5fbb52fa989)

EasyR1 version: [v0.3.2](https://github.com/hiyouga/EasyR1/tree/v0.3.2)

Welcome to contribute new data points!

## Algorithm Baselines

### [Qwen2.5-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) on [Math12k](https://huggingface.co/datasets/hiyouga/math12k)

| Size | Algorithm   | Bits | LR   | KL   | Test Accuracy        |
| ---- | ----------- | ---- | ---- | ---- | -------------------- |
| 7B   | GRPO        | AMP  | 1e-6 | 1e-2 | 0.75 -> 0.77 (+0.02) |

### [Qwen2.5-VL-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) on [Geometry3k](https://huggingface.co/datasets/hiyouga/geometry3k)

| Size | Algorithm   | Bits | LR   | KL   | Test Accuracy        |
| ---- | ----------- | ---- | ---- | ---- | -------------------- |
| 7B   | GRPO        | AMP  | 1e-6 | 1e-2 | 0.37 -> 0.48 (+0.11) |
| 7B   | GRPO        | BF16 | 1e-6 | 1e-2 | 0.37 -> 0.48 (+0.11) |
| 7B   | DAPO        | AMP  | 1e-6 | 1e-2 | 0.37 -> 0.50 (+0.13) |
| 3B   | GRPO        | AMP  | 1e-6 | 1e-2 | 0.24 -> 0.38 (+0.14) |
| 32B  | GRPO        | BF16 | 1e-6 | 1e-2 | 0.50 -> 0.56 (+0.06) |

> [!NOTE]
> The hyper-parameters not listed are all the same as the default values.

## Performance Baselines

### [Qwen2.5-VL-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) on [Geometry3k](https://huggingface.co/datasets/hiyouga/geometry3k)

| Size | GPU Type      | Bits | Batch Size | vLLM TP | Peak Mem | Peak VRAM | Throughput  | Sec per step | Actor MFU |
| ---- | ------------- | ---- | ---------- | ------- | -------- | --------- | ----------- | ------------ | --------- |
| 3B   | 8 * H100 80GB | AMP  | 1 / 2      | 2       | 120GB    | 54GB      | 1800 (+600) | 120s         | 8.1%      |
| 7B   | 8 * H100 80GB | AMP  | 1 / 2      | 2       | 120GB    | 68GB      | 1600 (+400) | 145s         | 16.0%     |
| 7B   | 8 * H100 80GB | AMP  | 4 / 8      | 2       | 200GB    | 72GB      | 2000 (+600) | 120s         | 23.2%     |
| 7B   | 8 * L20 48GB  | AMP  | 1 / 2      | 2       | 120GB    | 42GB      | 410  (+0)   | 580s         | 26.5%     |
| 7B   | 8 * H100 80GB | BF16 | 1 / 2      | 2       | 120GB    | 58GB      | 1600 (+320) | 145s         | 16.0%     |
| 32B  | 8 * H100 80GB | BF16 | 1 / 2      | 8       | 260GB    | 72GB      | 620  (+260) | 530s         | 25.8%     |

- Batch Size: micro_batch_size_per_device_for_update / micro_batch_size_per_device_for_experience
- vLLM TP: rollout.tensor_parallel_size
- Peak Mem: Peak CPU memory usage
- Peak VRAM: Peak GPU memory usage
- Throughput: Number of tokens per second per GPU by one training step (including the improvement compared to the [previous version](https://github.com/hiyouga/EasyR1/blob/v0.3.1/assets/baselines.md))
- Sec per step: Average time per step in seconds

> [!NOTE]
> The hyper-parameters not listed are all the same as the default values.
