# Baselines

Environment: [hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.3-flashinfer0.2.2-cxx11abi0](https://hub.docker.com/layers/hiyouga/verl/ngc-th2.6.0-cu126-vllm0.8.3-flashinfer0.2.2-cxx11abi0/images/sha256-335ed6cd1fe73090e458409cfa4394d6abf4cd0503ca44dbafdc28ff72e5ed20)

EasyR1 version: [v0.3.0](https://github.com/hiyouga/EasyR1/tree/v0.3.0)

Welcome to contribute new data points!

## Algorithm Baselines

### [Qwen2.5-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) on [Math12k](https://huggingface.co/datasets/hiyouga/math12k)

| Size | Algorithm   | Bits | LR   | KL   | Test Score |
| ---- | ----------- | ---- | ---- | ---- | ---------- |
| 7B   | GRPO        | AMP  | 1e-6 | 1e-2 | 0.73->0.79 |

### [Qwen2.5-VL-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) on [Geometry3k](https://huggingface.co/datasets/hiyouga/geometry3k)

| Size | Algorithm   | Bits | LR   | KL   | Test Score |
| ---- | ----------- | ---- | ---- | ---- | ---------- |
| 7B   | GRPO        | AMP  | 1e-6 | 1e-2 | 0.39->0.52 |
| 7B   | GRPO        | BF16 | 1e-6 | 1e-2 | 0.39->0.52 |
| 7B   | GRPO        | AMP  | 1e-6 | 1e-3 | 0.39->0.52 |
| 7B   | RLOO        | AMP  | 1e-6 | 1e-2 | 0.39->0.53 |
| 3B   | GRPO        | AMP  | 1e-6 | 1e-2 | 0.27->0.44 |
| 32B  | GRPO        | BF16 | 1e-6 | 1e-2 | 0.46->0.61 |

> [!NOTE]
> The hyper-parameters not listed are all the same as the default values.

## Performance Baselines

### [Qwen2.5-VL-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) on [Geometry3k](https://huggingface.co/datasets/hiyouga/geometry3k)

| Size | GPU Type      | Bits | Batch Size | vLLM Util | vLLM TP | Peak Mem | Peak VRAM | Throughput | Sec per step | Actor MFU |
| ---- | ------------- | ---- | ---------- | --------- | ------- | -------- | --------- | ---------- | ------------ | --------- |
| 3B   | 8 * H100 80GB | AMP  | 4 / 16     | 0.6       | 2       | 120GB    | 35GB      | 1200       | 180s         | 6.3%      |
| 7B   | 8 * H100 80GB | AMP  | 4 / 16     | 0.6       | 2       | 140GB    | 60GB      | 1200       | 180s         | 13.6%     |
| 7B   | 8 * H100 80GB | AMP  | 10 / 20    | 0.6       | 2       | 150GB    | 75GB      | 1400       | 170s         | 19.2%     |
| 7B   | 8 * L20 48GB  | AMP  | 4 / 16     | 0.6       | 2       | 150GB    | 44GB      | 410        | 580s         | 26.5%     |
| 7B   | 8 * H100 80GB | BF16 | 4 / 16     | 0.6       | 2       | 150GB    | 50GB      | 1280       | 190s         | 13.9%     |
| 32B  | 8 * H100 80GB | BF16 | 1 / 8      | 0.6       | 8       | 240GB    | 68GB      | 360        | 860s         | 11.2%     |

- Batch Size: micro_batch_size_per_device_for_update / micro_batch_size_per_device_for_experience
- vLLM Util: rollout.gpu_memory_utilization
- vLLM TP: rollout.tensor_parallel_size
- Peak Mem: Peak CPU memory usage
- Peak VRAM: Peak GPU memory usage
- Throughput: Number of tokens per second per GPU by one training step
- Sec per step: Average time per step in seconds

> [!NOTE]
> The hyper-parameters not listed are all the same as the default values.
