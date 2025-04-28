from vllm import LLM

from vllm.model_executor.models.registry import ModelRegistry
from verl.models.transformers.vllm_qwen import TimeSeriesQwen2_5_VLForConditionalGeneration

ModelRegistry.register_model(
    "TimeSeriesQwen2_5_VLForConditionalGeneration",
    TimeSeriesQwen2_5_VLForConditionalGeneration
)

llm = LLM(
    # model="Qwen/Qwen2.5-VL-3B-Instruct",
    model="/home/peili/EasyR1/verl/models/transformers/time_series_qwen2_5_vl",
    skip_tokenizer_init=False,
    tensor_parallel_size=1,
    dtype='bfloat16',
    gpu_memory_utilization=0.6,
    enforce_eager=False,
    max_model_len=4096,
    max_num_batched_tokens=8192,
    enable_sleep_mode=True,
    disable_custom_all_reduce=True,
    disable_mm_preprocessor_cache=True,
    disable_log_stats=True,
    enable_chunked_prefill=False,
    limit_mm_per_prompt={"image": 16},
    trust_remote_code=True,
)

import torch.distributed as dist
if dist.is_initialized():
    dist.destroy_process_group()
