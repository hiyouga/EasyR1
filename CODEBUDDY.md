# CODEBUDDY.md This file provides guidance to CodeBuddy Code when working with code in this repository.

## Project Overview

**EasyR1** is an efficient, scalable, multi-modality RL training framework built on top of veRL. It specializes in training vision-language models (VLMs) and language models using reinforcement learning algorithms like GRPO, DAPO, RLOO, etc.

**Key Technologies**: Ray distributed computing, vLLM inference, FSDP training, HybridEngine architecture

## Installation & Setup

```bash
# Clone and install
git clone https://github.com/hiyouga/EasyR1.git
cd EasyR1
pip install -e .

# Docker (recommended)
docker pull hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
docker run -it --ipc=host --gpus=all hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0

# Use ModelScope hub (China)
export USE_MODELSCOPE_HUB=1
```

**Requirements**: Python 3.9+, transformers>=4.54.0, flash-attn>=2.4.3, vllm>=0.8.3

## Core Commands

### Training

```bash
# Single-node training (main entry point)
python3 -m verl.trainer.main config=path/to/config.yaml

# Using example scripts
bash examples/qwen2_5_vl_7b_geo3k_grpo.sh

# Override config parameters
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    trainer.n_gpus_per_node=4 \
    worker.rollout.n=8
```

### Multi-node Training

```bash
# On head node
ray start --head --port=6379 --dashboard-host=0.0.0.0

# On worker nodes
ray start --address=<head_node_ip>:6379

# Check cluster
ray status

# Run training (on head node only)
bash examples/qwen2_5_vl_7b_geo3k_grpo.sh
```

### Checkpoint Management

```bash
# Merge checkpoint to HuggingFace format
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

### Code Quality

```bash
# Format code
make style

# Check quality
make quality

# Run tests
make test
pytest -vv tests/

# Pre-commit hooks
make commit
```

## Architecture Overview

### High-Level Training Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                   verl.trainer.main                     │
│  (Entry point - orchestrates entire RL training loop)   │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴──────────┐
         │   RayPPOTrainer      │  (Ray-based distributed trainer)
         │                      │
         │  ┌────────────────┐  │
         │  │ ResourcePool   │  │  Manages GPU resources across nodes
         │  │ Manager        │  │
         │  └────────────────┘  │
         └──────────┬───────────┘
                    │
    ┌───────────────┼───────────────┐
    │               │               │
    ▼               ▼               ▼
┌────────┐    ┌──────────┐    ┌──────────┐
│ Actor  │    │ Rollout  │    │ Reward   │
│ Worker │    │ Worker   │    │ Manager  │
│ (FSDP) │    │ (vLLM)   │    │          │
└────────┘    └──────────┘    └──────────┘
    │              │               │
    │         Policy Network       │
    │         generates responses  │
    │              │               │
    └──────────────┼───────────────┘
                   │
         Compute advantages & update
```

### Core Components

1. **verl/trainer/**
   - `main.py`: Entry point, sets up Ray cluster and instantiates trainer
   - `ray_trainer.py`: `RayPPOTrainer` - main training loop coordinator
   - `config.py`: `PPOConfig` - configuration dataclass
   - `data_loader.py`: Dataset loading and batching

2. **verl/workers/**
   - `fsdp_workers.py`: `FSDPWorker` - handles Actor/Critic with FSDP
   - `rollout/`: vLLM-based inference for generating responses
   - `reward/`: Reward function execution
   - `actor/`: Policy update logic
   - `critic/`: Value function (unused in GRPO)

3. **verl/single_controller/ray/**
   - `RayWorkerGroup`: Manages Ray remote workers
   - Resource allocation and task scheduling

4. **verl/models/**
   - Model wrappers for transformers
   - Vision-language model support

### Key Design Patterns

**HybridEngine Architecture**: Separates training (FSDP) and inference (vLLM) engines for efficiency
- **Actor Worker**: FSDP for gradient updates
- **Rollout Worker**: vLLM for fast inference generation

**Resource Management**:
- `ResourcePoolManager` assigns workers to GPU pools
- Ray handles distributed execution across nodes
- `Role.ActorRolloutRef`: Combined actor + rollout role

**GRPO Algorithm Flow**:
1. Sample prompts from dataset
2. Rollout: Generate n responses per prompt (e.g., n=4)
3. Compute rewards for each response
4. Calculate group-relative advantages (baseline = group mean)
5. Update policy using PPO-style objective (without Critic)

## Configuration System

Training uses YAML configs with OmegaConf. Structure:

```yaml
data:
  train_files: dataset_path          # HF dataset or local path
  format_prompt: path/to/template.jinja  # Jinja2 prompt template
  rollout_batch_size: 512            # Prompts per rollout
  max_prompt_length: 2048
  
algorithm:
  adv_estimator: grpo                # {grpo, gae, reinforce, ...}
  kl_coef: 0.01                      # KL divergence penalty
  
worker:
  actor:
    global_batch_size: 128           # Total batch for policy update
    model:
      model_path: model_name_or_path
    optim:
      lr: 1.0e-6
    fsdp:
      enable_full_shard: true
    offload:
      offload_params: true           # CPU offloading to save GPU
      
  rollout:
    n: 5                             # Responses per prompt (GRPO key param)
    temperature: 1.0
    gpu_memory_utilization: 0.6
    
  reward:
    reward_function: path/to/reward.py:compute_score
    
trainer:
  total_epochs: 15
  n_gpus_per_node: 8
  nnodes: 1
  save_freq: 5
```

**Override from CLI**:
```bash
python3 -m verl.trainer.main config=base.yaml trainer.n_gpus_per_node=4 worker.rollout.n=8
```

## Reward Functions

Reward functions must follow this interface:

```python
# Required metadata
REWARD_NAME = "your_reward"
REWARD_TYPE = "batch"  # Always "batch"

def compute_score(reward_inputs: list[dict[str, Any]], **kwargs) -> list[dict[str, float]]:
    """
    Args:
        reward_inputs: List of dicts with keys like:
            - response: Model's generated text
            - ground_truth: Expected answer (optional)
            - images: Image data (for VLM tasks)
            - [custom keys from your dataset]
    
    Returns:
        List of dicts with reward components:
            - overall: Main reward (REQUIRED - used for RL update)
            - [other]: Any additional metrics for logging
    """
    scores = []
    for reward_input in reward_inputs:
        # Your reward logic here
        scores.append({
            "overall": computed_reward,
            "accuracy": accuracy_score,
            "format": format_score,
        })
    return scores
```

**Location**: Place in `examples/reward_function/` or custom path, specify in config as `path/to/file.py:function_name`

## Dataset Format

### Text Dataset
```jsonl
{"problem": "What is 2+2?", "answer": "4"}
```

### Image-Text Dataset
```jsonl
{"problem": "What's in this image?", "images": ["data:image/png;base64,..."], "answer": "..."}
```

### Multi-Image Dataset
```jsonl
{"problem": "Compare these images", "images": ["base64_1", "base64_2"], "answer": "..."}
```

**Keys**:
- `prompt_key`: Field for input prompt (default: "problem")
- `answer_key`: Field for ground truth (default: "answer")
- `image_key`: Field for images (default: "images")

**Sources**: HuggingFace datasets (`hiyouga/math12k@train`) or local JSONL files

## Prompt Templates

Use Jinja2 templates in `format_prompt/`:

```jinja
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
{{ prompt }}
{% for image in images %}
<image>{{ image }}</image>
{% endfor %}
<|im_end|>
<|im_start|>assistant
```

Template receives: `prompt` (str), `images` (list), and any custom keys from dataset

## Project-Specific Applications

### number_game_agent/
Android number selection game RL training via ADB
- `agent/number_game_play_agent.py`: VLM-based game agent
- `android_env/adb_controller.py`: Android device control
- `reward_function/number_game_reward.py`: Game-specific rewards
- `config/number_game_grpo.yaml`: Training config

### wechat_tree_game_agent/
WeChat tree-cutting game agent (similar structure)

## Common Issues & Solutions

**"Image features and image tokens do not match"**
→ Increase `data.max_prompt_length` or reduce `data.max_pixels`

**"CUDA Error: out of memory"**
→ Reduce `worker.rollout.gpu_memory_utilization`, enable `worker.actor.offload.offload_params`

**"0 active drivers"**
→ Uninstall `deepspeed` from environment

**HuggingFace connection issues (China)**
→ `export HF_ENDPOINT=https://hf-mirror.com`

**BF16 training to save memory**
→ Set `worker.actor.fsdp.torch_dtype=bf16` and `worker.actor.optim.strategy=adamw_bf16`

## Key Algorithms

Supported via `algorithm.adv_estimator`:
- **GRPO**: Group Relative Policy Optimization (no Critic needed)
- **DAPO**: Data Augmented Policy Optimization (`algorithm.online_filtering=true`)
- **RLOO**: Reinforcement Learning with Likelihood Ratio Optimization
- **GSPO**: Group Sampling Policy Optimization
- **CISPO**: Constrained Sampling Policy Optimization
- **Reinforce++**: Enhanced REINFORCE

See `examples/` for algorithm-specific configs and scripts

## Monitoring

Training logs to:
- **File**: Local logs in experiment directory
- **WandB**: `trainer.logger: ["wandb"]`
- **SwanLab**: `trainer.logger: ["swanlab"]`
- **TensorBoard/Mlflow**: Also supported

Key metrics: `reward/overall`, `policy_loss`, `kl_divergence`, `entropy`

## Notes

- **No SFT/Inference here**: Use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for supervised fine-tuning and inference
- **Vision models + Ulysses**: Not yet compatible
- **LoRA support**: Coming in future updates
- **Checkpoint format**: Use `scripts/model_merger.py` to convert to standard HF format
