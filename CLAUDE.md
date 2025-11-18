# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

EasyR1 is an efficient, scalable, multi-modality reinforcement learning (RL) training framework. It's a fork of veRL designed to support vision-language models (VLMs) using the HybridEngine architecture and vLLM's SPMD mode.

## Key Commands

### Installation
```bash
pip install -e .
```

### Training
```bash
# Basic GRPO training example
bash examples/qwen2_5_vl_7b_geo3k_grpo.sh

# Generic training command
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=<dataset> \
    worker.actor.model.model_path=<model_path> \
    trainer.experiment_name=<exp_name> \
    trainer.n_gpus_per_node=8
```

### Model Merging
```bash
# Merge checkpoint to Hugging Face format
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

### Testing
```bash
# Run all tests
make test
# Or
pytest -vv tests/

# Run specific test
pytest -vv tests/test_dataproto.py
```

### Code Quality
```bash
# Check code quality
make quality

# Auto-fix formatting and linting
make style

# Check license headers
make license

# Setup pre-commit hooks
make commit
```

### Multi-node Training (Ray)
```bash
# On head node
ray start --head --port=6379 --dashboard-host=0.0.0.0

# On worker nodes
ray start --address=<head_node_ip>:6379

# Check cluster status
ray status

# Run training (only on head node)
bash examples/qwen2_5_vl_7b_geo3k_grpo.sh
```

## Architecture

### Core Components

1. **verl/trainer/** - Main training orchestration
   - `main.py`: Entry point that initializes Ray and runs the RL trainer
   - `ray_trainer.py`: RayPPOTrainer coordinates actor, rollout, and reward workers
   - `config.py`: PPOConfig dataclass defining all training parameters
   - `data_loader.py`: Dataset loading and preprocessing

2. **verl/workers/** - Distributed worker implementations
   - `fsdp_workers.py`: FSDP-based actor and reference model workers
   - `reward/`: Reward function managers (auto-loads from user-defined functions)

3. **verl/protocol.py** - Data transfer protocol
   - `DataProto`: Core data structure for passing batches between workers
   - Uses TensorDict for efficient tensor operations

4. **verl/models/** - Model implementations and patches
   - Monkey patches for transformers models to support RL training

5. **verl/single_controller/ray.py** - RayWorkerGroup for distributed execution

### Training Flow

```
1. Load dataset → DataLoader (with processor/tokenizer)
2. Initialize Ray cluster with resource pools
3. Create workers:
   - ActorRolloutRef: Combined actor model + vLLM rollout engine
   - Critic: (not used in GRPO, only in PPO variants)
   - Reward: User-defined reward function
4. Training loop (RayPPOTrainer.fit()):
   - Sample prompts from dataloader
   - Rollout: Generate N responses per prompt using vLLM
   - Reward: Compute rewards using reward function
   - Update: FSDP-based gradient update on actor model
5. Save checkpoints periodically
```

### Configuration System

Uses OmegaConf for hierarchical YAML configs. Main sections:

- **data**: Dataset paths, batch sizes, prompt/response lengths, image settings
- **algorithm**: RL algorithm (grpo, dapo, reinforce, etc.), KL penalty, advantage estimator
- **worker.actor**: Model path, optimizer, FSDP settings, gradient checkpointing
- **worker.rollout**: Sampling params (n, temperature, top_p), GPU memory, tensor parallelism
- **worker.ref**: Reference model settings (usually CPU-offloaded)
- **worker.reward**: Path to reward function module
- **trainer**: Epochs, logging, checkpointing, validation

CLI overrides: `python -m verl.trainer.main config=base.yaml data.train_files=new_data trainer.experiment_name=exp1`

### Custom Dataset Format

Datasets should be JSONL with fields:
- `prompt`: The question/task (can reference `prompt_key` in config)
- `answer`: Ground truth answer (can reference `answer_key`)
- `images`: List of image paths or URLs (for VLMs)
- `videos`: List of video paths (optional)

Example:
```jsonl
{"prompt": "Solve this geometry problem", "answer": "42", "images": ["path/to/image.jpg"]}
```

See examples: hiyouga/math12k, hiyouga/geometry3k

### Custom Reward Functions

Reward functions are Python modules with a `compute_score` function:

```python
# examples/reward_function/math.py
def compute_score(data: DataProto, **kwargs) -> dict:
    # data contains: prompts, responses, ground_truth
    # Return dict with 'reward' key containing tensor of shape [batch_size]
    rewards = []
    for response, answer in zip(data.responses, data.ground_truth):
        reward = 1.0 if extract_answer(response) == answer else 0.0
        rewards.append(reward)
    return {"reward": torch.tensor(rewards)}
```

Reference in config: `worker.reward.reward_function: ./examples/reward_function/math.py:compute_score`

### Prompt Templates

Jinja2 templates in `examples/format_prompt/` control how prompts are formatted:

```jinja
{# math.jinja #}
Solve this math problem: {{ problem }}
```

Reference in config: `data.format_prompt: ./examples/format_prompt/math.jinja`

## Supported Algorithms

- **GRPO** (Group Relative Policy Optimization): Default, no value network needed
- **DAPO** (Data-Augmented Policy Optimization): With online filtering
- **Reinforce++**: Classic policy gradient with baseline
- **ReMax**: Reward maximization variant
- **RLOO**: Reinforcement Learning with Leave-One-Out
- **GSPO** / **CISPO**: Step-wise policy optimization variants

## Environment Variables

```bash
# Use ModelScope hub instead of HuggingFace
export USE_MODELSCOPE_HUB=1

# Use HF mirror in China
export HF_ENDPOINT=https://hf-mirror.com

# Ray sets these automatically via runtime_env in main.py:
# TOKENIZERS_PARALLELISM=true
# NCCL_DEBUG=WARN
# VLLM_LOGGING_LEVEL=WARN
# TORCH_NCCL_AVOID_RECORD_STREAMS=1
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
```

## Docker

```bash
# Pull pre-built image
docker pull hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
docker run -it --ipc=host --gpus=all hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0

# Or with Apptainer
apptainer pull easyr1.sif docker://hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
apptainer shell --nv --cleanenv --bind /mnt/your_dir:/mnt/your_dir easyr1.sif
```

## Game Agent Projects

The repository includes two Android game agent implementations as examples:

### 1. Number Game Agent (`number_game_agent/`)
- Simple number guessing game for testing VLM agents
- Uses ADB to control Android browser
- Run: `python number_game_agent/agent/number_game_play_agent.py --debug`
- Supports Ollama and vLLM backends

### 2. WeChat Tree Game Agent (`wechat_tree_game_agent/`)
- More complex GRPO training example for WeChat mini-game
- Demonstrates real Android environment integration
- Includes complete data collection, reward function, and training pipeline
- See `wechat_tree_game_agent/README.md` for full implementation guide

Both serve as examples of integrating EasyR1 with real-world interactive environments.

## Important Notes

- **Memory Requirements**: See README.md table for GPU requirements by model size
- **BF16 Training**: Use `worker.actor.fsdp.torch_dtype=bf16` and `worker.actor.optim.strategy=adamw_bf16` to reduce memory
- **Checkpoint Loading**: Set `trainer.find_last_checkpoint=true` to auto-resume from latest checkpoint
- **Validation**: Set `trainer.val_freq=N` to validate every N epochs (set to -1 to disable)
- **Logging**: Supports wandb, swanlab, mlflow, tensorboard via `trainer.logger` (list of strings)
- **vLLM Settings**: Adjust `worker.rollout.gpu_memory_utilization` (default 0.6) and `tensor_parallel_size` based on model size

## Troubleshooting

**Image features and image tokens do not match**
- Increase `data.max_prompt_length` or reduce `data.max_pixels`

**CUDA out of memory in cumem_allocator.cpp**
- Reduce `worker.rollout.gpu_memory_utilization`
- Enable `worker.actor.offload.offload_params=true`

**"0 active drivers" error**
- Uninstall deepspeed: `pip uninstall deepspeed`

## Code Style

- **Formatter**: ruff (line length 119)
- **Python**: 3.9+
- **License Headers**: Apache 2.0 (checked via `make license`)
- **Pre-commit**: Run `make commit` to setup hooks
