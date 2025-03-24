set -x

/home/dvdai/miniconda3/bin/conda activate test

SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}.

When analyzing medical images, you must identify and outline all objects of interest that are relevant to diagnosis.
For each identified object, provide the following information in JSON format:

[
  {
    \"bbox_2d\": [x1, y1, x2, y2],  // Coordinates in format [x1, y1, x2, y2]
    \"label\": \"object_type\",       // Type of object or abnormality
    \"sub_label\": \"details\"        // Additional details or characteristics
  },
  // Additional objects as needed
]"""

python -m verl.trainer.main \
    config=examples/grpo_climb_engaging.yaml \
    data.system_prompt="${SYSTEM_PROMPT}" \
    worker.actor.model.model_path=Qwen/Qwen2.5-VL-7B-Instruct \
    trainer.n_gpus_per_node=4
