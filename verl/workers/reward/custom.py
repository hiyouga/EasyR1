# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import wandb
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from ...utils.reward_score import math_compute_score, r1v_compute_score, medical_compute_score


class CustomRewardManager:
    def __init__(self, tokenizer: PreTrainedTokenizer, num_examine: int, compute_score: str,
                 standard_weight: float = 0.7, bbox_weight: float = 0.3):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score_name = compute_score
        self.standard_weight = standard_weight
        self.bbox_weight = bbox_weight

        if compute_score == "math":
            self.compute_score = math_compute_score
        elif compute_score == "r1v":
            self.compute_score = r1v_compute_score
        elif compute_score == "medical":
            self.compute_score = medical_compute_score
        else:
            raise NotImplementedError()

    def __call__(self, data: DataProto) -> torch.Tensor:
        # For medical compute score, we need to output two rewards
        is_medical = self.compute_score_name == "medical"

        # Create reward tensor with the same shape as response tensor for consistency
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        # For medical scores, we'll also track the individual scores for logging
        if is_medical:
            standard_scores = []
            bbox_scores = []

        already_print = 0

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["ground_truth"]

            # Get segmentation mask and bounding box if available
            segmentation_mask = None
            if "segmentation_mask" in data_item.batch:
                segmentation_mask = data_item.batch["segmentation_mask"]

            bbox = None
            if "bbox" in data_item.batch:
                bbox = data_item.batch["bbox"]

            # For medical compute score, pass segmentation mask and bbox
            if is_medical:
                standard_score, bbox_score = self.compute_score(
                    response_str,
                    ground_truth,
                    segmentation_mask=segmentation_mask,
                    bbox=bbox
                )

                # Store individual scores for logging
                standard_scores.append(standard_score)
                bbox_scores.append(bbox_score)

                # Combine scores with weighted average
                combined_score = (self.standard_weight * standard_score +
                                  self.bbox_weight * bbox_score)

                # Store combined score at the last position
                reward_tensor[i, valid_response_length - 1] = combined_score
            else:
                score = self.compute_score(response_str, ground_truth)
                reward_tensor[i, valid_response_length - 1] = score

            if already_print < self.num_examine:
                already_print += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)

                if is_medical:
                    print("[standard_score]", standard_score)
                    print("[bbox_score]", bbox_score)
                    print("[combined_score]", reward_tensor[i, valid_response_length - 1].item())
                else:
                    print("[score]", score)

        # Log metrics to wandb if using medical score
        if is_medical:
            # Calculate averages
            avg_standard_score = sum(standard_scores) / len(standard_scores) if standard_scores else 0
            avg_bbox_score = sum(bbox_scores) / len(bbox_scores) if bbox_scores else 0
            avg_combined_score = (self.standard_weight * avg_standard_score +
                                  self.bbox_weight * avg_bbox_score)

            # Log to wandb
            wandb.log({
                "avg_standard_score": avg_standard_score,
                "avg_bbox_score": avg_bbox_score,
                "avg_combined_score": avg_combined_score
            })

            print(f"Average scores - Standard: {avg_standard_score:.4f}, "
                  f"Bbox: {avg_bbox_score:.4f}, Combined: {avg_combined_score:.4f}")

        return reward_tensor