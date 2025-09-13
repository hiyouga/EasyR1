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

import pytest
import torch
from PIL.Image import Image

from verl.utils.dataset import RLHFDataset
from verl.utils.tokenizer import get_processor, get_tokenizer


@pytest.mark.parametrize("use_fast", [True, False])
def test_image_dataset(use_fast: bool):
    tokenizer = get_tokenizer("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=use_fast)
    processor = get_processor("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=use_fast)
    dataset = RLHFDataset(
        data_path="hiyouga/geometry3k@test",
        tokenizer=tokenizer,
        processor=processor,
        prompt_key="problem",
        answer_key="answer",
        image_key="images",
        max_prompt_length=16,
        truncation="right",
        filter_overlong_prompts=False,
    )
    token_ids = [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 151652, 151655]
    assert set(dataset[0].keys()) == {
        "input_ids",
        "attention_mask",
        "position_ids",
        "raw_prompt_ids",
        "ground_truth",
        "multi_modal_data",
    }
    assert torch.all(dataset[0]["input_ids"] == torch.tensor(token_ids))
    assert torch.all(dataset[0]["attention_mask"] == torch.ones(16))
    assert torch.all(dataset[0]["position_ids"] == torch.arange(16).unsqueeze(0).expand(4, -1))
    assert list(dataset[0]["position_ids"].size()) == [4, 16]  # avoid fake positive caused by broadcasting
    assert dataset[0]["raw_prompt_ids"] == token_ids
    assert dataset[0]["ground_truth"] == "48"
    assert isinstance(dataset[0]["multi_modal_data"]["images"][0], Image)


if __name__ == "__main__":
    test_image_dataset()
