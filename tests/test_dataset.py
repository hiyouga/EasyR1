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

import json

import PIL.Image
import pytest
import torch
from PIL.Image import Image

from verl.utils.dataset import RLHFDataset, collate_fn
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


def test_mixed_image_video_dataset(tmp_path):
    image_path = str(tmp_path / "image.png")
    PIL.Image.new("RGB", (56, 56), "red").save(image_path)
    frame_paths = [str(tmp_path / f"frame_{i}.png") for i in range(2)]
    for frame_path in frame_paths:
        PIL.Image.new("RGB", (56, 56), "blue").save(frame_path)

    # a dataset that mixes modalities has both columns in every row
    rows = [
        {"problem": "<image>What color is it?", "answer": "red", "images": [image_path], "videos": []},
        {"problem": "<video>What color is it?", "answer": "blue", "images": [], "videos": [frame_paths]},
        {"problem": "<image><video>Same color?", "answer": "no", "images": [image_path], "videos": [frame_paths]},
    ]
    data_path = tmp_path / "mixed.jsonl"
    data_path.write_text("\n".join(json.dumps(row) for row in rows))
    tokenizer = get_tokenizer("Qwen/Qwen2.5-VL-7B-Instruct")
    processor = get_processor("Qwen/Qwen2.5-VL-7B-Instruct")
    dataset = RLHFDataset(
        data_path=str(data_path),
        tokenizer=tokenizer,
        processor=processor,
        prompt_key="problem",
        answer_key="answer",
        min_pixels=56 * 56,
        max_pixels=56 * 56,
        filter_overlong_prompts=False,
    )
    image_pad_id, video_pad_id = tokenizer.convert_tokens_to_ids(["<|image_pad|>", "<|video_pad|>"])
    image_example, video_example = dataset[0], dataset[1]
    assert list(image_example["multi_modal_data"]) == ["images"]
    assert list(video_example["multi_modal_data"]) == ["videos"]
    assert (image_example["input_ids"] == image_pad_id).any()
    assert not (image_example["input_ids"] == video_pad_id).any()
    assert (video_example["input_ids"] == video_pad_id).any()
    assert not (video_example["input_ids"] == image_pad_id).any()
    assert set(image_example) == set(video_example)
    batch = collate_fn([image_example, video_example])
    assert all(len(value) == 2 for value in batch.values())

    prompt_length = int(video_example["attention_mask"].sum())
    dataset.max_prompt_length = prompt_length
    assert dataset._filter_overlong_prompts(dataset.dataset[1])
    dataset.max_prompt_length = prompt_length - 1
    assert not dataset._filter_overlong_prompts(dataset.dataset[1])

    with pytest.raises(ValueError, match="both images and videos"):
        dataset[2]


if __name__ == "__main__":
    test_image_dataset()
