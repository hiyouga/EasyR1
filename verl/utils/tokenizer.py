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
"""Utils for tokenization."""

from typing import Optional

from transformers import AutoConfig, AutoProcessor, AutoTokenizer, PreTrainedTokenizer, ProcessorMixin


def get_tokenizer(model_path, correct_pad_token=True, correct_gemma=True, **kwargs) -> PreTrainedTokenizer:
    """Create a huggingface pretrained tokenizer.

    Args:
        name (str): The name of the tokenizer.
        correct_pad_token (bool): Whether to correct the pad token id.
        correct_gemma (bool): Whether to correct the gemma tokenizer.
        **kwargs: The keyword arguments for the tokenizer.

    Returns:
        transformers.PreTrainedTokenizer: The pretrained tokenizer.

    """
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)

    if correct_gemma and getattr(config, "model_type", None) in ["gemma", "gemma2"]:
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        print("Found gemma model. Set eos_token and eos_token_id to <end_of_turn> and 107.")
        tokenizer.eos_token = "<end_of_turn>"

    if correct_pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_processor(model_path, **kwargs) -> Optional[ProcessorMixin]:
    try:
        processor = AutoProcessor.from_pretrained(model_path, **kwargs)
    except Exception:
        processor = None

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None

    return processor
