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

from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizer, ProcessorMixin


def get_tokenizer(model_path: str, override_chat_template: Optional[str] = None, **kwargs) -> PreTrainedTokenizer:
    """Create a huggingface pretrained tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
    if override_chat_template is not None:
        tokenizer.chat_template = override_chat_template

    if tokenizer.bos_token == "<bos>" and tokenizer.eos_token == "<eos>":
        # the EOS token in gemma2 & gemma3 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        print("Found gemma model. Set eos_token and eos_token_id to <end_of_turn> and 107.")
        tokenizer.eos_token = "<end_of_turn>"

    if tokenizer.pad_token_id is None:
        print("Pad token is None. Set it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_processor(model_path: str, override_chat_template: Optional[str] = None, **kwargs) -> Optional[ProcessorMixin]:
    """Create a huggingface pretrained processor."""
    processor = AutoProcessor.from_pretrained(model_path, **kwargs)
    if override_chat_template is not None:
        processor.chat_template = override_chat_template

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.52.4/src/transformers/models/auto/processing_auto.py#L386
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None

    return processor
