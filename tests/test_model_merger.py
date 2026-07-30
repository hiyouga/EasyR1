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

import transformers.modeling_utils
from transformers import AutoModelForCausalLM, GenerationConfig, GPT2Config

from scripts.model_merger import save_pretrained_with_generation_config


def test_save_pretrained_preserves_checkpoint_generation_config(tmp_path, monkeypatch):
    monkeypatch.setattr(transformers.modeling_utils, "unwrap_model", lambda model: model)
    config = GPT2Config(
        architectures=["GPT2LMHeadModel"],
        bos_token_id=0,
        eos_token_id=1,
        n_embd=8,
        n_head=1,
        n_layer=1,
        vocab_size=16,
    )
    config.save_pretrained(tmp_path)
    GenerationConfig(eos_token_id=[1, 2], max_new_tokens=128, pad_token_id=1).save_pretrained(tmp_path)

    original_generation_config = GenerationConfig.from_pretrained(tmp_path)
    model = AutoModelForCausalLM.from_config(config)
    assert model.generation_config.eos_token_id == 1

    save_pretrained_with_generation_config(model, tmp_path, original_generation_config)

    saved_generation_config = GenerationConfig.from_pretrained(tmp_path)
    assert saved_generation_config.eos_token_id == [1, 2]
    assert saved_generation_config.max_new_tokens == 128
    assert saved_generation_config.pad_token_id == 1


def test_save_pretrained_without_checkpoint_generation_config(tmp_path, monkeypatch):
    monkeypatch.setattr(transformers.modeling_utils, "unwrap_model", lambda model: model)
    config = GPT2Config(
        architectures=["GPT2LMHeadModel"],
        bos_token_id=0,
        eos_token_id=1,
        n_embd=8,
        n_head=1,
        n_layer=1,
        vocab_size=16,
    )
    model = AutoModelForCausalLM.from_config(config)

    save_pretrained_with_generation_config(model, tmp_path, None)

    saved_generation_config = GenerationConfig.from_pretrained(tmp_path)
    assert saved_generation_config.eos_token_id == 1
