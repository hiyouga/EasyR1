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


from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .transformers.flash_attention_utils import flash_attention_forward
from .transformers.qwen2_vl import qwen2_vl_attn_forward

from typing import Any, Optional, Mapping


def apply_ulysses_patch(model_type: str) -> None:
    if model_type in ("llama", "gemma", "gemma2", "mistral", "qwen2"):
        ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward
    elif model_type in ("qwen2_vl", "qwen2_5_vl", "time_series_qwen2_5_vl"):
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLFlashAttention2
        from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLFlashAttention2

        Qwen2VLFlashAttention2.forward = qwen2_vl_attn_forward
        Qwen2_5_VLFlashAttention2.forward = qwen2_vl_attn_forward
    else:
        raise NotImplementedError(f"Model architecture {model_type} is not supported yet.")
    
def time_series_vllm_patch():
    from vllm.model_executor.models.registry import ModelRegistry
    from verl.models.transformers.vllm_qwen import TimeSeriesQwen2_5_VLForConditionalGeneration

    ModelRegistry.register_model(
        "TimeSeriesQwen2_5_VLForConditionalGeneration",
        TimeSeriesQwen2_5_VLForConditionalGeneration
    )

    import torch
    from vllm.multimodal.parse import MultiModalDataParser, ProcessorBatchItems, ModalityDataItems
    from vllm.multimodal.inputs import MultiModalFieldElem, MultiModalBatchedField, MultiModalKwargsItem

    # 1) Define a ProcessorBatchItems subclass for torch.Tensor time-series
    class TimeSeriesProcessorItems(ProcessorBatchItems[torch.Tensor]):
        def __init__(self, data: list[torch.Tensor]) -> None:
            # data: list of (n_channels, seq_len) tensors
            super().__init__(data, "time-series")
        
        def get_processor_data(self):
            return {}
            
        # def get_passthrough_data(self):
        #     return {"time-series": self.data}

    # 2) Monkey-patch MultiModalDataParser to recognize "time-series"
    _orig_get_subparsers = MultiModalDataParser._get_subparsers

    def _get_subparsers_with_ts(self) -> dict[str, callable]:
        subs = _orig_get_subparsers(self)
        subs["time-series"] = lambda data: _parse_ts(self, data)
        return subs

    def _parse_ts(self, data) -> Optional[ModalityDataItems[Any, Any]]:
        return TimeSeriesProcessorItems(data)

    # apply the patch
    MultiModalDataParser._get_subparsers = _get_subparsers_with_ts

    from vllm.multimodal.processing import BaseMultiModalProcessor
    _orig_validate = BaseMultiModalProcessor._validate_mm_kwargs

    def _validate_skip_ts(self, mm_kwargs, mm_item_counts):
        # drop time-series from the count map so it's never checked
        filtered_counts = {
            modality: cnt
            for modality, cnt in mm_item_counts.items()
            if modality != "time-series"
        }
        return _orig_validate(self, mm_kwargs, filtered_counts)

    # Apply the patch
    BaseMultiModalProcessor._validate_mm_kwargs = _validate_skip_ts

    _orig_validate_placeholders = BaseMultiModalProcessor._validate_mm_placeholders

    # 2) Define a wrapper that skips "time-series"
    def _validate_mm_placeholders_skip_ts(
        self,
        mm_placeholders: Mapping[str, list],
        mm_item_counts: Mapping[str, int],
    ) -> None:
        # Filter out the time-series entry so it's never checked
        filtered_counts = {
            modality: count
            for modality, count in mm_item_counts.items()
            if modality != "time-series"
        }
        # Delegate to the original for everything else
        return _orig_validate_placeholders(self, mm_placeholders, filtered_counts)

    # 3) Apply the monkey-patch
    BaseMultiModalProcessor._validate_mm_placeholders = _validate_mm_placeholders_skip_ts

    _orig_apply = BaseMultiModalProcessor.apply
    def _apply_with_time_series(
        self,
        prompt,
        mm_data,
        hf_processor_mm_kwargs,
        return_mm_hashes=False,
    ):
        multi_inputs = _orig_apply(
            self, prompt, mm_data, hf_processor_mm_kwargs, return_mm_hashes
        )
        # inject your tensor into the kwargs that get passed to get_multimodal_embeddings
        if "time-series" in mm_data:
            ts = mm_data["time-series"]
            multi_inputs["mm_kwargs"]["time-series"] = ts
            multi_inputs["mm_placeholders"]["time-series"] = [{
                "offset": multi_inputs["prompt_token_ids"].index(151665), # <|time_series_pad|>
                "length": 1
            }]
            multi_inputs["mm_kwargs"]._items_by_modality["time-series"] = [MultiModalKwargsItem({
                "time-series": MultiModalFieldElem(
                    modality="time-series",
                    key="time-series",
                    data=ts,
                    field=MultiModalBatchedField()
                )
            })]
        
        return multi_inputs

    BaseMultiModalProcessor.apply = _apply_with_time_series

    # _orig_call = BaseMultiModalProcessor._call_hf_processor

    # def _call_hf_processor_with_ts(self, *args, **kwargs):
    #     # args are (prompt or tokens, mm_data, hf_processor_mm_kwargs)
    #     hf_out = _orig_call(self, *args, **kwargs)
    #     mm_data = kwargs.get("mm_data") or (args[1] if len(args) > 1 else {})
    #     if mm_data and "time-series" in mm_data:
    #         # hf_out.kwargs is a MultiModalKwargs
    #         ts = mm_data["time-series"]
    #         # “add_passthrough” will carry it through to the model
    #         hf_out.kwargs.add_passthrough("time-series", ts)
    #     return hf_out

    # BaseMultiModalProcessor._call_hf_processor = (
    #     _call_hf_processor_with_ts
    # )

    # _orig_get_updates = BaseMultiModalProcessor._get_prompt_updates

    # def _get_prompt_updates_skip_ts(
    #     self,
    #     token_ids,
    #     hf_out,
    #     mm_data,
    #     mm_fields_config,
    # ):
    #     updates = _orig_get_updates(
    #         self, token_ids, hf_out, mm_data, mm_fields_config
    #     )
    #     # filter out any time-series updates
    #     return [u for u in updates if u.modality != "time-series"]

    # BaseMultiModalProcessor._get_prompt_updates = (
    #     _get_prompt_updates_skip_ts
    # )


