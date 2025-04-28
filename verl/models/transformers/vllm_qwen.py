from typing import TypedDict, Literal, Union

import torch
import torch.nn as nn
from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from vllm.model_executor.models.utils import WeightsMapper, merge_multimodal_embeddings
from vllm.transformers_utils.config import uses_mrope

from verl.models.transformers.time_series_qwen2_5_vl.modeling_time_series_qwen2_5_vl import TimeSeriesEmbedding

class Qwen2_5_VLTimesSeriesDataInputs(TypedDict):
    type: Literal["time-series"]
    time_series_data: torch.Tensor

class Qwen2_5_VLTimesSeriesEmbeddingInputs(TypedDict):
    type: Literal["time-series_embeds"]
    time_series_embeds: torch.Tensor

Qwen2_5_VLTimeSeriesInputs = Union[
    Qwen2_5_VLTimesSeriesDataInputs,
    Qwen2_5_VLTimesSeriesEmbeddingInputs,
]

class TimeSeriesQwen2_5_VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    
    # copied these from vllm.model_executor.models.qwen2_5_vl
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
        "lm_head.": "language_model.lm_head.",
        "model.": "language_model.model.",
    })
    
    def __init__(self, *, vllm_config, prefix):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.time_series_embedding = TimeSeriesEmbedding(2048, torch.bfloat16)

    def _validate_and_reshape_mm_tensor(self, mm_input, name):
        if name == 'time series data': # patch for time series data
            if isinstance(mm_input, list):
                mm_input = torch.stack(mm_input)
            return mm_input.squeeze((1, 2))

        return super()._validate_and_reshape_mm_tensor(mm_input, name)

    def _parse_and_validate_time_series_input(self, **kwargs):
        time_series_data = kwargs.pop("time-series", None)
        time_series_embeds = kwargs.pop("time-series_embeds", None)

        if time_series_data is None and time_series_embeds is None:
            return None

        if time_series_data is not None:
            time_series_data = self._validate_and_reshape_mm_tensor(
                time_series_data, "time series data")
            
            if not isinstance(time_series_data, (torch.Tensor, list)):
                raise ValueError("Incorrect type of time series data. "
                                 f"Got type: {type(time_series_data)}")
            
            return Qwen2_5_VLTimesSeriesDataInputs(
                type="time-series",
                time_series_data=time_series_data,
            )
        
        if time_series_embeds is not None:
            time_series_embeds = self._validate_and_reshape_mm_tensor(
                time_series_embeds, "time series embeds")
            
            if not isinstance(time_series_embeds, (torch.Tensor, list)):
                raise ValueError("Incorrect type of time series embeds. "
                                 f"Got type: {type(time_series_embeds)}")
            
            return Qwen2_5_VLTimesSeriesEmbeddingInputs(
                type="time-series_embeds",
                time_series_embeds=time_series_embeds,
            )
            
    def _process_time_series_input(self, time_series_input):
        if time_series_input["type"] == "time-series_embeds":
            time_series_embeds = time_series_input["time-series_embeds"].type(self.time_series_embedding.dtype)
        else:
            time_series_data = time_series_input["time_series_data"].type(self.time_series_embedding.dtype)
            time_series_embeds = self.time_series_embedding(time_series_data)

        return time_series_embeds.split(1)

    def _parse_and_validate_multimodal_inputs(self, **kwargs):
        mm_input_by_modality = {}

        for input_key in kwargs:
            if input_key in ("pixel_values", "image_embeds"
                             ) and "image" not in mm_input_by_modality:
                mm_input_by_modality[
                    "image"] = self._parse_and_validate_image_input(**kwargs)
            if input_key in ("pixel_values_videos", "video_embeds"
                             ) and "video" not in mm_input_by_modality:
                mm_input_by_modality[
                    "video"] = self._parse_and_validate_video_input(**kwargs)
            if input_key in ("time-series", "time-series_embeds",
                             ) and "time-series" not in mm_input_by_modality:
                mm_input_by_modality[
                    "time-series"] = self._parse_and_validate_time_series_input(**kwargs)
                
        return mm_input_by_modality
    
    def get_multimodal_embeddings(self, **kwargs):
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(
            **kwargs)
        if not mm_input_by_modality:
            return None

        multimodal_embeddings: tuple[torch.Tensor, ...] = ()
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                vision_embeddings = self._process_image_input(multimodal_input)
                multimodal_embeddings += vision_embeddings
            if modality == "video":
                video_embeddings = self._process_video_input(multimodal_input)
                multimodal_embeddings += video_embeddings
            if modality == "time-series":
                time_series_embeddings = self._process_time_series_input(multimodal_input)
                multimodal_embeddings += time_series_embeddings
        return multimodal_embeddings

    def get_input_embeddings_v0(
        self,
        input_ids,
        image_input = None,
        video_input = None,
        time_series_input = None,
    ):
        inputs_embeds = self.get_input_embeddings(input_ids)
        if image_input is not None:
            image_embeds = self._process_image_input(image_input)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                image_embeds,
                placeholder_token_id=self.config.image_token_id,
            )

        if video_input is not None:
            video_embeds = self._process_video_input(video_input)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                video_embeds,
                placeholder_token_id=self.config.video_token_id,
            )

        if time_series_input is not None:
            time_series_embeds = self._process_time_series_input(time_series_input)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                time_series_embeds,
                placeholder_token_id=self.config.time_series_token_id,
            )
        return inputs_embeds
    
    def get_input_embeddings(
        self,
        input_ids,
        multimodal_embeddings=None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                [
                    self.config.image_token_id,
                    self.config.video_token_id,
                    self.config.time_series_token_id,  # add time series token id
                ],
            )
        return inputs_embeds


    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors = None,
        inputs_embeds = None,
        **kwargs,
    ):

        if intermediate_tensors is not None:
            inputs_embeds = None

        elif inputs_embeds is None:
            image_input = self._parse_and_validate_image_input(**kwargs)
            video_input = self._parse_and_validate_video_input(**kwargs)
            time_series_input = self._parse_and_validate_time_series_input(**kwargs)

            if image_input is None and video_input is None and time_series_input is None:
                inputs_embeds = None
            else:
                if uses_mrope(self.config):
                    assert positions.ndim == 2 and positions.size(0) == 3, (
                        "multimodal section rotary embedding requires "
                        f"(3, seq_len) positions, but got {positions.size()}")
                inputs_embeds = self.get_input_embeddings_v0(
                    input_ids,
                    image_input=image_input,
                    video_input=video_input,
                    time_series_input=time_series_input)
                input_ids = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states
