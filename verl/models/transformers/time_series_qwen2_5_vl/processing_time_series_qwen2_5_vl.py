import torch
import numpy as np
from typing import List, Union, Optional
from transformers import Qwen2_5_VLProcessor, BatchFeature, AutoImageProcessor, AutoTokenizer
from transformers.processing_utils import Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.image_utils import ImageInput, VideoInput

class TimeSeriesQwen2_5_VLProcessor(Qwen2_5_VLProcessor):
    """
    Simplified processor to support time series inputs alongside text and vision
    """
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "TimeSeriesQwen2TokenizerFast"
    def __init__(
        self, 
        image_processor=None, 
        tokenizer=None, 
        time_series_token="<|time_series_pad|>",
        **kwargs
    ):
        super().__init__(
            image_processor=image_processor, 
            tokenizer=tokenizer,
        )
        self.time_series_token = time_series_token
        
        # Add time series token to tokenizer if not exists
        if not hasattr(self.tokenizer, 'time_series_token'):
            self.tokenizer.add_special_tokens({'additional_special_tokens': [self.time_series_token]})

        # self.chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

        self.chat_template = (
            "{% set image_count = namespace(value=0) %}\n"
            "{% set video_count = namespace(value=0) %}\n"
            "{% for message in messages %}\n"
            "{% if loop.first and message['role'] != 'system' %}"
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "{% endif %}"
            "<|im_start|>{{ message['role'] }}\n"
            "{% if message['content'] is string %}"
            "{{ message['content'] }}<|im_end|>\n"
            "{% else %}"
            "{% for content in message['content'] %}"
            "{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
            "{% set image_count.value = image_count.value + 1 %}"
            "{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}"
            "<|vision_start|><|image_pad|><|vision_end|>"
            "{% elif content['type'] == 'video' or 'video' in content %}"
            "{% set video_count.value = video_count.value + 1 %}"
            "{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}"
            "<|vision_start|><|video_pad|><|vision_end|>"
            "{% elif content['type'] == 'time-series' %}"
            "<|time_series_pad|>"
            "{% elif 'text' in content %}"
            "{{ content['text'] }}"
            "{% endif %}"
            "{% endfor %}<|im_end|>\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        )

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        videos: VideoInput = None,
        time_series_data: Optional[Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]]] = None,
        **kwargs
    ) -> BatchFeature:
        """
        Process inputs including time series data
        
        Args:
            time_series_data (np.ndarray or torch.Tensor): 
                Time series data to be processed. 
                Expected shape: (batch_size, num_features, time_steps)
        """
        # Process existing inputs
        batch_feature = super().__call__(
            text=text, 
            images=images, 
            videos=videos, 
            **kwargs
        )
        
        # Process time series if provided
        if time_series_data is not None:
            # Convert to tensor if needed
            if isinstance(time_series_data, np.ndarray):
                time_series_data = torch.from_numpy(time_series_data)
            
            # Add time series data to batch feature
            batch_feature.data['time_series_data'] = time_series_data
        
        return batch_feature

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load processor from pretrained model
        """
        # Load the base processor
        
        image_processor = AutoImageProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
        )
    
    @property
    def model_input_names(self):
        """
        Update model input names to include time series
        """
        base_names = super().model_input_names
        return base_names + ['time_series']