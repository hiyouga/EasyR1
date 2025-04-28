from transformers import Qwen2TokenizerFast

class TimeSeriesQwen2TokenizerFast(Qwen2TokenizerFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._add_timeseries_token()

    def _add_timeseries_token(self):
        token = "<|time_series_pad|>"
        if token not in self.get_added_vocab():
            self.add_special_tokens({"additional_special_tokens": [token]})
        self.time_series_token = token
        self.time_series_token_id = self.convert_tokens_to_ids(token)

from transformers import AutoTokenizer
from .configuration_time_series_qwen2_5_vl import TimeSeriesQwen2_5_VLConfig

AutoTokenizer.register(
    config_class=TimeSeriesQwen2_5_VLConfig,
    fast_tokenizer_class=TimeSeriesQwen2TokenizerFast
)