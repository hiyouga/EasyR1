from transformers import Qwen2_5_VLConfig

class TimeSeriesQwen2_5_VLConfig(Qwen2_5_VLConfig):
    model_type = "qwen2_5_vl"
    def __init__(
        self, 
        time_series_token_id=None,  # Unique token for time series
        time_series_embed_dim=None,  # Embedding dimension for time series
        time_series_max_length=None,  # Maximum length of time series
        **kwargs
    ):
        super().__init__(**kwargs)
        self.time_series_token_id = time_series_token_id or 151665
        self.time_series_embed_dim = time_series_embed_dim or self.hidden_size
        self.time_series_max_length = time_series_max_length or 1024