import transformers
from transformers import PreTrainedTokenizer, PreTrainedModel, AutoModelForCausalLM


def show_examples():
    # qwen 2.5 vl
    model = AutoModelForCausalLM