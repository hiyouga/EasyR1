# server.py
from threading import Thread
import litserve as ls
import torch
from litserve.specs.openai import ChatCompletionRequest
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    TextIteratorStreamer,
)

# Define your model constants
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN2_5_VL_MODELS = {
    "qwen2.5-vl-7b-instruct": "Qwen/Qwen2.5-VL-7B-Instruct"
}


def process_vision_info(messages):
    """
    Process image and video inputs from messages.
    Returns empty lists for text-only messages.
    """
    image_inputs = []
    video_inputs = []

    for message in messages:
        content = message.get("content", "")

        # Handle content that's a list (text + images)
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    # Handle image URLs
                    if item.get("type") == "image_url":
                        image_url = item.get("image_url", {}).get("url", "")
                        if image_url.startswith("data:image"):
                            # Extract base64 data
                            import base64
                            from io import BytesIO
                            from PIL import Image

                            try:
                                image_data = image_url.split(",")[1]
                                image_bytes = base64.b64decode(image_data)
                                image = Image.open(BytesIO(image_bytes))
                                image_inputs.append(image)
                            except Exception as e:
                                print(f"Error processing image: {e}")

                    # Handle video inputs if needed
                    elif item.get("type") == "video_url":
                        # Process video (not implemented in this example)
                        pass

    return image_inputs, video_inputs


class Qwen25VLAPI(ls.LitAPI):
    def setup(self, device, model_id=DEFAULT_MODEL):
        if model_id not in QWEN2_5_VL_MODELS.values():
            model_id = DEFAULT_MODEL

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            # quantization_config=quantization_config,  # Uncomment if needed
        ).eval()

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        self.device = device
        self.model_id = model_id

    def decode_request(self, request: ChatCompletionRequest, context: dict):
        # Set the model if different from current
        requested_model = request.model
        model_path = QWEN2_5_VL_MODELS.get(requested_model, DEFAULT_MODEL)

        if model_path != self.model_id:
            self.setup(self.device, model_path)

        # Set generation parameters
        context["generation_args"] = {
            "max_new_tokens": request.max_tokens if request.max_tokens else 2048,
            "temperature": request.temperature if request.temperature is not None else 0.7,
            "top_p": request.top_p if request.top_p is not None else 0.9,
        }

        # Process messages
        try:
            # Convert the Pydantic model to a dictionary
            messages = [
                message.model_dump(exclude_none=True) for message in request.messages
            ]

            # Pre-process messages to ensure correct format
            for message in messages:
                # Convert list content to proper format if needed
                if isinstance(message.get("content"), list):
                    # Ensure all text items are properly formatted
                    for i, item in enumerate(message["content"]):
                        if isinstance(item, str):
                            message["content"][i] = {"type": "text", "text": item}

            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process vision inputs
            image_inputs, video_inputs = process_vision_info(messages)

            # Prepare inputs for the model - handle both text-only and vision inputs
            if not image_inputs and not video_inputs:
                # Text-only request
                inputs = self.processor(
                    text=[text],
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
            else:
                # Vision + text request
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs if video_inputs else None,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)

            return inputs
        except Exception as e:
            # Log the error for debugging
            print(f"Error in decode_request: {e}")
            raise

    def predict(self, model_inputs, context: dict):
        # Set up generation parameters
        generation_kwargs = dict(
            model_inputs,
            streamer=self.streamer,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            **context["generation_args"],
        )

        # Start generation in a separate thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Stream the generated text
        for text in self.streamer:
            yield text


# Start the server
if __name__ == "__main__":
    api = Qwen25VLAPI()
    server = ls.LitServer(api, spec=ls.OpenAISpec(), accelerator="gpu")
    server.run(port=8000)