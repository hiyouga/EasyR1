import math
import os
import logging
import traceback
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF

# For video processing
import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('dataset_worker.log'), logging.StreamHandler()]
)
logger = logging.getLogger('RLHFDataset')


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def extract_video_frames(video_path: str, num_frames: int = 4) -> List[ImageObject]:
    """
    Extract a specified number of frames uniformly from a video.

    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract

    Returns:
        List of PIL Image objects
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return [Image.new("RGB", (224, 224), (128, 128, 128)) for _ in range(num_frames)]

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            logger.error(f"Invalid frame count for video: {video_path}")
            return [Image.new("RGB", (224, 224), (128, 128, 128)) for _ in range(num_frames)]

        # Calculate frame indices to extract
        frames_to_extract = []
        if num_frames == 1:
            # Just get the middle frame
            frames_to_extract = [total_frames // 2]
        else:
            # Get frames uniformly distributed across the video
            for i in range(num_frames):
                frame_idx = int(i * (total_frames - 1) / (num_frames - 1)) if num_frames > 1 else 0
                frames_to_extract.append(frame_idx)

        # Extract the frames
        frames = []
        for frame_idx in frames_to_extract:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                logger.warning(f"Failed to read frame {frame_idx} from video: {video_path}")
                frames.append(Image.new("RGB", (224, 224), (128, 128, 128)))
                continue

            # Convert from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)

        cap.release()
        return frames

    except Exception as e:
        logger.error(f"Error extracting frames from video {video_path}: {str(e)}")
        logger.error(traceback.format_exc())
        # Return placeholder images
        return [Image.new("RGB", (224, 224), (128, 128, 128)) for _ in range(num_frames)]


def process_image(image: ImageObject, max_pixels: int, min_pixels: int) -> ImageObject:
    try:
        if max_pixels is not None and (image.width * image.height) > max_pixels:
            resize_factor = math.sqrt(max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            logger.debug(
                f"Resizing image from {image.width}x{image.height} to {width}x{height} (max_pixels: {max_pixels})")
            image = image.resize((width, height), resample=Image.Resampling.NEAREST)

        if min_pixels is not None and (image.width * image.height) < min_pixels:
            resize_factor = math.sqrt(min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            logger.debug(
                f"Resizing image from {image.width}x{image.height} to {width}x{height} (min_pixels: {min_pixels})")
            image = image.resize((width, height), resample=Image.Resampling.NEAREST)

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image
    except Exception as e:
        logger.error(f"Error in process_image: {str(e)}")
        # Return a small fallback image instead of crashing
        fallback = Image.new("RGB", (224, 224), (128, 128, 128))
        return fallback


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        max_prompt_length: int = 1024,
        truncation: str = "error",
        system_prompt: str = None,
        max_pixels: int = None,
        min_pixels: int = None,
        video_frames=4
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.system_prompt = system_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.video_frames = video_frames
        self.worker_id = 0

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            self.dataset = load_dataset("json", data_dir=data_path, split="train")
        elif os.path.isfile(data_path):
            self.dataset = load_dataset("json", data_files=data_path, split="train")
        else:  # remote dataset
            self.dataset = load_dataset(data_path, split=data_split)
        self.data_dir = os.path.dirname(data_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row_dict: dict = self.dataset[index]
        messages = [{"role": "user", "content": row_dict[self.prompt_key]}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        processed_images = []
        if self.image_key in row_dict and row_dict["images"]:
            for i, image_item in enumerate(row_dict["images"]):
                try:
                    logger.debug(f"Worker {self.worker_id}: Processing image {i} for item {index}")

                    if isinstance(image_item, str):
                        # Load the image if it's a path
                        full_path = os.path.join(self.data_dir, image_item)
                        logger.debug(f"Worker {self.worker_id}: Loading image {i} from {full_path}")

                        if not os.path.exists(full_path):
                            logger.warning(f"Worker {self.worker_id}: Image file not found: {full_path}")
                            image = Image.new("RGB", (224, 224), (255, 255, 255))
                        else:
                            image = Image.open(full_path)
                    else:
                        image = image_item

                    # Process the image
                    processed_images.append(process_image(image, self.max_pixels, self.min_pixels))

                except Exception as e:
                    logger.error(f"Worker {self.worker_id}: Error processing image {i} for item {index}: {str(e)}")
                    logger.error(traceback.format_exc())

        # Process videos if they exist
        if "videos" in row_dict and row_dict["videos"]:
            logger.debug(f"Worker {self.worker_id}: Processing videos for item {index}")
            for i, video_item in enumerate(row_dict["videos"]):
                try:
                    if isinstance(video_item, str):
                        # Load the video if it's a path
                        full_path = os.path.join(self.data_dir, video_item)
                        logger.debug(f"Worker {self.worker_id}: Loading video {i} from {full_path}")

                        if not os.path.exists(full_path):
                            logger.warning(f"Worker {self.worker_id}: Video file not found: {full_path}")
                            # Add placeholder frames
                            for _ in range(self.video_frames):
                                processed_images.append(Image.new("RGB", (224, 224), (255, 255, 255)))
                        else:
                            # Extract frames from video
                            video_frames = extract_video_frames(full_path, self.video_frames)
                            for frame in video_frames:
                                processed_images.append(process_image(frame, self.max_pixels, self.min_pixels))
                    else:
                        logger.warning(
                            f"Worker {self.worker_id}: Video item type not supported: {type(video_item)}")

                except Exception as e:
                    logger.error(f"Worker {self.worker_id}: Error processing video {i} for item {index}: {str(e)}")
                    logger.error(traceback.format_exc())

        if "images" in row_dict and len(row_dict["images"]) > 0:
            data_source = row_dict["images"][0].split("/")[0]
            dataset = row_dict["images"][0].split("/")[1]
        elif "videos" in row_dict and len(row_dict["videos"]) > 0:
            data_source = row_dict["videos"][0].split("/")[0]
            dataset = row_dict["videos"][0].split("/")[1]
        else:
            data_source = "text"
            dataset = "text"
        row_dict["data_source"] = data_source
        row_dict["dataset"] = dataset

        # Ensure we have at least one image/frame
        if not processed_images:
            logger.debug(f"Worker {self.worker_id}: No images or videos found for item {index}, using placeholder")
            processed_images = [Image.new("RGB", (224, 224), (255, 255, 255))]

        row_dict["images"] = processed_images
        row_dict["multi_modal_data"] = {
            "image": processed_images
        }


        # Replace all image tokens in prompt with placeholders
        prompt = prompt.replace("<video>", "<image>")
        if "<image>" not in prompt:
            prompt = "<image> " + prompt
        image_count_in_prompt = prompt.count("<image>")
        image_count = len(processed_images)
        if len(processed_images) > 1 and image_count_in_prompt < len(processed_images):
            # add more image tokens to prompt
            missing_count = len(processed_images) - image_count_in_prompt
            prompt = prompt.replace("<image>", "<image> " * (missing_count + 1), 1)
        prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
        image_count_in_prompt = prompt.count("<|vision_start|>")
        assert image_count == image_count_in_prompt, f"Image count mismatch: {image_count} != {image_count_in_prompt}"
        model_inputs = self.processor(row_dict["multi_modal_data"]["image"], prompt, return_tensors="pt")
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
        row_dict["multi_modal_data"] = dict(model_inputs)
        position_ids = get_rope_index(
            self.processor,
            input_ids=input_ids,
            image_grid_thw=model_inputs["image_grid_thw"],
            attention_mask=attention_mask,
        )  # (3, seq_length)
        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["ground_truth"] = row_dict[self.answer_key]
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(prompt, add_special_tokens=False)
        return row_dict