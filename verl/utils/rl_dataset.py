import math
import os
import logging
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.models.transformers.qwen2_5_vl import get_rope_index

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
    try:
        tensors = defaultdict(list)
        non_tensors = defaultdict(list)
        for feature in features:
            for key, value in feature.items():
                if isinstance(value, torch.Tensor):
                    tensors[key].append(value)
                else:
                    non_tensors[key].append(value)

        for key, value in tensors.items():
            if key not in ["pixel_values", "image_grid_thw"]:
                try:
                    tensors[key] = torch.stack(value, dim=0)
                except Exception as e:
                    logger.error(f"Error stacking tensors for key {key}: {str(e)}")
                    logger.error(traceback.format_exc())
                    tensors[key] = value

        return {**tensors, **non_tensors}
    except Exception as e:
        logger.error(f"Error in collate_fn: {str(e)}")
        logger.error(traceback.format_exc())
        raise


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


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
            self,
            data_path: str,
            tokenizer: PreTrainedTokenizer,
            processor: Optional[ProcessorMixin],
            prompt_key="prompt",
            max_prompt_length=1024,
            truncation="error",
            system_prompt=None,
            max_pixels=None,
            min_pixels=None,
            video_frames=4,
            worker_id=None
    ):
        try:
            self.tokenizer = tokenizer
            self.processor = processor
            self.prompt_key = prompt_key
            self.max_prompt_length = max_prompt_length
            self.truncation = truncation
            self.system_prompt = system_prompt
            self.max_pixels = max_pixels
            self.min_pixels = min_pixels
            self.video_frames = video_frames
            self.worker_id = worker_id or os.getpid()

            logger.info(f"Worker {self.worker_id}: Initializing RLHFDataset with data_path={data_path}")

            if "@" in data_path:
                data_path, data_split = data_path.split("@")
            else:
                data_split = "train"

            logger.info(f"Worker {self.worker_id}: Loading dataset from {data_path}, split={data_split}")

            if '.json' in data_path:
                self.dataset = load_dataset('json', data_files=data_path, split='train')
            else:
                self.dataset = load_dataset(data_path, split=data_split)

            logger.info(f"Worker {self.worker_id}: Dataset loaded with {len(self.dataset)} samples")

            self.data_dir = os.path.dirname(data_path)
            logger.info(f"Worker {self.worker_id}: Data directory: {self.data_dir}")

        except Exception as e:
            logger.error(f"Worker {worker_id or os.getpid()}: Error initializing dataset: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        try:
            logger.debug(f"Worker {self.worker_id}: Processing item {index}")
            row_dict = self.dataset[index]

            # Validate prompt exists
            if self.prompt_key not in row_dict:
                logger.warning(
                    f"Worker {self.worker_id}: Item {index} missing prompt key '{self.prompt_key}'. Keys: {list(row_dict.keys())}")
                # Create a fallback empty prompt
                row_dict[self.prompt_key] = ""

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": row_dict[self.prompt_key]},
            ]

            logger.debug(f"Worker {self.worker_id}: Applying chat template for item {index}")
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            # Process images if they exist
            processed_images = []
            if "images" in row_dict and row_dict["images"]:
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
                            # Use the image directly if it's already a PIL Image
                            image = image_item

                        # Process the image
                        processed_images.append(process_image(image, self.max_pixels, self.min_pixels))

                    except Exception as e:
                        logger.error(f"Worker {self.worker_id}: Error processing image {i} for item {index}: {str(e)}")
                        logger.error(traceback.format_exc())
                        placeholder_image = Image.new("RGB", (224, 224), (255, 255, 255))
                        processed_images.append(process_image(placeholder_image, self.max_pixels, self.min_pixels))

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
                            # Add placeholder frames
                            for _ in range(self.video_frames):
                                processed_images.append(Image.new("RGB", (224, 224), (255, 255, 255)))

                    except Exception as e:
                        logger.error(f"Worker {self.worker_id}: Error processing video {i} for item {index}: {str(e)}")
                        logger.error(traceback.format_exc())
                        # Add placeholder frames
                        for _ in range(self.video_frames):
                            processed_images.append(Image.new("RGB", (224, 224), (255, 255, 255)))

            if "images" in row_dict and len(row_dict["images"]) > 0:
                data_source = row_dict["images"][0].split("/")[0]
            elif "videos" in row_dict and len(row_dict["videos"]) > 0:
                data_source = row_dict["videos"][0].split("/")[0]
            else:
                data_source = "text"
            row_dict["data_source"] = data_source

            # Ensure we have at least one image/frame
            if not processed_images:
                logger.debug(f"Worker {self.worker_id}: No images or videos found for item {index}, using placeholder")
                processed_images = [Image.new("RGB", (224, 224), (255, 255, 255))]

            row_dict["images"] = processed_images

            # Replace all image tokens in prompt with placeholders
            prompt = prompt.replace("<video>", "<image>")
            if "<image>" not in prompt:
                prompt = "<image> " + prompt
            raw_prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")

            logger.debug(f"Worker {self.worker_id}: Running image processor for item {index}")
            image_inputs = self.processor.image_processor(row_dict["images"], return_tensors="pt")
            image_grid_thw = image_inputs["image_grid_thw"]
            row_dict.update(image_inputs)

            if image_grid_thw is not None:
                logger.debug(f"Worker {self.worker_id}: Processing image grid for item {index}")
                merge_length = self.processor.image_processor.merge_size ** 2
                index = 0
                while "<image>" in prompt:
                    prompt = prompt.replace(
                        "<image>",
                        "<|vision_start|>"
                        + "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length)
                        + "<|vision_end|>",
                        1,
                    )
                    index += 1

                prompt = prompt.replace("<|placeholder|>", self.processor.image_token)

            logger.debug(f"Worker {self.worker_id}: Tokenizing prompt for item {index}")
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt,
                tokenizer=self.tokenizer,
                max_length=self.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation,
            )

            if image_grid_thw is not None:
                logger.debug(f"Worker {self.worker_id}: Getting rope indices for item {index}")
                position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    attention_mask=attention_mask,
                )  # (3, seq_len)
            else:
                position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seqlen,)

            row_dict["input_ids"] = input_ids
            row_dict["attention_mask"] = attention_mask
            row_dict["position_ids"] = position_ids

            logger.debug(f"Worker {self.worker_id}: Encoding raw prompt for item {index}")
            row_dict["raw_prompt_ids"] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

            logger.debug(f"Worker {self.worker_id}: Successfully processed item {index}")
            return row_dict

        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Error in __getitem__ for index {index}: {str(e)}")
            logger.error(traceback.format_exc())

            # Instead of crashing, return a minimal valid item
            fallback_dict = {
                "input_ids": torch.ones(1, dtype=torch.long),
                "attention_mask": torch.ones(1, dtype=torch.long),
                "position_ids": torch.zeros(1, dtype=torch.long),
                "raw_prompt_ids": torch.ones(1, dtype=torch.long),
                "data_source": "fallback",
            }
            return fallback_dict