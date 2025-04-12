import copy
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

    # Handle segmentation masks separately
    seg_masks = []
    max_height, max_width = 0, 0

    # First pass: collect all tensors and find max dimensions for segmentation masks
    for feature in features:
        assert 'segmentation_mask' in feature
        for key, value in feature.items():
            if key == "segmentation_mask":
                assert isinstance(value, torch.Tensor)
                if len(value.shape) == 3:  # [C, H, W]
                    _, h, w = value.shape
                    max_height = max(max_height, h)
                    max_width = max(max_width, w)
                seg_masks.append(value)
            elif isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    # Second pass: pad segmentation masks to max dimensions
    padded_masks = []
    for i, mask in enumerate(seg_masks):
        if mask is None:
            # Create zero tensor for missing segmentation masks
            padded_masks.append(torch.zeros(1, max_height, max_width, dtype=torch.float32))
        else:
            # Get current dimensions
            if len(mask.shape) == 3:  # [C, H, W]
                c, h, w = mask.shape
                # Calculate padding (bottom, right)
                pad_bottom = max_height - h
                pad_right = max_width - w
                # Pad the mask (pad order is [left, right, top, bottom])
                padded_mask = torch.nn.functional.pad(mask, (0, pad_right, 0, pad_bottom), mode='constant', value=0)
                padded_masks.append(padded_mask)
            else:
                # Handle unexpected shapes
                padded_masks.append(torch.zeros(1, max_height, max_width, dtype=torch.float32))

        # Stack the padded masks
        tensors["segmentation_mask"] = torch.stack(padded_masks, dim=0)

    # Stack other tensors
    for key, value in tensors.items():
        if key != "segmentation_mask":  # We've already handled segmentation masks
            tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    # Combine tensors and non-tensors
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
    finally:
        if cap is not None and cap.isOpened():
            cap.release()
            logger.debug(f"Released video capture for {video_path}")


class ImageProcessMixin:
    max_pixels: int
    min_pixels: int

    def process_image(self, image: Union[Dict[str, Any], ImageObject]) -> ImageObject:
        try:
            if self.max_pixels is not None and (image.width * image.height) > self.max_pixels:
                resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
                width, height = int(image.width * resize_factor), int(image.height * resize_factor)
                logger.debug(
                    f"Resizing image from {image.width}x{image.height} to {width}x{height} (max_pixels: {self.max_pixels})")
                image = image.resize((width, height), resample=Image.Resampling.NEAREST)

            if self.min_pixels is not None and (image.width * image.height) < self.min_pixels:
                resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
                width, height = int(image.width * resize_factor), int(image.height * resize_factor)
                logger.debug(
                    f"Resizing image from {image.width}x{image.height} to {width}x{height} (min_pixels: {self.min_pixels})")
                image = image.resize((width, height), resample=Image.Resampling.NEAREST)

            if image.mode != "RGB":
                image = image.convert("RGB")

            return image
        except Exception as e:
            logger.error(f"Error in process_image: {str(e)}")
            # Return a small fallback image instead of crashing
            fallback = Image.new("RGB", (224, 224), (128, 128, 128))
            return fallback


def resize_bbox(bbox, original_width, original_height, new_width, new_height):
    """
    Resize bounding box coordinates based on image resizing ratio.

    Args:
        bbox (list): Original bounding box in format [x_min, y_min, x_max, y_max]
        original_width (int): Width of the original image
        original_height (int): Height of the original image
        new_width (int): Width of the resized image
        new_height (int): Height of the resized image

    Returns:
        list: Resized bounding box coordinates
    """
    # Calculate scaling factors
    width_ratio = new_width / original_width
    height_ratio = new_height / original_height

    # Apply scaling to bounding box coordinates
    x_min, y_min, x_max, y_max = bbox

    # Scale coordinates
    new_x_min = x_min * width_ratio
    new_y_min = y_min * height_ratio
    new_x_max = x_max * width_ratio
    new_y_max = y_max * height_ratio

    return [new_x_min, new_y_min, new_x_max, new_y_max]


class RLHFDataset(Dataset, ImageProcessMixin):
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
            format_prompt: str = None,
            max_pixels: int = None,
            min_pixels: int = None,
            video_frames=4
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        print(f"Dataset processor has class {self.processor.__class__.__name__}, "
              f"image processor has class {self.processor.image_processor.__class__.__name__}")
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.format_prompt = format_prompt
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
        row_dict: dict = copy.deepcopy(self.dataset[index])
        prompt_str: str = row_dict[self.prompt_key]
        if self.format_prompt:
            prompt_str = prompt_str + " " + self.format_prompt.strip()

        processed_images = []
        original_dimensions = []  # Store original image dimensions

        # Extract data_source and dataset
        vision_path = row_dict['images']
        if len(vision_path) == 0:
            vision_path = row_dict['videos']
        if len(vision_path) == 0:
            row_dict["data_source"] = "unknown"
            row_dict["dataset"] = "unknown"
        vision_path = vision_path[0]
        row_dict["data_source"] = vision_path.split("/")[0]
        row_dict["dataset"] = vision_path.split("/")[1]

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

                    original_dimensions.append((image.width, image.height))
                    # Process the image
                    processed_images.append(self.process_image(image))

                except Exception as e:
                    logger.error(f"Worker {self.worker_id}: Error processing image {i} for item {index}: {str(e)}")
                    logger.error(traceback.format_exc())
                    original_dimensions.append((224, 224))

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
                                original_dimensions.append((224, 224))  # Add placeholder dimensions
                        else:
                            # Extract frames from video
                            video_frames = extract_video_frames(full_path, self.video_frames)
                            for frame in video_frames:
                                # Store original dimensions
                                original_dimensions.append((frame.width, frame.height))
                                processed_images.append(self.process_image(frame))
                    else:
                        logger.warning(
                            f"Worker {self.worker_id}: Video item type not supported: {type(video_item)}")

                except Exception as e:
                    logger.error(f"Worker {self.worker_id}: Error processing video {i} for item {index}: {str(e)}")
                    logger.error(traceback.format_exc())

        # get size from processed_images
        if len(processed_images) > 0:
            image_size = processed_images[0].size
            logger.debug(f"Worker {self.worker_id}: Processed images size: {image_size}")
        else:
            image_size = (224, 224)

        # Load segmentation mask if available
        if "segmentation_path" in row_dict and row_dict["segmentation_path"]:
            try:
                seg_path = os.path.join(self.data_dir, row_dict["segmentation_path"])
                if os.path.exists(seg_path):
                    logger.debug(f"Worker {self.worker_id}: Loading segmentation mask from {seg_path}")
                    segmentation_mask = Image.open(seg_path)
                    # Process the segmentation mask if needed
                    row_dict["segmentation_mask"] = segmentation_mask
                else:
                    logger.warning(f"Worker {self.worker_id}: Segmentation mask not found: {seg_path}")
                    row_dict["segmentation_mask"] = None
            except Exception as e:
                logger.error(f"Worker {self.worker_id}: Error loading segmentation mask: {str(e)}")
                logger.error(traceback.format_exc())
                row_dict["segmentation_mask"] = None
        else:
            row_dict["segmentation_mask"] = None

        # Ensure we have at least one image/frame
        if not processed_images:
            logger.debug(f"Worker {self.worker_id}: No images or videos found for item {index}, using placeholder")
            processed_images = [Image.new("RGB", (224, 224), (255, 255, 255))]
            original_dimensions = [(224, 224)]  # Add placeholder dimensions

        row_dict["images"] = processed_images
        row_dict["multi_modal_data"] = {
            "image": processed_images
        }

        # Replace all image tokens in prompt with placeholders
        prompt_str = prompt_str.replace("<video>", "<image>")
        if "<image>" not in prompt_str:
            prompt_str = "<image> " + prompt_str
        image_count_in_prompt = prompt_str.count("<image>")
        image_count = len(processed_images)
        if len(processed_images) > 1 and image_count_in_prompt < len(processed_images):
            # add more image tokens to prompt
            missing_count = len(processed_images) - image_count_in_prompt
            prompt_str = prompt_str.replace("<image>", "<image> " * (missing_count + 1), 1)
        image_count_in_prompt = prompt_str.count("<image>")
        assert image_count == image_count_in_prompt, f"Image count mismatch: {image_count} != {image_count_in_prompt}"
        content_list = []
        for i, content in enumerate(prompt_str.split("<image>")):
            if i != 0:
                content_list.append({"type": "image"})

            if content:
                content_list.append({"type": "text", "text": content})
        messages = [{"role": "user", "content": content_list}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        try:
            model_inputs = self.processor(row_dict["multi_modal_data"]["image"], [prompt], return_tensors="pt")
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Error processing model inputs: {str(e)}")
            # remove image
            row_dict["images"] = [Image.new("RGB", (224, 224), (255, 255, 255)) for _ in range(image_count)]
            row_dict["multi_modal_data"]["image"] = row_dict["images"]
            model_inputs = self.processor(row_dict["multi_modal_data"]["image"], prompt, return_tensors="pt")
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]

        # Resize segmentation mask to match the image dimensions in model_inputs
        if row_dict["segmentation_mask"] is not None:
            try:
                # Extract dimensions from image_grid_thw (time, height, width)
                # We need the height and width for resizing
                target_height, target_width = image_size

                logger.debug(f"Worker {self.worker_id}: Resizing segmentation mask to {target_width}x{target_height}")

                # Resize the segmentation mask to match the processed image dimensions
                resized_mask = row_dict["segmentation_mask"].resize(
                    (target_width, target_height),
                    resample=Image.Resampling.NEAREST
                )

                # Convert the mask to a tensor
                mask_array = np.array(resized_mask)

                # If mask is grayscale, add channel dimension
                if len(mask_array.shape) == 2:
                    mask_array = mask_array[np.newaxis, :, :]
                # If mask is RGB but we only need one channel for segmentation
                elif len(mask_array.shape) == 3 and mask_array.shape[2] == 3:
                    mask_array = np.mean(mask_array, axis=2)[np.newaxis, :, :]

                # Convert to torch tensor
                mask_tensor = torch.from_numpy(mask_array).float()
                row_dict["segmentation_mask"] = mask_tensor

                logger.debug(f"Worker {self.worker_id}: Successfully resized segmentation mask")
            except Exception as e:
                logger.error(f"Worker {self.worker_id}: Error resizing segmentation mask: {str(e)}")
                logger.error(traceback.format_exc())

        if "segmentation_mask" not in row_dict or row_dict["segmentation_mask"] is None:
            target_width, target_height = image_size
            row_dict["segmentation_mask"] = torch.zeros(1, target_height, target_width, dtype=torch.float32)

        # Handle bounding box information
        if "bbox" in row_dict and row_dict["bbox"]:
            try:
                target_width, target_height = image_size

                # Get original dimensions of the corresponding image
                # We assume the bbox corresponds to the first image
                if original_dimensions:
                    original_width, original_height = original_dimensions[0]
                    # Resize the bounding box
                    resized_bbox = resize_bbox(
                        row_dict["bbox"],
                        original_width,
                        original_height,
                        target_width,
                        target_height
                    )

                    logger.debug(f"Worker {self.worker_id}: Resized bbox from {row_dict['bbox']} to {resized_bbox}. "
                                f"Original dimensions: {original_dimensions[0]}, "
                                f"Target dimensions: {target_width}x{target_height}")
                    row_dict["bbox"] = resized_bbox
                else:
                    logger.warning(f"Worker {self.worker_id}: No original dimensions available for bbox resizing")
            except Exception as e:
                logger.error(f"Worker {self.worker_id}: Error resizing bounding box: {str(e)}")
                logger.error(traceback.format_exc())
        else:
            # Use empty list as placeholder if not available
            row_dict["bbox"] = [0, 0, 0, 0]

        # Make bbox tensor
        row_dict["bbox"] = torch.tensor(row_dict["bbox"], dtype=torch.float32)

        row_dict["multi_modal_inputs"] = dict(model_inputs)
        if self.processor is not None:
            # and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor"
            # qwen2vl mrope
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw"),
                attention_mask=attention_mask,
            )  # (3, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)
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
        row_dict.pop("segmentation_path", None)
        return row_dict