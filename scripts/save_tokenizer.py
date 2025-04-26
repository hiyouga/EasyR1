import logging
import os


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def save_qwen25_processor(output_path):
    """
    Save the tokenizer and image processor of Qwen2.5 in a way compatible with huggingface transformers.

    Args:
        output_path: Directory to save the tokenizer and processor to
    """
    from transformers import AutoTokenizer, AutoProcessor

    logger.info("Saving Qwen2.5 processor components...")

    # Load and save the fast tokenizer
    try:
        # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)
        # tokenizer.save_pretrained(output_path)
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)
        processor.save_pretrained(output_path)
        logger.info(f"Saved fast tokenizer to {output_path}")
    except Exception as e:
        logger.warning(f"Error saving fast tokenizer: {e}")
        # Fall back to slow tokenizer if fast version fails
        try:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=False)
            tokenizer.save_pretrained(output_path)
            logger.info(f"Saved slow tokenizer to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save tokenizer: {e}")

    logger.info(f"Saved Qwen2.5 processor components to {output_path}")

    return output_path


if __name__ == "__main__":
    save_qwen25_processor('/scratch/outputs/qwen/qwen25_vision_model')