import streamlit as st
from openai import OpenAI
import base64
from io import BytesIO
from PIL import Image
import json
import random
import os
import numpy as np
import io
import cv2

# Configure the page
st.set_page_config(
    page_title="Medical Image Analysis Demo",
    page_icon="🔬",
    layout="wide"
)

# Initialize the OpenAI client to connect to your local API
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"  # The API key doesn't matter as it's just a placeholder
)

# Set up the Streamlit app
st.title("Medical Image Analysis Demo")
st.markdown("Analyze medical images using AI")


# ---- Image Select Component from paste.txt ----
@st.cache_data
def _encode_file(img):
    if isinstance(img, str) and img[-4:] in ['.mp4', '.mov', '.avi']:
        # Sample 1st frame from video
        cap = cv2.VideoCapture(img)
        ret, frame = cap.read()
        if not ret:
            return ''
        # Convert to RGB
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', frame)
        encoded = base64.b64encode(buffer).decode()
        return f"data:image/jpeg;base64, {encoded}"
    else:
        with open(img, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
        return f"data:image/jpeg;base64, {encoded}"


@st.cache_data
def _encode_numpy(img):
    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64, {encoded}"


def image_select(
        label: str,
        images: list,
        captions: list = None,
        index: int = 0,
        *,
        use_container_width: bool = True,
        return_value: str = "original",
        key: str = None,
):
    """Shows several images and returns the image selected by the user."""
    # Do some checks to verify the input.
    if len(images) < 1:
        raise ValueError("At least one image must be passed but `images` is empty.")
    if captions is not None and len(images) != len(captions):
        raise ValueError(
            "The number of images and captions must be equal but `captions` has "
            f"{len(captions)} elements and `images` has {len(images)} elements."
        )
    if index >= len(images):
        raise ValueError(
            f"`index` must be smaller than the number of images ({len(images)}) "
            f"but it is {index}."
        )

    # Encode local images/numpy arrays/PIL images to base64.
    encoded_images = []
    for img in images:
        if isinstance(img, (np.ndarray, Image.Image)):  # numpy array or PIL image
            encoded_images.append(_encode_numpy(np.asarray(img)))
        elif os.path.exists(img):  # local file
            encoded_images.append(_encode_file(img))
        else:  # url, use directly
            encoded_images.append(img)

    # Import the component function here to avoid importing errors
    # when the script is run outside Streamlit
    from streamlit_image_select import _component_func

    # Pass everything to the frontend.
    component_value = _component_func(
        label=label,
        images=encoded_images,
        captions=captions,
        index=index,
        use_container_width=use_container_width,
        key=key,
        default=index,
    )

    # The frontend component returns the index of the selected image but we want to
    # return the actual image.
    if return_value == "original":
        return images[component_value]
    elif return_value == "index":
        return component_value
    else:
        raise ValueError(
            "`return_value` must be either 'original' or 'index' "
            f"but is '{return_value}'."
        )


# ---- End of Image Select Component ----

# Function to load JSONL data
def load_jsonl_samples(file_path, num_samples=12):
    """Load a random sample of entries from a JSONL file"""
    samples = []
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            st.error(f"JSONL file not found: {file_path}")
            # Return empty list but don't crash
            return []

        with open(file_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            st.error(f"JSONL file is empty: {file_path}")
            return []

        # Parse all lines
        all_data = []
        for i, line in enumerate(lines):
            try:
                all_data.append(json.loads(line))
            except json.JSONDecodeError as e:
                st.error(f"Error parsing JSON at line {i + 1}: {e}")

        st.success(f"Successfully loaded {len(all_data)} entries from JSONL file")

        # Select random samples
        if len(all_data) > num_samples:
            samples = random.sample(all_data, num_samples)
        else:
            samples = all_data

    except Exception as e:
        st.error(f"Error loading JSONL data: {e}")
        st.error(f"Exception details: {type(e).__name__}: {str(e)}")

    return samples


# Function to encode image to base64 for the chat
def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


# Function to clear the chat history
def clear_chat():
    st.session_state.messages = []
    st.session_state.image_select_key += 1
    st.rerun()


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "image_select_key" not in st.session_state:
    st.session_state.image_select_key = 0

# Sidebar for settings
with st.sidebar:
    st.header("Settings")

    model_name = st.selectbox(
        "Select model",
        ["qwen2-vl-7b-instruct"],  # You can add more models later
        index=0
    )

    # Clear chat button
    if st.button("Clear Chat"):
        clear_chat()
        st.success("Chat history cleared!")

# Display image grid or chat
if len(st.session_state.messages) == 0:
    st.header("Select a medical image to analyze")

    # For debugging purposes, let's add a file path check
    jsonl_path = "/mnt/8T/high_modality/geom_valid.jsonl"
    st.write(f"Attempting to load JSONL from: {jsonl_path}")
    st.write(f"File exists: {os.path.exists(jsonl_path)}")

    # Fallback to a sample if needed
    sample_data = [
        {"problem": "<image>\nWhat does this show?", "answer": "Test", "images": ["sample.jpg"]},
        {"problem": "<image>\nDescribe this image.", "answer": "Test", "images": ["sample.jpg"]},
    ]

    # Create a test image if needed
    test_image_path = "sample.jpg"
    if not os.path.exists(test_image_path):
        test_img = Image.new('RGB', (300, 300), color='blue')
        test_img.save(test_image_path)

    try:
        # Load samples from JSONL
        samples = load_jsonl_samples(jsonl_path)

        # If no samples were loaded, use our test samples
        if not samples:
            st.warning("Using test samples since no data was loaded from JSONL")
            samples = sample_data

        st.write(f"Number of samples loaded: {len(samples)}")

        # Extract images and problems
        images = []
        problems = []
        image_paths = []

        for sample in samples:
            # Store the original problem text (stripping the image tag)
            problem_text = sample["problem"].replace("<image>\n", "")
            problems.append(problem_text)

            # Get the first image path from each sample
            try:
                img_path = os.path.join("/mnt/8T/high_modality", sample["images"][0])
                # Check if the image file exists
                if not os.path.exists(img_path):
                    st.warning(f"Image not found: {img_path}. Using test image.")
                    img_path = test_image_path
            except (KeyError, IndexError) as e:
                st.warning(f"Error accessing image path in sample: {e}. Using test image.")
                img_path = test_image_path

            image_paths.append(img_path)

        # Load the images (we need actual images for the image_select component)
        pil_images = []
        for i, path in enumerate(image_paths):
            try:
                img = Image.open(path)
                pil_images.append(img)
            except Exception as e:
                st.error(f"Error loading image {path}: {e}")
                # Add a placeholder image if loading fails
                placeholder = Image.new('RGB', (300, 300), color=(100, 100, 100))
                pil_images.append(placeholder)

        # Add debug information
        st.write(f"Loaded {len(pil_images)} images")

        # Safety check - make sure we have images before calling image_select
        if len(pil_images) == 0:
            st.error("No images were loaded. Please check the file path.")
        else:
            # Use the image_select component to display the grid
            # We don't show captions because they would reveal the problem text
            if "previous_selected_idx" not in st.session_state:
                st.session_state.previous_selected_idx = -1

            selected_idx = image_select(
                label="Select an image to analyze",
                images=pil_images,
                captions=None,  # No captions to avoid showing the problem text
                index=0,  # This must be a valid index for the component to work
                use_container_width=True,
                return_value="index",
                key=f"image_select_{st.session_state.image_select_key}"
            )

            # Only process selection if it's different from the previous state
            # This avoids auto-selecting the first image on page load
            if selected_idx != st.session_state.previous_selected_idx:
                st.session_state.previous_selected_idx = selected_idx

                # Check if this is a valid selection and not the initial load
                if "initial_load" not in st.session_state:
                    st.session_state.initial_load = True
                    st.rerun()  # Force a rerun to avoid auto-selection on first load
                elif selected_idx >= 0 and selected_idx < len(pil_images):
                    # Get selected image and problem
                    selected_image = pil_images[selected_idx]
                    selected_problem = problems[selected_idx]

                    # Encode image to base64
                    img_base64 = encode_image(selected_image)

                    # Add the selected image to the chat
                    st.session_state.current_image = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}"
                        }
                    }

                    # Create user message with image and problem
                    user_message = {
                        "role": "user",
                        "content": [
                            st.session_state.current_image,
                            {"type": "text", "text": selected_problem}
                        ]
                    }

                    # Add to messages
                    st.session_state.messages.append(user_message)

                    # Reset the image state and selection key
                    st.session_state.current_image = None
                    st.session_state.image_select_key += 1

                    # Rerun to update UI
                    st.rerun()

    except Exception as e:
        st.error(f"Error setting up image selection: {e}")
else:
    # Display chat messages
    st.header("Analysis")

    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], str):
                st.markdown(message["content"])
            elif isinstance(message["content"], list):
                for item in message["content"]:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            st.markdown(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            image_url = item["image_url"]["url"]
                            if image_url.startswith("data:image"):
                                image_data = image_url.split(",")[1]
                                image_bytes = base64.b64decode(image_data)
                                image = Image.open(BytesIO(image_bytes))
                                st.image(image, use_column_width=True)
                    elif isinstance(item, str):
                        st.markdown(item)
            else:
                st.markdown(message["content"])

    # User input
    user_input = st.chat_input("Type your follow-up question here...")

    if user_input:
        # Create message with text only (no image since we've already processed it)
        user_message = {
            "role": "user",
            "content": user_input
        }

        # Add user message to chat history
        st.session_state.messages.append(user_message)

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_message["content"])

# Generate assistant response if last message is from user
if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # Prepare system message
            system_message = {
                "role": "system",
                "content": "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."
            }

            # Call the API with streaming
            response = client.chat.completions.create(
                model=model_name,
                messages=[system_message] + st.session_state.messages,
                stream=True,
                max_tokens=1024
            )

            # Process the streaming response
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content_chunk = chunk.choices[0].delta.content
                    full_response += content_chunk
                    message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })

        except Exception as e:
            error_message = f"Error: {str(e)}"
            st.error(error_message)
            message_placeholder.markdown(error_message)