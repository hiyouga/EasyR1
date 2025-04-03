import streamlit as st
from openai import OpenAI
import base64
from io import BytesIO
from PIL import Image
import json
import random
import os
import numpy as np

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


# Note: This function is kept for reference but no longer used directly
# We now use the cached samples approach for consistency
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

        # Select random samples
        if len(all_data) > num_samples:
            samples = random.sample(all_data, num_samples)
        else:
            samples = all_data

    except Exception as e:
        st.error(f"Error loading JSONL data: {e}")

    return samples


# Function to encode image to base64 for the chat
def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


# Function to clear the chat history
def clear_chat():
    st.session_state.messages = []
    st.session_state.current_image = None
    st.session_state.processed_image = False
    st.session_state.uploader_key += 1
    # Do not clear loaded_samples or sample_indices to keep image consistency
    st.rerun()


# Function to handle clear image button
def clear_image():
    st.session_state.current_image = None
    st.session_state.processed_image = False
    st.session_state.uploader_key += 1


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "processed_image" not in st.session_state:
    st.session_state.processed_image = False
if "model_name" not in st.session_state:
    st.session_state.model_name = "qwen2-vl-7b-instruct"
if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = None
# Store fixed sample indices for consistency across refreshes
if "sample_indices" not in st.session_state:
    st.session_state.sample_indices = list(range(1000))  # Large enough number for any reasonable sample
    random.shuffle(st.session_state.sample_indices)
# Store loaded samples to avoid re-randomization
if "loaded_samples" not in st.session_state:
    st.session_state.loaded_samples = None

# Create a two-column layout
left_col, right_col = st.columns([3, 2])

# Main chat area
with left_col:
    # Display chat header
    st.header("Chat")

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
                                st.image(image, use_container_width=True)
                    elif isinstance(item, str):
                        st.markdown(item)
            else:
                st.markdown(message["content"])

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
                    model=st.session_state.model_name,
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

                # Final response
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

                # For debugging
                import traceback

                st.code(traceback.format_exc())

    # User input
    user_input = st.chat_input("Type your message here...")

    if user_input:
        # Create message with text and optional image
        if st.session_state.current_image:
            user_message = {
                "role": "user",
                "content": [
                    st.session_state.current_image,
                    {"type": "text", "text": user_input}
                ]
            }

            # Reset the image state after creating the message
            # This way the uploader is reset but the sample images remain
            st.session_state.uploader_key += 1  # Increment to reset the uploader
        else:
            user_message = {
                "role": "user",
                "content": user_input
            }

        # Add user message to chat history
        st.session_state.messages.append(user_message)

        # Display user message
        with st.chat_message("user"):
            if isinstance(user_message["content"], list):
                for item in user_message["content"]:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            st.markdown(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            image_url = item["image_url"]["url"]
                            if image_url.startswith("data:image"):
                                image_data = image_url.split(",")[1]
                                image_bytes = base64.b64decode(image_data)
                                image = Image.open(BytesIO(image_bytes))
                                st.image(image, use_container_width=True)
                    elif isinstance(item, str):
                        st.markdown(item)
            else:
                st.markdown(user_message["content"])

        # Now that we've used the image, we can clear it for future uploads
        st.session_state.current_image = None
        st.session_state.processed_image = False

        # Rerun to display user message and trigger assistant response
        st.rerun()

# Sample images and sidebar
with right_col:
    # Sidebar-like area for settings
    st.header("Settings")

    model_name = st.selectbox(
        "Select model",
        ["qwen2-vl-7b-instruct"],  # You can add more models later
        index=0,
        key="model_name"  # This ties the widget to session state
    )

    st.header("Upload Image")

    # Use a key based on uploader_key to force re-rendering
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        key=f"uploader_{st.session_state.uploader_key}"
    )

    # Process the uploaded file
    if uploaded_file is not None and not st.session_state.processed_image:
        # Display the uploaded image
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Convert to base64 and store for use in the chat
            img_base64 = encode_image(image)
            st.session_state.current_image = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}"
                }
            }
            # Mark as processed to prevent re-processing
            st.session_state.processed_image = True
        except Exception as e:
            st.error(f"Error loading image: {e}")

    # Clear image button
    if st.session_state.current_image and st.button("Clear Image"):
        clear_image()
        st.success("Image cleared!")

    # Clear chat button
    if st.button("Clear Chat"):
        clear_chat()
        st.success("Chat history cleared!")

    # Sample images section
    st.header("Sample Medical Images")

    # For debugging purposes, let's add a file path check
    jsonl_path = "/mnt/8T/high_modality/geom_valid_sampled.jsonl"

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
        # Use cached samples if they exist, otherwise load and cache them
        if st.session_state.loaded_samples is None:
            # Load all samples from JSONL first
            all_samples = []
            try:
                if os.path.exists(jsonl_path):
                    with open(jsonl_path, 'r') as f:
                        lines = f.readlines()

                    for line in lines:
                        try:
                            all_samples.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

                # If no samples were loaded, use our test samples
                if not all_samples:
                    st.warning("Using test samples since no data was loaded from JSONL")
                    all_samples = sample_data

                # Select consistent random samples using pre-generated indices
                samples = []
                num_samples = min(12, len(all_samples))
                for i in range(num_samples):
                    index = st.session_state.sample_indices[i] % len(all_samples)
                    samples.append(all_samples[index])

                # Cache the samples
                st.session_state.loaded_samples = samples
            except Exception as e:
                st.error(f"Error loading all samples: {e}")
                st.session_state.loaded_samples = sample_data

        # Use the cached samples
        samples = st.session_state.loaded_samples

        # Create a 3x4 grid using st.columns and st.container
        n_cols = 3
        n_rows = 4

        for row in range(n_rows):
            cols = st.columns(n_cols)
            for col in range(n_cols):
                sample_idx = row * n_cols + col

                if sample_idx < len(samples):
                    sample = samples[sample_idx]

                    # Get the problem and clean it
                    problem_text = sample["problem"].replace("<image>\n", "")

                    # Get the image path
                    try:
                        img_path = os.path.join("/mnt/8T/high_modality", sample["images"][0])
                        if not os.path.exists(img_path):
                            img_path = test_image_path
                    except (KeyError, IndexError):
                        img_path = test_image_path

                    # Display the image and button in the column
                    with cols[col]:
                        try:
                            img = Image.open(img_path)
                            # Resize for thumbnails
                            img.thumbnail((150, 150))
                            st.image(img, use_container_width=True)

                            # Create unique button key for each image
                            if st.button(f"Select", key=f"img_btn_{sample_idx}"):
                                # Store the selected sample for use in messages
                                st.session_state.selected_sample = sample_idx

                                # Convert to base64
                                img_base64 = encode_image(img)

                                # Store image in session state
                                st.session_state.current_image = {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{img_base64}"
                                    }
                                }

                                # Create user message with the selected image and prompt
                                user_message = {
                                    "role": "user",
                                    "content": [
                                        st.session_state.current_image,
                                        {"type": "text", "text": problem_text}
                                    ]
                                }

                                # Add to messages
                                st.session_state.messages.append(user_message)

                                # We don't clear current_image yet, so it's still available
                                # for the next step in the conversation

                                # Force a rerun to update the chat
                                st.rerun()

                        except Exception as e:
                            st.error(f"Error with image {sample_idx}: {e}")

    except Exception as e:
        st.error(f"Error with sample images: {e}")