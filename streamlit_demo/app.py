import streamlit as st
from openai import OpenAI
import base64
from io import BytesIO
from PIL import Image

# Configure the page
st.set_page_config(
    page_title="Qwen2.5-VL Chat Demo",
    page_icon="🤖",
    layout="wide"
)

# Initialize the OpenAI client to connect to your local API
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"  # The API key doesn't matter as it's just a placeholder
)

# Set up the Streamlit app
st.title("Qwen2.5-VL Vision-Language Model Demo")
st.markdown("Chat with the Qwen2.5-VL model and upload images for analysis.")

# Initialize session state for chat history and temporary image storage
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_image" not in st.session_state:
    st.session_state.current_image = None


# Function to encode image to base64
def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


# Sidebar for settings and image upload
with st.sidebar:
    st.header("Settings")

    model_name = st.selectbox(
        "Select model",
        ["qwen2-vl-7b-instruct"],  # You can add more models later
        index=0
    )

    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert to base64 and store for use in the chat
        img_base64 = encode_image(image)
        st.session_state.current_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_base64}"
            }
        }

    if st.session_state.current_image and st.button("Clear Image"):
        st.session_state.current_image = None
        st.success("Image cleared!")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.current_image = None
        st.success("Chat history cleared!")

# Display chat messages - single unified chat interface
st.header("Chat")

# Just use Streamlit's built-in chat message rendering
# This avoids duplicate chat displays
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], str):
            st.markdown(message["content"])
        elif isinstance(message["content"], list):
            for item in message["content"]:
                if isinstance(item, str):
                    st.markdown(item)
                elif isinstance(item, dict) and item.get("type") == "image_url":
                    # Display images in chat history
                    image_url = item["image_url"]["url"]
                    if image_url.startswith("data:image"):
                        image_data = image_url.split(",")[1]
                        image_bytes = base64.b64decode(image_data)
                        image = Image.open(BytesIO(image_bytes))
                        st.image(image, use_column_width=True)

# User input - use Streamlit's chat_input which automatically handles new messages
user_input = st.chat_input("Type your message here...")

if user_input:
    # Create message with text and optional image
    if st.session_state.current_image:
        user_message = {
            "role": "user",
            "content": [
                st.session_state.current_image,
                {"type": "text", "text": user_input}  # Format text as a content object
            ]
        }
        # Clear the image after sending
        current_image = st.session_state.current_image
        st.session_state.current_image = None
    else:
        user_message = {
            "role": "user",
            "content": user_input
        }

    # Add user message to chat history
    st.session_state.messages.append(user_message)

    # Display user message - Streamlit will automatically add this to the chat
    with st.chat_message("user"):
        if isinstance(user_message["content"], list):
            for item in user_message["content"]:
                if isinstance(item, str):
                    st.markdown(item)
                elif isinstance(item, dict) and item.get("type") == "image_url":
                    image_url = item["image_url"]["url"]
                    if image_url.startswith("data:image"):
                        image_data = image_url.split(",")[1]
                        image_bytes = base64.b64decode(image_data)
                        image = Image.open(BytesIO(image_bytes))
                        st.image(image)
        else:
            st.markdown(user_message["content"])

    # Generate assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # Call the API with streaming - send the FULL conversation history
            response = client.chat.completions.create(
                model=model_name,
                messages=st.session_state.messages,  # Send all previous messages for context
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