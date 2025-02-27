import os
import streamlit as st
from PIL import Image
from pillow_heif import register_heif_opener
import utils
from typing import Optional

from image_processing.pfm_reader import read_pfm

# Construct the path relative to the current script's directory
DEFAULT_IMAGE = os.path.join(os.path.dirname(__file__), "default_image.jpeg")

def read_picture() -> Optional[Image.Image]:
    """
    Handles image uploads, processes them, and displays the image in Streamlit.

    Returns:
        Optional[Image.Image]: A PIL Image object if successful, None otherwise.
    """
    uploaded_image = st.file_uploader(
        "Upload Image",
        type=['png', 'jpeg', 'jpg', 'bmp', 'dng', 'tif', 'tiff', 'webp', 'pfm', 'HEIC'],
        on_change=utils.deactivation_callback,
    )

    with st.spinner(text='Resource Loading...'):
        if uploaded_image is not None:
            st.info("Selected Image:")
            try:
                file_extension = uploaded_image.name.split(".")[-1].lower()

                if file_extension in ['png', 'jpeg', 'jpg', 'bmp', 'dng', 'tif', 'tiff', 'webp']:
                    picture = Image.open(uploaded_image).convert("RGB")  # Ensure RGB for consistent display
                    st.image(picture)
                    return picture

                elif file_extension == 'heic':
                    register_heif_opener()
                    picture = Image.open(uploaded_image).convert("RGB")  # Ensure RGB for consistent display
                    st.image(picture)
                    return picture

                elif file_extension == 'pfm':
                    pfm_data = read_pfm(uploaded_image)
                    if pfm_data is None: #check for none return from pfm_reader
                        st.error("Error reading PFM file.")
                        return None
                    picture = Image.fromarray(pfm_data, mode="RGB")
                    st.image(picture)
                    return picture

                else:
                    st.error(f"Unsupported file format: .{file_extension}")
                    return None

            except Exception as e:
                st.error(f"An error occurred with .{file_extension} extension: {e}")
                return None

        else:
            st.info("Default Image:")
            try:
                picture = Image.open(DEFAULT_IMAGE).convert("RGB")  # Ensure RGB for consistent display
                st.image(picture)
                return picture
            except FileNotFoundError:
                st.error(f"Default image '{DEFAULT_IMAGE}' not found.")
                return None
            except Exception as e:
                st.error(f"An error occurred while opening the default image: {e}")
                return None