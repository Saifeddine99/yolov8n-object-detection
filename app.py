import io
import streamlit as st
from PIL import Image
import time
import utils

from image_processing.image_processor import read_picture
from prediction import load_model, predict

# Constants
COCO_CLASSES = [  # List of all Classes
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush", "all"
]
DEFAULT_CLASSES = [80] #index of "all"
DEFAULT_MAX_BOXES = 5
DEFAULT_MIN_SCORE = 0.25

if "detection_button" not in st.session_state:
    st.session_state.detection_button = False

def setup_page():
    st.set_page_config(page_title="Yolo V8 Multiple Object Detection", page_icon="🤖")

def display_title():
    st.markdown("<h3 style='text-align: center; color: red;'>Yolo V8</h3>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: black;'>Multiple Object Detection</h2>", unsafe_allow_html=True)
    st.markdown('#')

def setup_sidebar():
    with st.sidebar:
        classes_index = st.multiselect("Select Classes", range(len(COCO_CLASSES)), format_func=lambda x: COCO_CLASSES[x], default=DEFAULT_CLASSES, on_change=utils.deactivation_callback)
        max_boxes = st.number_input('Maximum Boxes To Draw', value=DEFAULT_MAX_BOXES, min_value=1, max_value=20, on_change=utils.deactivation_callback)
        min_score = st.slider('Min Confidence Score Threshold', min_value=0.0, max_value=1.0, value=DEFAULT_MIN_SCORE, on_change=utils.deactivation_callback)
        picture = read_picture()
    return classes_index, max_boxes, min_score, picture

def display_results(result, annotated_image):
    st.markdown('#')
    st.text("Here you have the original image with bounding boxes drawn around detected objects:")
    st.image(annotated_image)
    st.markdown('#')
    st.text("Here you have the list of all detected classes:")
    classes_dataframe = result.to_df()
    if 'box' in classes_dataframe.columns:
        classes_dataframe.drop('box', axis='columns', inplace=True)
    st.dataframe(classes_dataframe)
    img_bytes = io.BytesIO()
    annotated_image.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    if st.download_button(label="Download image", data=img_bytes, file_name="Annotated_image.jpg", mime="image/jpg"):
        st.balloons()

def main():
    setup_page()
    display_title()
    classes_index, max_boxes, min_score, picture = setup_sidebar()

    if 80 in classes_index or len(classes_index) == 0:
        classes_index = None
    print("Selected Classes: ", classes_index)

    if picture:
        if st.button('Detect Objects', on_click = utils.activation_callback) or st.session_state.detection_button:
            model = load_model()
            
            runtime = time.time()
            result, inference_date = predict(model, picture, classes_index, min_score, max_boxes)

            if result:
                if abs(inference_date - runtime) > 1:
                    st.success("Data Loaded directly from Cache!")
                else:
                    st.info("Data loaded after model inference!")

                annotated_image_np = result.plot()
                annotated_image = Image.fromarray(annotated_image_np[..., ::-1])
                display_results(result, annotated_image)

            else:
                st.error("Sorry, No object detected!")
    else:
        st.error("Can't read Image! Try again...")

if __name__ == "__main__":
    main()