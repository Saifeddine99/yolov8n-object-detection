import io
import streamlit as st
from PIL import Image

import utils

st.set_page_config(page_title="Yolo V8 Multiple Object Detection", page_icon="🤖")

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

# Initialize session state for detection status
if "detection_button" not in st.session_state:
    st.session_state.detection_button = False

#################### Title #####################################################
st.markdown("<h3 style='text-align: center; color: red;'>Yolo V8</h3>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: black;'>Multiple Object Detection</h2>",
            unsafe_allow_html=True)
st.markdown('#')
#################### /Title #####################################################


#################### Sidebar #####################################################
with st.sidebar:
    #Parameters to setup:
    classes_index = st.multiselect("Select Classes", range(
        len(COCO_CLASSES)), format_func = lambda x: COCO_CLASSES[x], default = 80)

    MAX_BOXES_TO_DRAW = st.number_input('Maximum Boxes To Draw', value = 5, min_value = 1, max_value = 20, on_change = utils.deactivation_callback) #max_det
    MIN_SCORE_THRES = st.slider('Min Confidence Score Threshold', min_value = 0.0, max_value = 1.0, value = 0.25, on_change = utils.deactivation_callback)

    # Image Upload:
    from image_processing.image_processor import read_picture
    picture = read_picture()
#################### /Sidebar #####################################################

if picture:
    #################### Object Detection ########################################
    if st.button('Detect Objects', on_click = utils.activation_callback) or st.session_state.detection_button:

        from prediction import predict
        import time

        st.write("app.py, calling predict function:", time.time()) # Debuggeing
        
        result = predict(picture, classes_index, MIN_SCORE_THRES, MAX_BOXES_TO_DRAW)
            
        with st.spinner(text = 'Image Processing:'):

            # Converting result to DataFrame:
            classes_dataframe = result.to_df()

            if len(classes_dataframe):

                annotated_image_np = result.plot() # BGR-order numpy array
                # Convert NumPy array to a PIL image
                annotated_image = Image.fromarray(annotated_image_np[..., ::-1]) # RGB-order PIL image

                st.markdown('#') # insert empty space

                #  Display the original image with bounding boxes drawn around detected objects
                st.text("Here you have the original image with bounding boxes drawn around detected objects:")
                st.image(annotated_image)

                st.markdown('#') # insert empty space

                # Show a list of all detected classes below the image
                st.text("Here you have the list of all detected classes:")
                if 'box' in classes_dataframe.columns:
                    classes_dataframe.drop('box', axis='columns', inplace=True)
                st.dataframe(classes_dataframe)


                # Convert the PIL image to a BytesIO object
                img_bytes = io.BytesIO()
                annotated_image.save(img_bytes, format="PNG")  # Save in-memory as PNG
                img_bytes.seek(0)  # Move to the start of the BytesIO buffer

                # Option to download the image with bounding boxes:
                if st.download_button(
                    label="Download image",
                    data=img_bytes,
                    file_name="Annotated_image.jpg",
                    mime="image/jpg",
                ):
                    st.balloons()

            else:
                st.error("Sorry, No object detected!")
    #################### /Object Detection ########################################
else:
    st.error("Can't read Image! Try again...")