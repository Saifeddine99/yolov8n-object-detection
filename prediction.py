import streamlit as st
from ultralytics import YOLO
import time


@st.cache_resource(show_spinner="Loading Model...")
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

@st.cache_data(show_spinner='Inferencing, Please Wait.....')
def predict(picture, classes_index, MIN_SCORE_THRES, MAX_BOXES_TO_DRAW):

    st.write("launching predict function date:", time.time()) # Debuggeing

    isAllinList = 80 in classes_index
    if isAllinList is True:
        classes_index = classes_index.clear()
    print("Selected Classes: ", classes_index)

    if classes_index:
        results = model.predict(source = picture,
                                conf = MIN_SCORE_THRES,
                                classes = classes_index,
                                max_det = MAX_BOXES_TO_DRAW
                                )

    else:
        results = model.predict(source = picture,
                                conf = MIN_SCORE_THRES,
                                max_det = MAX_BOXES_TO_DRAW
                                )
            
    return results[0]