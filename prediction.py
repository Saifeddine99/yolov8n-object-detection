import streamlit as st
from ultralytics import YOLO

def predict(picture, classes_index, MIN_SCORE_THRES, MAX_BOXES_TO_DRAW):
    # Initialize session state for model loading (only once)
    if "model" not in st.session_state:
        print("Dkhalna")
        st.session_state.model = YOLO("yolov8n.pt")
    isAllinList = 80 in classes_index
    if isAllinList is True:
        classes_index = classes_index.clear()
    print("Selected Classes: ", classes_index)

    if classes_index:
        results = st.session_state.model.predict(source = picture,
                                conf = MIN_SCORE_THRES,
                                classes = classes_index,
                                max_det = MAX_BOXES_TO_DRAW
                                )

    else:
        results = st.session_state.model.predict(source = picture,
                                conf = MIN_SCORE_THRES,
                                max_det = MAX_BOXES_TO_DRAW
                                )
        
    return results[0]