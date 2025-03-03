import datetime
import time
import streamlit as st
from ultralytics import YOLO


@st.cache_resource(ttl=datetime.timedelta(days=2), show_spinner="Loading Model...") # 👈 Cache data for 2 days
def load_model():
    return YOLO("yolov8n.pt")


@st.cache_data(max_entries=1000, show_spinner='Inferencing, Please Wait.....') # 👈 Maximum 1000 entries in the cache
def predict(_model, picture, classes_index, MIN_SCORE_THRES, MAX_BOXES_TO_DRAW): # 👈 Don't hash _model
    
    inference_date = time.time()
    
    try:
        results = _model.predict(source=picture,
                                conf=MIN_SCORE_THRES,
                                classes=classes_index,
                                max_det=MAX_BOXES_TO_DRAW)

        return results[0], inference_date

    except Exception as e:
        st.error(f"Error during inference: {e}")
        return None, inference_date