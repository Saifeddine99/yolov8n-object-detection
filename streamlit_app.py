from ultralytics import YOLO
from PIL import Image
import streamlit as st

st.set_page_config(page_title = "Yolo V8 Multiple Object Detection on Pretrained Model", page_icon="🤖")

#################### Title #####################################################
st.markdown("<h3 style='text-align: center; color: red; font-family: font of choice, fallback font no1, sans-serif;'>Yolo V8</h3>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: black; font-family: font of choice, fallback font no1, sans-serif;'>Multiple Object Detection on Pretrained Model</h2>", unsafe_allow_html=True)

st.markdown('#') # inserts empty space
#################### /Title #####################################################


# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

def activation_callback():
    # Button was clicked!
    st.session_state.detection_button = True

def deactivation_callback():
    st.session_state.detection_button = False

DEMO_PIC = "default_image.jpeg"

if "detection_button" not in st.session_state:
    st.session_state.detection_button = False


#################### Parameters to setup ########################################
cocoClassesLst = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat", \
"dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",\
"baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog",\
"pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink",\
"refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush", "All"]

classes_index = st.sidebar.multiselect("Select Classes", range(
    len(cocoClassesLst)), format_func = lambda x: cocoClassesLst[x], default = 80)

isAllinList = 80 in classes_index
if isAllinList == True:
    classes_index = classes_index.clear()
    
print("Selected Classes: ", classes_index)

MAX_BOXES_TO_DRAW = st.sidebar.number_input('Maximum Boxes To Draw', value = 5, min_value = 1, max_value = 20) #max_det
MIN_SCORE_THRES = st.sidebar.slider('Min Confidence Score Threshold', min_value = 0.0, max_value = 1.0, value = 0.25)
#################### /Parameters to setup ########################################


#################### Image Upload ########################################
uploaded_image = st.sidebar.file_uploader(
    "Upload Image", type = ['png', 'jpeg', 'jpg', 'bmp', 'dng', 'mpo', 'tif', 'tiff', 'webp', 'pfm', 'HEIC'], on_change=deactivation_callback)

if uploaded_image is not None:
    with st.spinner(text = 'Resource Loading...'):
        st.sidebar.info("Uploaded Picture!")
        st.sidebar.image(uploaded_image)
        picture = Image.open(uploaded_image)

else:
    st.sidebar.info("Here is a Demo Picture!")
    st.sidebar.image(DEMO_PIC)
    picture = DEMO_PIC
#################### /Image Upload ########################################


#################### Object Detection ########################################
if st.button('Detect Objects', on_click=activation_callback) or st.session_state.detection_button:

    if classes_index:
        with st.spinner(text = 'Inferencing, Please Wait.....'):
            result = model.predict(source = picture,
                                    conf = MIN_SCORE_THRES,
                                    classes = classes_index,
                                    max_det = MAX_BOXES_TO_DRAW
                                    )
    
    else:
        with st.spinner(text = 'Inferencing, Please Wait.....'):
            result = model.predict(source = picture,
                                    conf = MIN_SCORE_THRES,
                                    max_det = MAX_BOXES_TO_DRAW
                                    )
            
    result[0].save(filename="result.jpg")
    
    with st.spinner(text = 'Preparing Images'):

        classes_dataframe = result[0].to_df()

        if len(classes_dataframe):
            st.markdown('#') # inserts empty space

            #  Display the original image with bounding boxes drawn around detected objects
            st.text("Here you have the original image with bounding boxes drawn around detected objects:")
            st.image("result.jpg", channels= "BGR")

            st.markdown('#') # inserts empty space

            # Show a list of all detected classes below the image
            st.text("Here you have the list of all detected classes:")
            if 'box' in classes_dataframe.columns:
                classes_dataframe.drop('box', axis='columns', inplace=True)
            st.dataframe(classes_dataframe)

            # Option to download the image with bounding boxes:
            with open("result.jpg", "rb") as file:
                if st.download_button(
                    label="Download image",
                    data=file,
                    file_name="result.jpg",
                    mime="image/jpg",
                ):
                    st.balloons()

        else:
            st.error("Sorry, No object detected!")
#################### /Object Detection ########################################