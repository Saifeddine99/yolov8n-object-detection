# YOLOv8 Object Detection Web Application

A Streamlit-based web application for real-time object detection using YOLOv8. This project allows users to detect multiple objects in images with customizable settings and provides a user-friendly interface.

## 🎥 Tutorial Video

This project is based on the following YouTube tutorial:
[YOLOv8 Object Detection Tutorial](https://www.youtube.com/watch?v=OO6dm6KcmlI)

## 🚀 Features

- Real-time object detection using YOLOv8
- Support for 80+ COCO dataset classes
- Customizable detection settings:
  - Class selection
  - Maximum number of bounding boxes
  - Confidence score threshold
- Interactive web interface using Streamlit
- Image upload and processing
- Download detected images with bounding boxes
- Detailed detection results display

## 🛠️ Installation & Usage

### Quick Start (Recommended)

1. Clone the repository:

```bash
git clone https://github.com/Saifeddine99/yolov8n-object-detection.git
cd yolov8_object_detection
```

2. Make the setup script executable:

```bash
chmod +x run.sh
```

3. Run the setup script:

```bash
./run.sh
```

This script will automatically:

- Create and activate a virtual environment
- Install all required dependencies
- Start the application on localhost

### Manual Setup (Alternative)

If you prefer to set up manually:

1. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run app.py
```

## 📦 Project Structure

```
yolov8_object_detection/
├── app.py              # Main Streamlit application
├── prediction.py       # Model loading and prediction functions
├── utils.py           # Utility functions
├── image_processing/  # Image processing modules
├── requirements.txt   # Project dependencies
├── run.sh            # Automated setup and run script
└── yolov8n.pt        # YOLOv8 model weights
```

## 🎯 Using the Application

1. Once the application is running, open your web browser and navigate to http://localhost:8501

2. Use the sidebar to:

   - Select object classes to detect
   - Set maximum number of bounding boxes
   - Adjust confidence threshold
   - Upload an image

3. Click "Detect Objects" to perform object detection

4. View the results and download the annotated image if desired

## 🔧 Dependencies

The project uses several key libraries:

- ultralytics (YOLOv8)
- streamlit (Web interface)
- torch (PyTorch for deep learning)
- opencv-python (Image processing)
- pillow (Image handling)

For a complete list of dependencies, see `requirements.txt`
