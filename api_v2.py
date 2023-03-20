import os
import modal
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile 

from Code.Detection.A_load_image import load_image
from Code.Detection.B_find_bounding_boxes import find_bounding_boxes
from Code.Detection.C_jsonify import jsonify

app = FastAPI()
stub = modal.Stub("yolo-api")

# Create image (Docker-like) to be used by Modal backend
image = modal.Image.debian_slim(python_version="3.10")

# Pip install packages
image = image.pip_install(
        "numpy",
        "pandas",
        "pyyaml",
        "fastapi[all]",
        "uvicorn",
        "python-multipart",
        "pillow",
        "ultralytics",
        "opencv-python"
    )

image = image.extend(dockerfile_commands=["FROM base", 
                                          "WORKDIR /root",
                                          "RUN apt-get update",
                                          "RUN apt-get -y upgrade",
                                          "RUN apt-get -y install ocl-icd-libopencl1",
                                          "RUN apt-get -y install opencl-headers",
                                          "RUN apt-get -y install clinfo",
                                          "RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y"])

# Model directory
model_directory = "Models"

#  Create mounting
mount = modal.Mount.from_local_dir(model_directory, remote_path="/")

# Place model where it can be read
image = image.copy(mount, remote_path="/root/Models")

# Assign image to stub
stub.image = image

@app.post("/detect")
def detect(image_file: UploadFile = File(...)):
    
    # Path to model directory
    model_directory_path = "Models/Pretrained"

    # Path to model and relative classes
    model_path = os.path.join(model_directory_path, "yolov8m.pt")
    class_path = os.path.join(model_directory_path, "classes.yaml")

    # Load model
    model = YOLO(model_path)
    
    # Load image from file
    image = load_image(image_file)

    # Find bounding boxes
    bounding_boxes = find_bounding_boxes(image, model=model, class_path=class_path, conf=0.25)[0]

    # Convert detection to json
    bounding_boxes_json = jsonify(bounding_boxes)
    
    return bounding_boxes_json

@stub.asgi(image=image)
def fastapi_app():
    return app
    