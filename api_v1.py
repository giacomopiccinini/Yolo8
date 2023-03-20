import os
import cv2
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile 

from Code.Detection.A_load_image import load_image
from Code.Detection.B_find_bounding_boxes import find_bounding_boxes
from Code.Detection.C_jsonify import jsonify

app = FastAPI()

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
    