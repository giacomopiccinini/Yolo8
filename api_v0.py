import os
import cv2
from ultralytics import YOLO

from Code.Detection.B_find_bounding_boxes import find_bounding_boxes
from Code.Detection.C_jsonify import jsonify

def detect(image):
    
    image = cv2.imread(image, -1)

    # Path to model directory
    model_directory_path = "Models/Pretrained"

    # Path to model and relative classes
    model_path = os.path.join(model_directory_path, "yolov8m.pt")
    class_path = os.path.join(model_directory_path, "classes.yaml")

    # Load model
    model = YOLO(model_path)

    # Find bounding boxes
    bounding_boxes = find_bounding_boxes(image, model=model, class_path=class_path, conf=0.25)[0]

    # Convert detection to json
    bounding_boxes_json = jsonify(bounding_boxes)
    
    return bounding_boxes_json