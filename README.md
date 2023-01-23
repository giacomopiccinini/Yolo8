if it does not find the model in that path, it is downloaded and stored at that location. 

if you want to run on multiple images at once, import them as numpy arrays and then pass a list of array [image_1, image_2, ...]

access boxes, classes and probabilities with

results.boxes.boxes
results.boxes.cls
results.boxes.conf

notice that each row in boxes.boxes has length 6 because it contains also the conf and cls as 5th and 6th component 

Custom train following 
https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data

Use model.predict(image, conf=0.2) to set a threshold in confidence under which we discard results, defaults to 0.25

xyxy are (x_bottom_left, y_bottom_left, x_top_right, y_top_right)