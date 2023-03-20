import pandas as pd
import numpy as np
from yaml import safe_load


def find_bounding_boxes(images: list, model, class_path, conf:float=0.25) -> list:

    """Find the bounding boxes associated to a list of images. Images should already be
    in the numpy array format. Returns a list of pandas DataFrame corresponding to the
    result of the detection"""

    # Load YOLO classes
    with open(class_path, "r") as file:
        yolo_classes = safe_load(file)["names"]

    # Define dataframe columns
    columns = [
        "x_top_left",
        "y_top_left",
        "x_bottom_right",
        "y_bottom_right",
        "confidence",
        "class_numeric",
    ]

    # Use the model
    results = model.predict(images, conf=conf)

    # Init list of detection
    detection_df_list = []

    # Loop over result for every image (YOLO always returns a list, even with a single file)
    for result in results:

        # Create detection DataFrame
        df = pd.DataFrame(result.boxes.boxes.cpu().numpy(), columns=columns)

        # Fix datatypes
        df = df.convert_dtypes()

        if len(df) > 0:
            # Add descriptive class
            df["class"] = np.vectorize(yolo_classes.get)(df["class_numeric"])

        # Append to list
        detection_df_list.append(df)

    return detection_df_list
