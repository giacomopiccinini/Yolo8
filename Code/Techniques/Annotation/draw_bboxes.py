import cv2
import numpy as np
import pandas as pd
from copy import deepcopy


def draw_bounding_boxes(
    image: np.array,
    bounding_boxes: pd.DataFrame,
    write_class: bool = True,
    write_confidence: bool = True,
    colours: dict = {"default": (255, 0, 0)},
) -> np.array:

    """Draw bounding boxes for an image given the YOLO detections accomodated
    in a pandas DataFrame.

    Recall that the y-direction stretches from the top downwards, so some formulas might
    be misleading."""

    # Instantiate image with bounding boxes
    image_bb = deepcopy(image)

    for bbox in bounding_boxes.iterrows():

        # Define colour based on user's choice

        # If user has defined a colour for that class
        if bbox[1]["class"] in colours.keys():
            colour = colours[bbox[1]["class"]]
        # Else use default colour
        else:
            colour = colours["default"]

        # Draw rectangle
        image_bb = cv2.rectangle(
            image_bb,
            (bbox[1]["x_top_left"], bbox[1]["y_top_left"]),
            (bbox[1]["x_bottom_right"], bbox[1]["y_bottom_right"]),
            colour,
            1,
        )

        if write_class or write_confidence:

            # Get confidence in percentage form
            confidence = str(round(bbox[1]["confidence"] * 100))

            # Define font to use
            FONT = cv2.FONT_HERSHEY_SIMPLEX

            # Create text to write. Use the flags to remove undesired parts
            text_to_write = (
                bbox[1]["class"] * write_class
                + " "
                + confidence * write_confidence
                + "%"
            )

            # Determine the space used by the text to correctly dimension the rectangle for the text
            text_size, baseline = cv2.getTextSize(
                text_to_write, fontFace=FONT, fontScale=0.5, thickness=1
            )

            # Get the top left corner of the rectangle including the text (minus because if counted backwards)
            text_top_left = (
                bbox[1]["x_top_left"],
                bbox[1]["y_top_left"] - text_size[1] - baseline,
            )

            # Get the bottom right corner of the rectangle including the text
            text_bottom_right = (
                bbox[1]["x_top_left"] + text_size[0],
                bbox[1]["y_top_left"],
            )

            # Draw rectangle
            cv2.rectangle(
                image_bb, text_top_left, text_bottom_right, colour, -1, cv2.LINE_AA
            )

            # Put text
            cv2.putText(
                image_bb,
                text_to_write,
                (bbox[1]["x_top_left"], bbox[1]["y_top_left"] - baseline + 1),
                fontFace=FONT,
                fontScale=0.5,
                color=[255, 255, 255],
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    return image_bb
