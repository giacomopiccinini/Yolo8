import argparse
from argparse import ArgumentParser


def parse():

    """Parse command line arguments"""

    # Initiate argparser
    parser = ArgumentParser()

    # * #################################
    # * ########### DETECTION ###########
    # * #################################

    # Add group for general info on the project
    detect_group = parser.add_argument_group("Detect", "Arguments for detection")

    # Add arguments

    # Model
    detect_group.add_argument(
        "--model",
        const="Models/yolov8m.pt",
        default="Models/yolov8m.pt",
        nargs="?",
        type=str,
        help="Path to the pretrained YOLO model to be used",
    )

    # Source
    detect_group.add_argument(
        "--source",
        const="Input",
        default="Input",
        nargs="?",
        type=str,
        help="Path to images/videos to use",
    )

    # Confidence
    detect_group.add_argument(
        "--confidence",
        const=0.25,
        default=0.25,
        nargs="?",
        type=float,
        help="Confidence in detection, i.e. threhsold over which we consider a detection",
    )

    # Classes
    detect_group.add_argument(
        "--classes",
        default=[],
        nargs="+",
        help="Classes to keep, defaults to all",
    )

    # Show result
    detect_group.add_argument(
        "--show",
        const=True,
        default=True,
        nargs="?",
        type=bool,
        help="Show result of detection",
    )

    # Draw bounding box
    detect_group.add_argument(
        "--bb",
        const=True,
        default=True,
        nargs="?",
        type=bool,
        help="Draw bounding box on image",
    )

    # Save detection result
    detect_group.add_argument(
        "--save",
        const=True,
        default=True,
        nargs="?",
        type=bool,
        help="Save image with bbs",
    )

    # * #################################
    # * ############ PARSING ############
    # * #################################

    # Parse arguments
    args = parser.parse_args()

    # Initialise dictionary for groups of arguments
    arg_groups = {}

    # Add arguments to relevant group
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = argparse.Namespace(**group_dict)

    return arg_groups
