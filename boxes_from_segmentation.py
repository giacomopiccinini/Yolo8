import logging
import os

from tqdm import tqdm
from pathlib import Path
from torchvision.datasets import ImageFolder
from torchvision.io import read_image

from Code.Parser.parser import parse
from Code.Techniques.Annotation.segmentation_to_detection import (
    labelled_mask_to_boxes,
    save_bounding_boxes,
)


if __name__ == "__main__":

    # Set logger
    logging.basicConfig(level=logging.NOTSET)
    log = logging.getLogger(__name__)

    # Parse arguments related to detection
    args = parse()["Convert"]

    # Get the source file(s)
    masks_source = args["source_mask"]

    # Ascertain type
    if Path(masks_source).is_dir():

        # Create dataset
        dataset = ImageFolder(root=masks_source)

    elif Path(masks_source).is_file():

        # Read image as PyTorch tensor
        mask = read_image(masks_source)

        # Recreate dataset-like structure (the second entry should be the label)
        dataset = [(mask, None)]

    else:

        # Raise error
        raise ValueError(
            "Path to masks is not admissible. It is neither a directory nor a file"
        )

    if Path(args["bbox_path"]).is_dir():

        # Create target directory if necessary
        os.makedirs(args["bbox_path"], exist_ok=True)

    elif Path(args["bbox_path"]).is_file():

        if len(dataset) != 1:
            # Raise error
            raise ValueError(
                "Output path is invalid. Trying to write boxes from multiple images on a single file"
            )

    else:

        # Raise error
        raise ValueError("Output path is invalid. It is neither a directory nor a file")

    # Loop over dataset
    for i in tqdm(range(len(dataset))):

        # Retrieve image (and label)
        mask, _ = dataset[i]

        if isinstance(dataset, ImageFolder):
            # Retrive image filename (no extension and no path to it)
            filename = Path(dataset.imgs[i][0]).stem
        else:
            filename = Path(masks_source).stem

        # Create YOLO boxes
        yolo_boxes = labelled_mask_to_boxes(mask)

        if Path(args["bbox_path"]).is_dir():
            # Create path where to store the bboxes
            bboxes_path = os.path.join(args["bbox_path"], filename)
        else:
            bboxes_path = args["bbox_path"]

        # Save the results
        save_bounding_boxes(yolo_boxes, bboxes_path, format=args["bbox_save_format"])
