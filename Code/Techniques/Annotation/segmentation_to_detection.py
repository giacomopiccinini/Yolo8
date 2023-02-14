import torch
import numpy as np
from torchvision.ops import masks_to_boxes
from pathlib import Path


def labelled_mask_to_boxes(mask: torch.Tensor) -> torch.Tensor:

    """Given a *labelled* (i.e. each entity has a different colour) mask of
    homogeneous objects, return the bounding boxes in YOLO format as a Torch tensor"""

    # Get the unique colors, as these are object ids
    object_ids = torch.unique(mask)

    # Remove the first ID, i.e. the background
    object_ids = object_ids[1:]

    # Split the color-encoded mask into a set of boolean masks
    masks = mask == object_ids[:, None, None]

    # Get boxes in (xmin, ymin, xmax, ymax) format
    boxes = masks_to_boxes(masks)

    # Create column of IDs for annotation purposes
    id_column = torch.zeros(boxes.shape[0])

    # Create YOLO-like boxes
    yolo_boxes = torch.column_stack((id_column, boxes))

    # Convert to integer
    yolo_boxes = yolo_boxes.type(torch.int)

    return yolo_boxes


def save_bounding_boxes(
    yolo_boxes: torch.Tensor, file_name: str = "annotation.txt", format: str = "%i"
) -> None:

    """Save bounding boxes to text file"""

    # Check target file is .txt
    if Path(file_name).suffix != ".txt":
        raise ValueError("File name must be a .txt file")

    # Save yoloboxes in specific format (default: integer)
    np.savetxt(file_name, yolo_boxes.numpy(), fmt=format)
