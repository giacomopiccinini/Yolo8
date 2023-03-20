import io
import numpy as np
from PIL import Image


def load_image(image_file):
    
    """Load image from file being uploaded to the API"""

    # Load from file and convert to numpy array
    image = np.asarray(Image.open(io.BytesIO(image_file.file.read())))
    
    return image