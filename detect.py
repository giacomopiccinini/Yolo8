import cv2
import logging

logging.getLogger("everett").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)

from tqdm import tqdm
from ultralytics import YOLO
from pathlib import Path

from Code.Loader.StreamLoader import StreamLoader
from Code.Loader.MediaLoader import MediaLoader
from Code.Parser.parser import parse
from Code.Detection.A_find_bounding_boxes import find_bounding_boxes
from Code.Techniques.Annotation.draw_bboxes import draw_bounding_boxes


if __name__ == "__main__":

    # Set logger
    logging.basicConfig(level=logging.NOTSET)
    log = logging.getLogger(__name__)

    # Parse arguments related to detection
    args = parse()["Detect"]

    if args.source != "webcam":
        # Load media
        log.info("Loading media")
        Data = MediaLoader(args.source)
    else:
        # Load Stream
        log.info("Loading stream")
        Data = StreamLoader()

    # Load a pre-trained model
    log.info("Loading model")
    model = YOLO(args.model)
    
    # Get path to classes stored in yaml
    class_path = Path.joinpath(Path(args.model).parent, "classes.yaml")

    # Set variables for video writing (when necessary)
    video_name, video_writer = None, None

    # Loop over images (or frames) in dataset
    for name, image, capture in tqdm(Data):

        # Find bounding boxes
        bounding_boxes = find_bounding_boxes(image, model=model, class_path=class_path, conf=args.confidence)[0]
        
        # Filter by class
        if args.classes:
            
            bounding_boxes = bounding_boxes.query("`class`== @args.classes")

        # Write boxes if needed
        if args.bb:
            image = draw_bounding_boxes(image, bounding_boxes)

        # When streaming, always show the result
        if Data.mode == "Stream":
            cv2.imshow(name, image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                Data.capture.release()
                break

        # Show image (on request) if not streaming
        elif args.show:

            # Show image
            cv2.imshow(name, image)

            # Different rules depending on the type of data
            if Data.mode == "Video":
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    Data.capture.release()
                    break
            # Case of images
            else:
                cv2.waitKey(0)

        # Save image (on request)
        if args.save:

            # In the case of image
            if Data.mode == "Image":
                cv2.imwrite(f"Output/{name}", image)

            # In case of stream or video
            else:

                # If we can read from the video
                if capture and Data.mode == "Video":

                    # Get video FPS
                    fps = capture.get(cv2.CAP_PROP_FPS)

                    # Get video width
                    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

                    # Get video height
                    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Case of stream
                else:

                    # Fetch values
                    fps = 8  # Data.fps
                    width = Data.width
                    height = Data.height

                # Initiate video name the first time
                if not video_name:
                    video_name = name

                # Case of new video
                if video_name != name:

                    # Release old video writer
                    video_writer.release()

                    # Re-initialise video writer
                    video_writer = None

                    # Reset video name
                    video_name = name

                # If no video writer is instantiated
                if not video_writer:

                    # Create a new video writer
                    video_writer = cv2.VideoWriter(
                        f"Output/{name}",
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )

                video_writer.write(image)
