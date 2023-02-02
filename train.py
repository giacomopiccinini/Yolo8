import logging

logging.getLogger("everett").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)

import comet_ml

from ultralytics import YOLO
from Code.Parser.parser import parse

if __name__ == '__main__':
    
    # Set logger
    logging.basicConfig(level=logging.NOTSET)
    log = logging.getLogger(__name__)

    # Parse arguments related to detection
    args = parse()["Train"]
    
    # Experiment
    experiment = comet_ml.Experiment()

    # Load a model
    model = YOLO(args.pretrained)
    
    # Finetune the model
    model.train(data=args.data, 
                          epochs=args.epochs, 
                          batch=args.batch, 
                          pretrained=True,
                          lr0=0.0001,
                          lrf=0.01)  