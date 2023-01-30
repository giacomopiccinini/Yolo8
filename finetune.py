import logging

from ultralytics import YOLO
from Code.Parser.parser import parse

if __name__ == '__main__':
    
    # Set logger
    logging.basicConfig(level=logging.NOTSET)
    log = logging.getLogger(__name__)

    # Parse arguments related to detection
    args = parse()["Finetune"]

    # Load a model
    model = YOLO(args.pretrained)

    # Finetune the model
    results = model.train(data=args.data, 
                          epochs=args.epochs, 
                          batch=args.batch, 
                          pretrained=True,
                          lr0=0.0001,
                          lrf=0.01)  
    
    # Save model
    # model.export()