## Inference

To run the model in inference mode

```
python detect.py --source <PATH_TO_IMAGE> --model <PATH_TO_MODEL> --confidence <CONFIDENCE_THRESHOLD> --classes <CLASSES_TO_KEEP>

```

Explanation of parameters:

- *source*: path to file to be processed; 
- *model*: path to the model to use for inference (the name of the pre-trained models available can be found in `Info/models.yaml`). If a pre-trained model is not present in the specified path, it will be automatically downloaded;
- *confidence*: acceptable confidence in the prediction. Must be a float number between 0 and 1. Predictions below this confidence threshold will be ignored;
- *classes*: classes to keep. These could be e.g. "person", "car", etc. The name of the classes for pre-trained models are available at `Info/classes.yaml`. 