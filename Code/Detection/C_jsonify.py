import pandas as pd

def jsonify(bounding_boxes: pd.DataFrame):
    
    """ Convert a Pandas df containing bounding boxes into a JSON """
    
    # Turn the dataframe into a list of dictionaries
    detection_list = bounding_boxes.to_dict(orient="records")
    
    # Convert it to a pure dictionary
    results_json = {f"detection_{i}": detection for i, detection in enumerate(detection_list)}
    
    return results_json