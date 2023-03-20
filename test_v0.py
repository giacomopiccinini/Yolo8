import api_v0

# Test endopoint
def test_detect_route():
    
    # Path to bus image
    file_path = "Input/bus.jpg"
    
    # Run inference
    response = api_v0.detect(file_path)

    # Set required response
    required_response = {'detection_0': \
                            {'x_top_left': 3, 'y_top_left': 229, 'x_bottom_right': 804, 'y_bottom_right': 741, 'confidence': 0.95947265625, 'class_numeric': 5, 'class': 'bus'}, \
                         'detection_1': \
                            {'x_top_left': 50, 'y_top_left': 400, 'x_bottom_right': 247, 'y_bottom_right': 904, 'confidence': 0.92724609375, 'class_numeric': 0, 'class': 'person'}, \
                         'detection_2': \
                            {'x_top_left': 668, 'y_top_left': 395, 'x_bottom_right': 810, 'y_bottom_right': 881, 'confidence': 0.92236328125, 'class_numeric': 0, 'class': 'person'}, \
                         'detection_3': \
                            {'x_top_left': 222, 'y_top_left': 411, 'x_bottom_right': 344, 'y_bottom_right': 861, 'confidence': 0.9013671875, 'class_numeric': 0, 'class': 'person'}, \
                         'detection_4': \
                            {'x_top_left': 0, 'y_top_left': 550, 'x_bottom_right': 78, 'y_bottom_right': 872, 'confidence': 0.79296875, 'class_numeric': 0, 'class': 'person'}}

    # Assert the reponse is correct
    assert response == required_response