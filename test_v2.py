import requests

def test_predict_route():
    
    """ Test app works """

    # Filename of audio file to test the API against
    file_name = "Input/bus.jpg"

    # Post request
    response = requests.post(
        "https://giacomopiccinini--yolo-api-fastapi-app-dev.modal.run/detect", files={"image_file": ("image", open(file_name, "rb"), "image/jpg")}
    )

    # Assert the status code is 200
    assert response.status_code == 200
