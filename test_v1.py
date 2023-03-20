import api_v1
import requests
from fastapi.testclient import TestClient

# Test endopoint
def test_predict_route():
    
    # Create client for app
    client = TestClient(api_v1.app)
    
    # Filename of audio file to test the API against
    file_name = "Input/bus.jpg"

    # Post request
    response = client.post(
        "/detect", files={"image_file": ("image", open(file_name, "rb"), "image/jpg")}
    )

    # Assert the status code is 200
    assert response.status_code == 200
