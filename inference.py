import requests
import base64
import cv2
from google.auth import default
from google.auth.transport.requests import Request

# LOCAL INF

url = "http://localhost:8080/predict"

image = cv2.imread("test.jpg")
_, buffer = cv2.imencode('.jpg', image)
image_bytes = base64.b64encode(buffer).decode('utf-8')

data = {
    "instances": [
        {
            "image": image_bytes
        }
    ]
}

response = requests.post(url, json=data)
try:
    response.raise_for_status()
    print(response.json())
except requests.exceptions.HTTPError as err:
    print(f"HTTP error occurred: {err}")
except requests.exceptions.RequestException as err:
    print(f"Other error occurred: {err}")


# # REMOTE INF


# PROJECT_ID = "492916107091"
# ENDPOINT_ID = "6233091835443544064"

# image = cv2.imread("test.jpg")
# _, buffer = cv2.imencode('.jpg', image)
# image_bytes = base64.b64encode(buffer).decode('utf-8')

# data = {
#     "instances": [
#         {
#             "image": image_bytes
#         }
#     ]
# }

# credentials, project_id = default()
# credentials.refresh(Request())
# token = credentials.token

# endpoint_url = f"https://europe-west1-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/europe-west1/endpoints/{ENDPOINT_ID}:predict"

# headers = {
#     "Authorization": f"Bearer {token}",
#     "Content-Type": "application/json"
# }

# print(token)

# response = requests.post(endpoint_url, headers=headers, json=data)
# try:
#     response.raise_for_status()
#     print(response.json())
# except requests.exceptions.HTTPError as err:
#     print(f"HTTP error occurred: {err}")
# except requests.exceptions.RequestException as err:
#     print(f"Other error occurred: {err}")
