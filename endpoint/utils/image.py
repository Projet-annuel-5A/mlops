from PIL import Image
import numpy as np
import base64
import cv2

def transform_image(b64_img: str):
    image_bytes = base64.b64decode(b64_img)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    return pil_image