from typing import List, Dict, Any
from pydantic import BaseModel


class ImageRequest(BaseModel):
    image: str

class PredictionRequest(BaseModel):
    instances: List[ImageRequest]

class PredictionResponse(BaseModel):
    predictions: List[Any]