from utils.storage import download_model_file
from contextlib import asynccontextmanager
from utils.model import get_model_uri
from utils.logger import logger
from ultralytics import YOLO
from fastapi import FastAPI
import torch

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Starting application lifespan...")

    # DOWNLOADING MODEL
    model_uri = get_model_uri()
    download_model_file(uri=model_uri)
    ml_models['yolo'] = YOLO('./model.pt')
    
    logger.info(f"Model downloaded and loaded from {model_uri}")
    logger.info("Starting with CUDA: %s", torch.cuda.is_available())

    yield


    # CLEANING MODEL
    del ml_models['yolo']
    torch.cuda.empty_cache()
    logger.info("Cleaned up model and released resources")