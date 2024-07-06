from google.cloud import aiplatform
from utils.logger import logger

def get_model_uri(model_id:str = "7605563821884702720"):
    """
    Return the model artifact URI from the model version gived artifact uri after training.
    """
    aiplatform.init(project="annual-project-427112", location="europe-west1")
    model_uri = aiplatform.Model(model_id).uri
    logger.info(f"Importing model on uri: {model_uri} ...")
    return model_uri