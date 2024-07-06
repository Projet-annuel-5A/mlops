from google.cloud import storage
from utils.logger import logger

def extract_from_uri(uri: str):
    """
    Extract le bucket name and artifact folder path form the gs uri.
    """
    if not uri.startswith("gs://"):
        raise ValueError("L'URI must start with'gs://'")
    
    uri_parts = uri[5:].split('/', 1)
    if len(uri_parts) != 2:
        raise ValueError("uri must have the form of 'gs://bucket_name/path/to/blob'")
    
    bucket_name = uri_parts[0]
    blob_path = uri_parts[1]
    
    return bucket_name, blob_path    

def download_model_file(uri:str):
    """
    Download the model .pt weights file.
    """
    bucket_name, artifacts_path = extract_from_uri(uri)
    logger.info(f"Downloading model file from bucket: {bucket_name} and artifact path: {artifacts_path}")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"{artifacts_path}/weights/best.pt")
    blob.download_to_filename("model.pt")


