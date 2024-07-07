from google.cloud.storage import transfer_manager
from google.cloud import storage
import os

# def download_bucket(bucket_name, destination_folder):
#     """
#     Download an entire bucket.
#     """
#     storage_client = storage.Client()
#     bucket = storage_client.bucket(bucket_name)
#     blobs = bucket.list_blobs()

#     for blob in blobs:
#         destination_file_name = os.path.join(destination_folder, blob.name)
        
#         if not os.path.exists(os.path.dirname(destination_file_name)):
#             os.makedirs(os.path.dirname(destination_file_name))
        
#         if blob.name.endswith('/'):
#             continue
        
#         blob.download_to_filename(destination_file_name)
#         print(f"Downloaded {blob.name} to {destination_file_name}")

def download_bucket(bucket_name: str, destination_folder: str, workers: int = 3):
    """
    Download an entire bucket using transfer_manager for concurrent downloads.
    """
    print(f"Downloading training and validation data on {bucket_name} into {destination_folder} ...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    blobs = list(bucket.list_blobs())
    blob_names = [blob.name for blob in blobs]
    print(f"{len(blob_names)} files ready to download")

    for blob_name in blob_names:
        destination_file_name = os.path.join(destination_folder, blob_name)
        if not os.path.exists(os.path.dirname(destination_file_name)):
            os.makedirs(os.path.dirname(destination_file_name))
    
    results = transfer_manager.download_many_to_path(
        bucket,
        blob_names,
        destination_directory=destination_folder,
        max_workers=workers
    )
    
    for blob_name, result in zip(blob_names, results):
        destination_file_name = os.path.join(destination_folder, blob_name)
        if isinstance(result, Exception):
            print(f"Failed to download {blob_name} due to exception: {result}")
        else:
            print(f"Downloaded {blob_name} to {destination_file_name}")

def upload_folder_to_bucket(folder_path, bucket_name, model_version):
    """
    Upload the entire training result folder.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            blob_path = os.path.relpath(file_path, folder_path).replace("\\", "/")
            blob = bucket.blob(f"yolo/{model_version}/{blob_path}")
            blob.upload_from_filename(file_path)
            print(f"Uploaded {file_path} to gs://{bucket_name}/yolo/{model_version}/{blob_path}")

    root_artifact_path = f"gs://{bucket_name}/yolo/{model_version}"
    model_weights_path = f"{root_artifact_path}/weights/best.pt"
    return root_artifact_path, model_weights_path
