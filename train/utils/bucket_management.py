from google.cloud import storage
import os

def download_bucket(bucket_name, destination_folder):
    """
    Download an entire bucket.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs()

    for blob in blobs:
        destination_file_name = os.path.join(destination_folder, blob.name)
        
        if not os.path.exists(os.path.dirname(destination_file_name)):
            os.makedirs(os.path.dirname(destination_file_name))
        
        if blob.name.endswith('/'):
            continue
        
        blob.download_to_filename(destination_file_name)
        print(f"Downloaded {blob.name} to {destination_file_name}")


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
