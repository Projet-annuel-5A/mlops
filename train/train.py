from utils.model_management import get_model_registery, get_model_versions, deploy_model
from utils.bucket_management import download_bucket, upload_folder_to_bucket
from google.cloud import aiplatform
from ultralytics import YOLO
import torch

# STARTING
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f"Starting training with device: {device}")


# DFINE METADATA
model_id = "7605563821884702720"
model_registery = get_model_registery(model_id=model_id)
last_model_version, new_model_version = get_model_versions(model_registery=model_registery)
dataset_bucket_name = "interviewz-training-data"
output_bucket_name = "interviewz-models"
dataset_path = "./dataset"
training_results_path = "./runs/classify/train"
model_path = f"{training_results_path}/weights/best.pt"


# IMPORT DATASET
download_bucket(dataset_bucket_name, dataset_path)


# MODEL TRAINING
model = YOLO('yolov8x-cls.pt')
results = model.train(data='./dataset', epochs=100, imgsz=48, device=device)
metric = results.top1

# SEND RESULTS ARTIFACTS TO GCS
root_artifact_path, model_weights_path = upload_folder_to_bucket(training_results_path, output_bucket_name, new_model_version)

# TRIGGER NEW MODEL DEPLOYEMENT
deploy_model(
    model_registry=model_registery,
    model_id=model_id,
    trained_model_metric=73,
    artifact_uri = f"gs://interviewz-models/yolo/{new_model_version}"
)

# UPDATE ENDPOINT
