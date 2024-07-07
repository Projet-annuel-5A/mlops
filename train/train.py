from utils.model_management import get_model_registery, get_model_versions, deploy_model, redeploy_endpoint
from utils.bucket_management import download_bucket, upload_folder_to_bucket
from utils.train import train
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
artifact_uri = f"gs://interviewz-models/yolo/{new_model_version}"
serving_container_image_uri = "gcr.io/annual-project-427112/model:latest"


# IMPORT DATASET
download_bucket(dataset_bucket_name, dataset_path, workers=4)


# MODEL TRAINING
metric = train(device=device, epoch=100)

# SEND RESULTS ARTIFACTS TO GCS
root_artifact_path, model_weights_path = upload_folder_to_bucket(training_results_path, output_bucket_name, new_model_version)

# TRIGGER NEW MODEL DEPLOYEMENT
model_promoted = deploy_model(
    model_registry=model_registery,
    model_id=model_id,
    trained_model_metric=metric,
    artifact_uri = artifact_uri,
    serving_container_image_uri=serving_container_image_uri
)

if model_promoted:
    # UPDATE ENDPOINT
    print("Model promoted, redploying endpoint")
    redeploy_endpoint(model_id=model_id,endpoint_display_name="yolo_predict")
else :
    print("Skipping promotion since model not promotted to production")