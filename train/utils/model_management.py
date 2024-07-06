from google.cloud import aiplatform

def get_model_registery(project:str = "annual-project-427112", location:str = "europe-west1", model_id:str = "7605563821884702720"):
    """
    Return the vertexai model registry instance of the yolo model
    """
    aiplatform.init(project=project, location=location)
    return aiplatform.ModelRegistry(model=model_id)

def get_model_versions(model_registery: aiplatform.ModelRegistry):
    """
    Return the newest model version number and future model version number of the newly created model
    """
    versions_nb = []
    for model_version in model_registery.list_versions():
        versions_nb.append(int(model_version.version_id))

    last_model_version = max(versions_nb)
    new_model_version = last_model_version + 1
    return last_model_version, new_model_version

def upload_model(model_id: str, to_production: bool, metric: int, artifact_uri: str):
    model = aiplatform.Model(model_name=model_id)
    print(f"Uploading model {model_id} ...")
    model.upload(
        parent_model=model_id,
        is_default_version=to_production,
        artifact_uri=artifact_uri,
        serving_container_image_uri="gcr.io/annual-project-427112/model@sha256:d08ab2c77bb7b0dd9da33af5ce5bfd6a868e1c90d751f70a7d1c2d295ccea04a",
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        labels={"metric": str(metric)}
    )
    model.wait()

def deploy_model(
        model_registry: aiplatform.ModelRegistry,
        model_id: str,
        trained_model_metric: int,
        artifact_uri:str,
        production_alias:str = "default"
    ):
    """
    Create a new version of the trained model with a promotion to production or not
    """
    production_model_metric = 0

    for model_version in model_registry.list_versions():
        if production_alias in model_version.version_aliases:
            production_model_metric = int(aiplatform.Model("7605563821884702720",version=model_version.version_id).labels["metric"])
            print(f"Current production model metric is {production_model_metric}")
            if trained_model_metric > production_model_metric:
                print(f"New trained model with metric: {trained_model_metric} is better than current production model with metric: {production_model_metric}")
                print(f"Creating model version with promotion to production with alias: {production_alias} ...")
                promotion = True
            else :
                promotion = False
                print("New trained model is not better than current production model. Skipping model promotion ...")
            upload_model(model_id=model_id, to_production=promotion, metric=trained_model_metric,artifact_uri=artifact_uri)
    
