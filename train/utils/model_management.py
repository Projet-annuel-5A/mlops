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

def upload_model(
        model_id: str,
        to_production: bool,
        metric: int,
        artifact_uri: str,
        serving_container_image_uri: str
    ):
    """
    Upload a new model in the container registry with the serving image
    """

    model = aiplatform.Model(model_name=model_id)
    print(f"Uploading model {model_id} ...")
    model.upload(
        parent_model=model_id,
        is_default_version=to_production,
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_container_image_uri,
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
        serving_container_image_uri:str,
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
            upload_model(model_id=model_id, to_production=promotion, metric=trained_model_metric,artifact_uri=artifact_uri, serving_container_image_uri=serving_container_image_uri)
        break

    return promotion
    


def redeploy_endpoint(model_id:str = "7605563821884702720", endpoint_display_name: str = "yolo_predict"):

    endpoint_found = False
    for endpoint in aiplatform.Endpoint.list():
        if endpoint.display_name == endpoint_display_name:
            print(f"Found endpoint with display name :{endpoint_display_name}")
            endpoint_found = True
            current_endpoint = aiplatform.Endpoint(endpoint_name=endpoint.name)

            current_endpoint.undeploy_all(sync=True)

            current_endpoint.deploy(
                model=aiplatform.Model(model_id),
                traffic_percentage=100,
                machine_type="n1-standard-4",
                accelerator_type="NVIDIA_TESLA_T4",
                accelerator_count=1,
                min_replica_count=1,
                max_replica_count=1,
                sync=False
            )
    
    if not endpoint_found:
        print(f"No endpoint found with display name: {endpoint_display_name}. Creating a new endpoint.")
        
        new_endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_display_name
        )

        new_endpoint.deploy(
            model=aiplatform.Model(model_id),
            traffic_percentage=100,
            machine_type="n1-standard-4",
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1,
            min_replica_count=1,
            max_replica_count=1,
            sync=True
        )

        print(f"New endpoint created and model deployed with display name: {endpoint_display_name}")




