from utils.data_models import ImageRequest, PredictionRequest, PredictionResponse
from utils.context_manager import lifespan, ml_models
from fastapi import FastAPI, HTTPException
from utils.results import format_results
from utils.image import transform_image
from utils.logger import logger

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # IMAGE PROCESSING
        image_data = request.instances[0].image
        image = transform_image(b64_img=image_data)

        # INFERENCE
        model = ml_models['yolo']
        results = model.predict(image)
        predictions = format_results(results=results)

        return {"predictions": [predictions]}

    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

