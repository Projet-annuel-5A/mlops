# FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

RUN pip install fastapi uvicorn numpy opencv-python-headless ultralytics pillow google-cloud-storage google-cloud-aiplatform

COPY application_default_credentials.json /root/.config/gcloud/application_default_credentials.json
ENV GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json
ENV GOOGLE_CLOUD_PROJECT=annual-project-427112


COPY app.py /workspace/app.py

COPY utils/ /workspace/utils/

WORKDIR /workspace

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
