# FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

RUN pip install numpy ultralytics pillow google-cloud-storage google-cloud-aiplatform

COPY application_default_credentials.json /root/.config/gcloud/application_default_credentials.json
ENV GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json
ENV GOOGLE_CLOUD_PROJECT=annual-project-427112

COPY train.py /workspace/train.py

COPY utils/ /workspace/utils/

WORKDIR /workspace

CMD ["python", "train.py"]
