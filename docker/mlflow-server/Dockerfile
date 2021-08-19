
# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.9-slim

# Install mlflow.

RUN pip install mlflow[extras] google-cloud-storage

# Create shared directories

RUN mkdir /database
RUN mkdir /gcloud

# Server needs a service account key file to access buckets.
ENV GOOGLE_APPLICATION_CREDENTIALS /gcloud/service_account_key.json

# Run the server
ENV MLFLOW_SERVER_HOST=0.0.0.0
ENV MLFLOW_SERVER_PORT=5000
ENV MLFLOW_SERVER_DEFAULT_ARTIFACT_ROOT="gs://sentience-mlflow"
ENV MLFLOW_SERVER_FILE_STORE="sqlite:////database/mlflow.sqlite"

CMD ["/bin/bash", "-c", "mlflow server --backend-store-uri $MLFLOW_SERVER_FILE_STORE --default-artifact-root $MLFLOW_SERVER_DEFAULT_ARTIFACT_ROOT --host $MLFLOW_SERVER_HOST --port $MLFLOW_SERVER_PORT"]

#ENTRYPOINT ["mlflow", "server", "--host", $MLFLOW_SERVER_HOST, "--port", $MLFLOW_SERVER_PORT]