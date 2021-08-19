#!/bin/bash
DATABASE_DIR=$1
SERVICE_ACCOUNT_KEY_DIR=$2
docker run -d -p 5000:5000 --mount type=bind,source=$DATABASE_DIR,target=/database --mount type=bind,source=$SERVICE_ACCOUNT_KEY_DIR,target=/gcloud mlflow-server