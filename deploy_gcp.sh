#!/bin/bash
set -Ceuo pipefail

source .env

gcloud config set builds/use_kaniko False
gcloud builds submit \
	--tag "gcr.io/${PROJECT_ID}/${CONTAINER_NAME}" \
	--project "${PROJECT_ID}"
gcloud run deploy "${CONTAINER_NAME}"\
	 --image "gcr.io/${PROJECT_ID}/${CONTAINER_NAME}" \
	  --platform managed \
	  --region asia-east1 \
	  --memory 2Gi
