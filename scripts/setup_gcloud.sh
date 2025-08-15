#!/bin/bash
# scripts/setup_gcloud.sh

# Load environment variables if .env exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Use environment variable or prompt user
if [ -z "$GOOGLE_CLOUD_PROJECT" ]; then
    echo "Please set GOOGLE_CLOUD_PROJECT in your .env file or environment"
    echo "Example: GOOGLE_CLOUD_PROJECT=mlops-zoomcamp-yourname"
    exit 1
fi

echo "Setting up Google Cloud for project: $GOOGLE_CLOUD_PROJECT"

# Set up Google Cloud using $GOOGLE_CLOUD_PROJECT
gcloud auth login
gcloud projects create $GOOGLE_CLOUD_PROJECT || echo "Project may already exist"
gcloud config set project $GOOGLE_CLOUD_PROJECT
gcloud services enable bigquery.googleapis.com

# Create service account
gcloud iam service-accounts create jupyter-bigquery \
    --description="Jupyter BigQuery access" \
    --display-name="Jupyter BigQuery" || echo "Service account may already exist"

# Add BigQuery permissions to service account
echo "Adding BigQuery permissions..."
gcloud projects add-iam-policy-binding $GOOGLE_CLOUD_PROJECT \
    --member="serviceAccount:jupyter-bigquery@$GOOGLE_CLOUD_PROJECT.iam.gserviceaccount.com" \
    --role="roles/bigquery.user"

gcloud projects add-iam-policy-binding $GOOGLE_CLOUD_PROJECT \
    --member="serviceAccount:jupyter-bigquery@$GOOGLE_CLOUD_PROJECT.iam.gserviceaccount.com" \
    --role="roles/bigquery.jobUser"

# Create credentials file
gcloud iam service-accounts keys create ./bigquery-credentials.json \
    --iam-account=jupyter-bigquery@$GOOGLE_CLOUD_PROJECT.iam.gserviceaccount.com

echo "Setup complete!"
echo "BigQuery permissions added successfully"
