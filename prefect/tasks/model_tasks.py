"""
Model inference tasks for GDELT pipeline
"""

import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
from prefect import task, get_run_logger
from typing import Dict, Any
import json


@task
def get_model_predictions(
    features_path: str,
    data_version: str,
    model_service_url: str = "http://localhost:5002"
) -> str:
    """Get predictions from deployed model service"""
    logger = get_run_logger()
    
    # get project root directory
    project_root = Path(__file__).parent.parent.parent
    
    logger.info(f"Using model service at {model_service_url}")
    
    try:
        # check if model service is healthy
        health_response = requests.get(f"{model_service_url}/health", timeout=10)
        if health_response.status_code != 200:
            raise RuntimeError(f"Model service unhealthy: {health_response.status_code}")
        
        health_data = health_response.json()
        logger.info(f"Model service healthy: {health_data}")
        
        # load features (path is relative to project root)
        features_full_path = project_root / features_path
        logger.info(f"Loading features from {features_full_path}")
        X = pd.read_parquet(features_full_path)
        logger.info(f"Loaded {len(X)} rows for prediction")
        
        # get model info to validate features
        model_info_response = requests.get(f"{model_service_url}/model-info", timeout=10)
        if model_info_response.status_code == 200:
            model_info = model_info_response.json()
            expected_features = model_info["expected_features"]
            
            # validate features match model expectations
            missing_features = [f for f in expected_features if f not in X.columns]
            extra_features = [f for f in X.columns if f not in expected_features]
            
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            if extra_features:
                logger.warning(f"Extra features found (will be ignored): {extra_features}")
                
            # select only the expected features in correct order
            X = X[expected_features]
            logger.info(f"Using {len(expected_features)} features for prediction")
        else:
            logger.warning("Could not get model info, proceeding with all features")

        # prepare data for API call - convert to list of dicts
        logger.info("Preparing data for API call...")
        records = X.to_dict('records')
        
        # make batch prediction request
        logger.info(f"Making batch prediction request for {len(records)} records...")
        prediction_response = requests.post(
            f"{model_service_url}/predict",
            json=records,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if prediction_response.status_code != 200:
            raise RuntimeError(f"Prediction request failed: {prediction_response.status_code} - {prediction_response.text}")
        
        # parse response
        prediction_data = prediction_response.json()
        
        # handle both single and batch response formats
        if "predictions" in prediction_data:
            predictions = prediction_data["predictions"]
        elif "prediction" in prediction_data:
            predictions = [prediction_data["prediction"]]
        else:
            raise ValueError(f"Unexpected response format: {prediction_data}")
        
        logger.info(f"Received {len(predictions)} predictions from model service")
        
        # create predictions dataframe with metadata
        predictions_df = pd.DataFrame({
            'prediction': predictions,
            'timestamp': datetime.now(),
            'model_service_url': model_service_url,
            'data_version': data_version,
            'row_id': range(len(predictions))
        })
        
        # save predictions (relative to project root)
        predictions_dir = project_root / "monitoring" / "predictions" / data_version
        predictions_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = predictions_dir / "predictions.parquet"
        
        predictions_df.to_parquet(predictions_path, index=False)
        logger.info(f"Saved {len(predictions)} predictions to {predictions_path}")
        
        # return relative path
        return f"monitoring/predictions/{data_version}/predictions.parquet"
        
    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP request failed: {e}")
        raise RuntimeError(f"Failed to connect to model service at {model_service_url}: {e}")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


@task
def store_prediction_metadata(
    predictions_path: str,
    features_path: str,
    data_version: str
) -> Dict[str, Any]:
    """Store prediction metadata for monitoring"""
    logger = get_run_logger()
    
    # get project root directory
    project_root = Path(__file__).parent.parent.parent
    
    # load predictions and features (convert relative paths to absolute)
    predictions_full_path = project_root / predictions_path
    features_full_path = project_root / features_path
    
    predictions_df = pd.read_parquet(predictions_full_path)
    features_df = pd.read_parquet(features_full_path)
    
    # calculate summary statistics
    metadata = {
        'data_version': data_version,
        'timestamp': datetime.now().isoformat(),
        'num_predictions': len(predictions_df),
        'prediction_stats': {
            'mean': float(predictions_df['prediction'].mean()),
            'std': float(predictions_df['prediction'].std()),
            'min': float(predictions_df['prediction'].min()),
            'max': float(predictions_df['prediction'].max()),
            'median': float(predictions_df['prediction'].median())
        },
        'feature_stats': {
            'num_features': len(features_df.columns),
            'num_rows': len(features_df)
        }
    }
    
    # save metadata
    metadata_dir = project_root / "monitoring" / "predictions" / data_version
    metadata_path = metadata_dir / "metadata.json"
    
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Stored prediction metadata to {metadata_path}")
    logger.info(f"Prediction summary: {metadata['prediction_stats']}")
    
    return metadata