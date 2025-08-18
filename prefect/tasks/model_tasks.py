"""
Model inference tasks for GDELT pipeline
"""

import pandas as pd
import mlflow
import mlflow.xgboost
from pathlib import Path
from datetime import datetime
from prefect import task, get_run_logger
from typing import Dict, Any


@task
def get_model_predictions(
    features_path: str,
    data_version: str,
    model_name: str = "gdelt-event-classifier",
    model_alias: str = "cyber_dragon"
) -> str:
    """Get predictions from MLflow model"""
    logger = get_run_logger()
    
    # get project root directory
    project_root = Path(__file__).parent.parent.parent
    
    # set MLflow tracking URI for local development
    mlflow.set_tracking_uri("http://127.0.0.1:5001")
    
    logger.info(f"Loading model {model_name}@{model_alias}")
    logger.info(f"MLflow URI: http://127.0.0.1:5001")
    
    try:
        # load model from registry
        model_uri = f"models:/{model_name}@{model_alias}"
        model = mlflow.xgboost.load_model(model_uri)
        logger.info("Model loaded successfully")
        
        # load features (path is relative to project root)
        features_full_path = project_root / features_path
        logger.info(f"Loading features from {features_full_path}")
        X = pd.read_parquet(features_full_path)
        logger.info(f"Loaded {len(X)} rows for prediction")
        
        # validate features match model expectations
        expected_features = [
            "QuadClass", "GoldsteinScale", "ActionGeo_Lat", "ActionGeo_Long",
            "EventCode", "EventBaseCode", "EventRootCode", "Actor1Code", 
            "Actor1Name", "Actor1CountryCode", "ActionGeo_CountryCode",
            "year", "month", "day_of_year", "day_of_week", "is_weekend"
        ]
        
        missing_features = [f for f in expected_features if f not in X.columns]
        extra_features = [f for f in X.columns if f not in expected_features]
        
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        if extra_features:
            logger.warning(f"Extra features found (will be ignored): {extra_features}")
            
        # select only the expected features in correct order
        X = X[expected_features]
        logger.info(f"Using {len(expected_features)} features for prediction")

        # make predictions
        logger.info("Generating predictions...")
        predictions = model.predict(X)
        
        # create predictions dataframe with metadata
        predictions_df = pd.DataFrame({
            'prediction': predictions,
            'timestamp': datetime.now(),
            'model_name': model_name,
            'model_alias': model_alias,
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