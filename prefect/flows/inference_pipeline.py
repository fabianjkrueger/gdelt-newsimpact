"""
GDELT Daily Inference Pipeline

This flow orchestrates the daily inference process:
1. Download fresh GDELT data for the target date
2. Prepare data using existing encoder from MLflow
3. Generate predictions using deployed model
4. Store results for monitoring
"""

from datetime import datetime, timedelta
from prefect import flow, get_run_logger
from typing import Optional
import sys
from pathlib import Path

# add the prefect directory to Python path so we can import our tasks
sys.path.append(str(Path(__file__).parent.parent))

from tasks.data_tasks import download_fresh_gdelt_data, prepare_inference_data
from tasks.model_tasks import get_model_predictions, store_prediction_metadata


@flow(name="gdelt-daily-inference", log_prints=True)
def daily_inference_pipeline(
    target_date: Optional[str] = None,
    limit: int = 1000
) -> dict:
    """
    Run daily inference pipeline for GDELT data
    
    Args:
        target_date: Target date in YYYY-MM-DD format (defaults to yesterday)
        limit: Number of rows to process
        
    Returns:
        dict: Pipeline execution summary
    """
    logger = get_run_logger()
    
    # default to yesterday if no date provided
    if target_date is None:
        target_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    data_version = f"inference_{target_date.replace('-', '_')}"
    
    logger.info(f"Starting daily inference pipeline for {target_date}")
    logger.info(f"Data version: {data_version}")
    
    try:
        # step 1: download fresh GDELT data
        logger.info("Step 1: Downloading fresh GDELT data...")
        raw_data_path = download_fresh_gdelt_data(
            target_date=target_date,
            limit=limit,
            data_version=data_version
        )
        
        # step 2: prepare data for inference
        logger.info("Step 2: Preparing data for inference...")
        features_path = prepare_inference_data(
            raw_data_path=raw_data_path,
            data_version=data_version
        )
        
        # step 3: generate predictions
        logger.info("Step 3: Generating predictions...")
        predictions_path = get_model_predictions(
            features_path=features_path,
            data_version=data_version
        )
        
        # step 4: store metadata for monitoring
        logger.info("Step 4: Storing prediction metadata...")
        metadata = store_prediction_metadata(
            predictions_path=predictions_path,
            features_path=features_path,
            data_version=data_version
        )
        
        # summary
        summary = {
            'status': 'success',
            'target_date': target_date,
            'data_version': data_version,
            'files': {
                'raw_data': raw_data_path,
                'features': features_path,
                'predictions': predictions_path
            },
            'metadata': metadata
        }
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Processed {metadata['num_predictions']} predictions")
        
        return summary
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    # test run
    result = daily_inference_pipeline()
    print(f"Pipeline result: {result}")