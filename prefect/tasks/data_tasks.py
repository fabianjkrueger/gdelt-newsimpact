"""
Data processing tasks for GDELT inference pipeline
"""

import subprocess
import os
from datetime import datetime, timedelta
from pathlib import Path
from prefect import task, get_run_logger
from typing import Optional


@task(retries=3, retry_delay_seconds=60)
def download_fresh_gdelt_data(
    target_date: Optional[str] = None,
    limit: int = 1000,
    data_version: Optional[str] = None
) -> str:
    """Download fresh GDELT data for target date"""
    logger = get_run_logger()
    
    # get project root directory (two levels up from prefect/tasks/)
    project_root = Path(__file__).parent.parent.parent
    
    # default to yesterday if no date provided
    if target_date is None:
        target_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    # default data version based on date
    if data_version is None:
        data_version = f"inference_{target_date.replace('-', '_')}"
    
    logger.info(f"Downloading GDELT data for {target_date} with version {data_version}")
    logger.info(f"Working directory: {project_root}")
    
    # prepare command
    cmd = [
        "uv", "run", "python", "scripts/download_data_with_BigQuery.py",
        "--start_date", target_date,
        "--end_date", target_date,
        "--limit", str(limit),
        "--version_name", data_version,
        "--dry_run", "False"
    ]
    
    # run the download script from project root
    try:
        result = subprocess.run(
            cmd, 
            cwd=project_root,  # Set working directory
            check=True, 
            capture_output=True, 
            text=True
        )
        logger.info("Download completed successfully")
        logger.info(f"Script output: {result.stdout}")
        
        # return expected file path (relative to project root)
        return f"data/raw/gdelt_events_{data_version}_full.parquet"
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Download failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        raise


@task
def prepare_inference_data(
    raw_data_path: str,
    data_version: str
) -> str:
    """Prepare raw data for inference using existing encoder"""
    logger = get_run_logger()
    
    # get project root directory
    project_root = Path(__file__).parent.parent.parent
    
    logger.info(f"Preparing inference data from {raw_data_path}")
    logger.info(f"Working directory: {project_root}")
    
    # prepare command for query mode (inference)
    cmd = [
        "uv", "run", "python", "scripts/prepare_data_for_modeling.py",
        "--query",
        "--data-version", data_version,
        "--query-data-file", raw_data_path
    ]
    
    try:
        result = subprocess.run(
            cmd, 
            cwd=project_root,  # Set working directory
            check=True, 
            capture_output=True, 
            text=True
        )
        logger.info("Data preparation completed successfully")
        logger.info(f"Script output: {result.stdout}")
        
        # return expected features file path
        return f"data/intermediate/{data_version}/X_query.parquet"
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Data preparation failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        raise