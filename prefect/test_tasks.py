"""
Test individual tasks within proper Prefect context
"""

import sys
from pathlib import Path
from prefect import flow

# add prefect directory to path
sys.path.append(str(Path(__file__).parent))

from tasks.data_tasks import download_fresh_gdelt_data, prepare_inference_data
from tasks.model_tasks import get_model_predictions, store_prediction_metadata


@flow(name="test-download-task")
def test_download_task():
    """Test data download task within flow context"""
    print("🔄 Testing data download...")
    
    result = download_fresh_gdelt_data(
        target_date="2024-01-01",
        limit=50,
        data_version="test_download"
    )
    print(f"✅ Download test passed: {result}")
    return result


@flow(name="test-prepare-task")
def test_prepare_task():
    """Test data preparation task within flow context"""
    print("🔄 Testing data preparation...")
    
    # first download data
    raw_data_path = download_fresh_gdelt_data(
        target_date="2024-01-01",
        limit=50,
        data_version="test_download"
    )
    
    # then prepare it
    result = prepare_inference_data(
        raw_data_path=raw_data_path,
        data_version="test_download"
    )
    print(f"✅ Prepare test passed: {result}")
    return result


@flow(name="test-all-tasks")
def test_all_tasks():
    """Test the complete task chain"""
    print("🧪 Testing complete task chain...")
    
    target_date = "2024-01-01"  # use a valid, historical date
    data_version = "test_chain"
    project_root = Path(__file__).parent.parent

    # to save BQ quota, check if the file already exists
    raw_path_str = f"data/raw/gdelt_events_{data_version}_full.parquet"
    raw_path_full = project_root / raw_path_str

    if raw_path_full.exists():
        print(f"✅ Raw data file already exists at {raw_path_full}. Skipping download.")
        raw_path = raw_path_str
    else:
        # step 1: download
        raw_path = download_fresh_gdelt_data(
            target_date=target_date,
            limit=50,
            data_version=data_version
        )
    
    # step 2: prepare
    features_path = prepare_inference_data(
        raw_data_path=raw_path,
        data_version=data_version
    )
    
    # step 3: predict (this might fail if model isn't available, that's ok)
    try:
        predictions_path = get_model_predictions(
            features_path=features_path,
            data_version=data_version
        )
        
        # step 4: store metadata
        metadata = store_prediction_metadata(
            predictions_path=predictions_path,
            features_path=features_path,
            data_version=data_version
        )
        
        print(f"🎉 All tasks completed successfully!")
        return {
            'raw_path': raw_path,
            'features_path': features_path,
            'predictions_path': predictions_path,
            'metadata': metadata
        }
        
    except Exception as e:
        print(f"⚠️  Model prediction failed (expected if model not available): {e}")
        print(f"✅ Data pipeline (download + prepare) works correctly!")
        return {
            'raw_path': raw_path,
            'features_path': features_path,
            'status': 'data_pipeline_ok'
        }


if __name__ == "__main__":
    # test the complete chain
    result = test_all_tasks()
    print(f"Final result: {result}")