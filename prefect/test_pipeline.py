"""
Test script for the inference pipeline
"""

import os
import sys
from pathlib import Path

# add prefect directory to path
sys.path.append(str(Path(__file__).parent))

from flows.inference_pipeline import daily_inference_pipeline


def test_pipeline():
    """Test the pipeline with a small sample"""
    print("🧪 Testing GDELT inference pipeline...")
    
    # run with small limit for testing
    result = daily_inference_pipeline(
        target_date="2025-03-13",  # use a known date
        limit=100  # small sample for testing
    )
    
    print(f"✅ Test completed!")
    print(f"Result: {result}")
    return result


if __name__ == "__main__":
    test_pipeline()