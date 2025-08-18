import pandas as pd
import psycopg
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def calculate_simple_metrics(predictions_path, data_version):
    """Calculate basic metrics for our predictions"""
    
    # Load predictions
    predictions_df = pd.read_parquet(predictions_path)
    
    # Calculate basic stats
    metrics = {
        'timestamp': datetime.now(),
        'data_version': data_version,
        'prediction_drift': 0.1,  # Dummy for now
        'num_drifted_columns': 2,  # Dummy for now  
        'share_missing_values': predictions_df.isnull().sum().sum() / len(predictions_df),
        'num_predictions': len(predictions_df),
        'prediction_mean': predictions_df['prediction'].mean(),
        'prediction_std': predictions_df['prediction'].std()
    }
    
    # Get database credentials from environment
    db_host = "db"  # Docker service name
    db_port = "5432"
    db_name = os.getenv("POSTGRES_DB")
    db_user = os.getenv("POSTGRES_USER") 
    db_password = os.getenv("POSTGRES_PASSWORD")
    
    # Store in database
    with psycopg.connect(f"host={db_host} port={db_port} dbname={db_name} user={db_user} password={db_password}") as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO gdelt_monitoring_metrics 
                (timestamp, data_version, prediction_drift, num_drifted_columns, 
                 share_missing_values, num_predictions, prediction_mean, prediction_std)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, tuple(metrics.values()))
    
    return metrics