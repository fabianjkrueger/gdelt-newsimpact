"""
This script downloads a subset of the GDELT 2.0 Event Database
using Google BigQuery.
The full dataset is way too large, so this only downloads a subset of it.
This is the initial data set for development and testing.

There are two subsets:
- 1k rows for rapid prototyping and interactive use in notebooks
- 10k rows for actual training and testing in background jobs
    - This set will be split into 80:20 train:test
    - The models will be developed using 5-fold cross validation
    
Both subsets are saved as parquet files.

Usage:
- Create BigQuery credentials file by following the instructions in the README
- Run the script in the project's uv environment with the following command:
- This works from any subdirectory of the project, as long as you specify the
    path to the script correctly, because paths for saving the data are managed
    relative to the project root
- If you want to reproduce my results, use the default settings.
```
# test it with a dry run first
uv run python download_data_with_BigQuery.py --dry_run True

# default settings
uv run python scripts/download_data_with_BigQuery.py

# custom settings (example, but is the same as the default settings)
uv run python scripts/download_data_with_BigQuery.py \
    --start_date 2024-01-01 \
    --end_date 2024-12-31 \
    --limit 10000 \
    --dry_run False \
    --version_name 2024_subset_10k \
    --seed 1337 \
    --test_size 0.2
```
"""

# Setup
# -----
# Dependencies
import os
import pandas_gbq
import pandas as pd
import click

from pathlib import Path
from dotenv import load_dotenv
from google.cloud import bigquery
from sklearn.model_selection import train_test_split

# Load environment variables
load_dotenv()

# Paths
PATH_REPO = Path(__file__).resolve().parent.parent
PATH_DATA = PATH_REPO / "data" / "raw"
PATH_DATA.mkdir(parents=True, exist_ok=True)

# Functions
# ---------
def setup_bigquery_client():
    """
    Set up BigQuery client using credentials file.
    
    Returns:
        bigquery.Client: BigQuery client
        str: Path to credentials file
    """
    # Check if credentials file exists
    cred_path = str(PATH_REPO / "bigquery-credentials.json")
    if not Path(cred_path).exists():
        raise FileNotFoundError(f"Credentials file not found: {cred_path}")
    
    # Set environment variable for this session
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = cred_path
    
    # Get project ID from environment
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT not set in .env file")
    
    # Initialize client
    client = bigquery.Client(project=project_id)
    return client, cred_path

def safe_gdelt_query(
    start_date,
    end_date,
    limit=10000,
    dry_run=True,
    client=None,
    seed=42  # Add seed parameter for reproducibility
):
    """
    Safely query GDELT data with automatic cost estimation
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format  
        limit (int): Maximum number of rows to return
        dry_run (bool): If True, only estimate query cost
        client (bigquery.Client): BigQuery client
        seed (int): Random seed for reproducible sampling
    Returns:
        pd.DataFrame: Dataframe with GDELT data
    """

    if client is None:
        raise ValueError("BigQuery client not initialized")
    
    # Convert dates to GDELT format (YYYYMMDD) as integers
    start_gdelt = int(start_date.replace('-', ''))
    end_gdelt = int(end_date.replace('-', ''))

    query = f"""
    SELECT 
        SQLDATE,                -- event date
        MonthYear,              -- month and year
        EventCode,
        EventBaseCode,
        EventRootCode,
        QuadClass,
        GoldsteinScale,
        Actor1Code,
        Actor1Name,
        Actor1CountryCode,
        ActionGeo_CountryCode,
        ActionGeo_Lat,
        ActionGeo_Long,
        NumArticles             -- target variable
    FROM `gdelt-bq.gdeltv2.events`
    WHERE SQLDATE >= {start_gdelt}  -- start date
      AND SQLDATE <= {end_gdelt}    -- end date
    ORDER BY 
        -- Create deterministic "random" order using hash of multiple columns
        -- This ensures same seed always gives same data, but data appears random
        MOD(
            ABS(FARM_FINGERPRINT(
                CONCAT(
                    CAST(SQLDATE AS STRING),
                    CAST(EventCode AS STRING),
                    CAST({seed} AS STRING)  -- Include seed in hash
                )
            )), 
            1000000
        )
    LIMIT {limit}                   -- limit the number of rows to return
    """

    # Always do a dry run first for cost estimation
    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    dry_job = client.query(query, job_config=job_config)

    bytes_processed = dry_job.total_bytes_processed
    estimated_cost = (bytes_processed / 1e12) * 5  # $5 per TB

    print(
        f"Query will process: {bytes_processed:,} bytes "
        f"({bytes_processed/1e6:.2f} MB) or rather "
        f"({bytes_processed/1e9:.2f} GB)."
    )
    # Disclaimer: I don't know how exact the cost estimation really is
    print(f"Estimated cost: ${estimated_cost:.6f}")

    if dry_run:
        print("Dry run complete - no data retrieved")
        return None

    # Execute the actual query
    print("Executing query...")
    df = pandas_gbq.read_gbq(query, project_id=client.project, dialect='standard')

    print(f"Query completed! Retrieved {len(df)} rows")
    return df

def save_data_to_parquet(
    data_full: pd.DataFrame,
    data_train: pd.DataFrame,
    data_test: pd.DataFrame,
    version_name: str
):
    """
    Save data to parquet files.
    This always saves the full data, the train data and the test data.
    Each split gets the appropriate suffix.
    User must specify a version name.
    
    Args:
        data_full (pd.DataFrame): Full data
        data_train (pd.DataFrame): Train data
        data_test (pd.DataFrame): Test data
        version_name (str): Version name. E.g. "2024_subset_10k"
    """
    
    path_full = PATH_DATA / f"gdelt_events_{version_name}_full.parquet"
    path_train = PATH_DATA / f"gdelt_events_{version_name}_train.parquet"
    path_test = PATH_DATA / f"gdelt_events_{version_name}_test.parquet"
    
    data_full.to_parquet(path_full, index=False)
    data_train.to_parquet(path_train, index=False)
    data_test.to_parquet(path_test, index=False)
    
    print(f"Data saved to:\n{path_full}\n{path_train}\n{path_test}")

@click.command()
@click.option(
    "--start_date",
    type=str,
    default="2024-01-01",
    help="Start date in 'YYYY-MM-DD' format"
)
@click.option(
    "--end_date",
    type=str,
    default="2024-12-31",
    help="End date in 'YYYY-MM-DD' format"
)
@click.option(
    "--limit",
    type=int,
    default=10000,
    help="Number of rows to return"
)
@click.option(
    "--dry_run",
    type=bool,
    default=False,
    help="If True, only estimate query cost"
)
@click.option(
    "--version_name",
    type=str,
    default="2024_subset_10k",
    help="Version name. E.g. '2024_subset_10k'"
)
@click.option(
    "--seed",
    type=int,
    default=1337,
    help=(
        "Random seed for reproducible sampling. "
        "Don't change this if you want to exactly reproduce my results!"
    )
)
@click.option(
    "--test_size",
    type=float,
    default=0.2,
    help="Test size. E.g. 0.2 for 20% test size"
)
def main(
    start_date,
    end_date,
    limit,
    dry_run,
    version_name,
    seed,
    test_size
):
    """
    Main function to run all steps.
    This initializes the BigQuery client, queries the data,
    splits it into train and test, and saves it to parquet files.
    """
    # Get data from Google BigQuery
    # -----------------------------

    # Initialize BigQuery client
    try:
        client, cred_path = setup_bigquery_client()
        print(f"BigQuery client initialized successfully!")
        print(f"Project: {client.project}")
        print(f"Using credentials from: {cred_path}")
    except Exception as e:
        print(f"Error setting up BigQuery client: {e}")
        return  # Exit early if client setup fails

    # Query the data (only runs if client setup succeeded)
    data_gdelt = safe_gdelt_query(
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        dry_run=dry_run,
        client=client,
        seed=seed
    )

    # Don't continue if dry_run or no data
    if dry_run or data_gdelt is None:
        return

    # Split data into train and test
    train_df, test_df = train_test_split(
        data_gdelt,
        test_size=test_size,
        random_state=seed
    )

    # Save data to parquet files
    save_data_to_parquet(
        data_full=data_gdelt,
        data_train=train_df,
        data_test=test_df,
        version_name=version_name
    )


# Main
# ----
if __name__ == "__main__":
    main()
