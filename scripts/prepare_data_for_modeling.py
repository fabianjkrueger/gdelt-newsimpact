"""
Data Preparation Script for GDELT News Impact Modeling

This script prepares the GDELT event data downloaded from BigQuery for machine learning modeling.
It handles data preprocessing, feature engineering, categorical encoding, and saves the prepared
data in a structured format for training and inference.

OVERVIEW:
---------
The script can operate in three modes:
1. TRAIN MODE: Prepares training data and creates/saves the categorical encoder to MLflow
2. QUERY MODE: Prepares new data for inference using a specific hardcoded encoder  
3. SUBSET MODE: Creates smaller subsets of training data for rapid prototyping

DATA FLOW:
----------
Raw Data (parquet) → Data Cleaning → Feature Engineering → Categorical Encoding → Processed Data (parquet)
                                                                ↓
                                                        Encoder → MLflow Artifact Store

DIRECTORY STRUCTURE:
-------------------
data/
├── raw/                                    # Input data location
│   ├── gdelt_events_2024_subset_10k_train.parquet
│   ├── gdelt_events_2024_subset_10k_test.parquet
│   └── gdelt_events_<version>_*.parquet
└── intermediate/                           # Output data location
    ├── <data_version>/                     # Main processed data
    │   ├── X_train.parquet                 # Training features
    │   ├── y_train.parquet                 # Training labels
    │   ├── X_query.parquet                 # Query/test features
    │   └── y_query.parquet                 # Query/test labels
    └── subset_<size>_<version>/            # Subset data
        ├── X_train.parquet                 # Subset features
        └── y_train.parquet                 # Subset labels

PARAMETERS:
-----------
--train                 : Prepare training data and fit encoder
--query                 : Prepare query/test data using existing encoder
--subset                : Create subset of training data (requires --train)
--subset-size <int>     : Size of subset (default: 1000)
--data-version <str>    : Version name for output data (default: "2024_subset_10k")
--source-data-version <str> : Source version to load from (defaults to data-version)
--query-data-file <path>    : Custom query data file path

KEY BEHAVIORS:
--------------
• Data Version Logic:
    - For TRAIN mode: Loads from gdelt_events_{load_version}_train.parquet
    - For QUERY mode: Loads from gdelt_events_{load_version}_test.parquet
    - Output always saved to data/intermediate/{data_version}/
    - If source-data-version not specified, uses data-version for both loading and saving

• Encoder Management:
    - TRAIN mode: Creates new encoder and saves to MLflow artifact store
    - QUERY mode: Uses HARDCODED encoder from run ID "58eccb619a3b45359b1e9bcd5b1c9a6d"
    - This ensures consistent categorical transformations and allows the script to run
      independently without requiring a previous training run
    - The hardcoded encoder is always used to prevent conflicts from other experiments

• MLflow Integration:
    - Connects to MLflow server at http://127.0.0.1:5001
    - Uses "data_preparation" experiment (FIXME: make configurable)
    - Logs encoder artifacts for reproducible preprocessing
    - QUERY mode always uses the specific encoder from run "58eccb619a3b45359b1e9bcd5b1c9a6d"

PREPROCESSING STEPS:
-------------------
1. Drop columns with >50% missing values
2. Impute missing values:
    - Categorical columns: "UNKNOWN"
    - Numerical columns: 999 (out-of-range value)
3. Date feature engineering from SQLDATE:
    - Extract year, month, day_of_year, day_of_week, is_weekend
4. Categorical encoding using OrdinalEncoder:
    - Maps categories to integers
    - Handles unknown categories with -1

USAGE EXAMPLES:
---------------
# First-time setup: prepare all data
uv run python scripts/prepare_data_for_modeling.py --train --query --subset

# Training workflow: prepare training data with custom version
uv run python scripts/prepare_data_for_modeling.py --train --data-version "experiment_v1"

# Inference workflow: prepare new data using same version
uv run python scripts/prepare_data_for_modeling.py --query --data-version "experiment_v1"
# → Loads: gdelt_events_experiment_v1_test.parquet
# → Saves: data/intermediate/experiment_v1/X_query.parquet
# → Uses: encoder from hardcoded run "58eccb619a3b45359b1e9bcd5b1c9a6d"

# Cross-version inference: use different source and output versions
uv run python scripts/prepare_data_for_modeling.py --query \
    --data-version "new_experiment" \
    --source-data-version "2024_subset_10k"
# → Loads from: gdelt_events_2024_subset_10k_test.parquet  
# → Saves: data/intermediate/new_experiment/X_query.parquet
# → Uses: encoder from hardcoded run "58eccb619a3b45359b1e9bcd5b1c9a6d"

# Custom data inference: prepare completely custom dataset
uv run python scripts/prepare_data_for_modeling.py --query \
    --data-version "custom_analysis" \
    --query-data-file "/path/to/new_data.parquet"

# Prototyping workflow: create small subset for development
uv run python scripts/prepare_data_for_modeling.py --train --subset \
    --subset-size 500 --data-version "prototype"
# → Creates: data/intermediate/subset_500_prototype/

# Production workflow: prepare large datasets
uv run python scripts/prepare_data_for_modeling.py --train \
    --data-version "production_v1"

REQUIREMENTS:
-------------
• MLflow server running at http://127.0.0.1:5001
• Source data files must exist in data/raw/ directory
• For QUERY mode: The hardcoded encoder run "58eccb619a3b45359b1e9bcd5b1c9a6d" must exist in MLflow
• For SUBSET mode: Must be combined with --train flag

ERROR HANDLING:
---------------
• Missing source files: Lists available files in data/raw/
• Missing hardcoded encoder: Clear error message about the specific required run ID
• Invalid combinations: Clear error messages for incompatible flag combinations

OUTPUTS:
--------
• Processed datasets saved as parquet files
• Encoder artifacts logged to MLflow
• Console output showing data shapes and file locations
• MLflow run IDs for tracking experiments

The subset will be saved to a different directory.
The dir name is subset_<subset_size>_<data_version>.

So basically, if you want to prepare data for a new experiment called "my_experiment":
1. Run with --train --data-version "my_experiment" to create the encoder
2. Run with --query --data-version "my_experiment" to prepare test data
3. Query mode will always use the same hardcoded encoder for consistency across all experiments

For cross-experiment analysis, you can mix and match versions using --source-data-version
to load from one experiment and save to another.
"""

# Setup
# -----

# Dependencies
import os
import pandas as pd
import mlflow
import joblib
import click

from pathlib import Path
from typing import Optional, Tuple
from sklearn.preprocessing import OrdinalEncoder

# Functions
# ---------

# helper function for saving the labels
def save_series_as_parquet(series, filepath):
    """Save a pandas Series as parquet file"""
    # Convert Series to DataFrame with proper column name
    df = series.to_frame(name=series.name if series.name else 'target')
    df.to_parquet(filepath)

# helper function for getting the hardcoded encoder run ID
def get_hardcoded_encoder_run_id():
    """Get the hardcoded run ID that contains the standard encoder artifact"""
    # hardcoded run ID to ensure consistency across all experiments
    # this prevents issues when running query mode after other experiments
    return "5664a4bf4d274268bdcf7694267abbcb"

# function for data preparation
def prepare_data(
    df: pd.DataFrame,
    is_train: bool,
    save_data: bool = False,
    save_path_dir: Optional[str] = None,
    path_repo: Optional[str] = None,
    use_active_run: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for training or testing.
    
    Args:
        df: The dataframe to prepare.
        is_train: If True, prepares data for training. If False, prepares data
        for testing.
        save_data: If True, saves the data to the artifact store.
        save_path_dir: If provided, saves the data to this directory.
        path_repo: If provided, uses this path to save the data.
        use_active_run: If True, uses the currently active MLflow run instead of
        creating a new one.

    Returns:
        X: The features.
        y: The labels.
        (encoder): The encoder (only returned for training mode).

    Raises:
        ValueError: If hardcoded encoder cannot be loaded for inference mode.
    """
    
    # Constants
    target_label = "NumArticles"
    
    # path to intermediate data
    if save_data == True:
        PATH_DATA = Path(path_repo) / "data/intermediate"

    # schema definition for consistency
    expected_columns = [
        # numerical columns (including target)
        "SQLDATE", "MonthYear", "QuadClass", "GoldsteinScale", 
        "ActionGeo_Lat", "ActionGeo_Long", "NumArticles",
        
        # categorical columns  
        "EventCode", "EventBaseCode", "EventRootCode",
        "Actor1Code", "Actor1Name", "Actor1CountryCode", 
        "ActionGeo_CountryCode"
    ]
    
    # verify all expected columns are present
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        # add missing columns with NaN values as a safety net
        print(f"WARNING: GDELT schema change detected!")
        print(f"Adding missing columns with NaN values: {missing_cols}")
        print(f"Available columns in data: {list(df.columns)}")
        print(f"This may impact model performance - consider updating the pipeline")
        
        for col in missing_cols:
            # use appropriate NA type based on expected column type
            if col in categorical_columns:
                df[col] = pd.NA  # will be filled with "UNKNOWN" later
            else:
                df[col] = pd.NA  # will be filled with 999 later
    
    # select only expected columns (removes any extra columns)
    df = df[expected_columns].copy()
    
    # hardcoded column type definitions
    categorical_columns = [
        "EventCode", "EventBaseCode", "EventRootCode",
        "Actor1Code", "Actor1Name", "Actor1CountryCode", 
        "ActionGeo_CountryCode"
    ]
    
    numerical_columns = [
        "SQLDATE", "MonthYear", "QuadClass", "GoldsteinScale", 
        "ActionGeo_Lat", "ActionGeo_Long", "NumArticles"
    ]
    
    # imputation strategy (hardcoded for consistency)
    imputation_strategy = {
        # categorical: unknown value
        "EventCode": "UNKNOWN",
        "EventBaseCode": "UNKNOWN", 
        "EventRootCode": "UNKNOWN",
        "Actor1Code": "UNKNOWN",
        "Actor1Name": "UNKNOWN",
        "Actor1CountryCode": "UNKNOWN",
        "ActionGeo_CountryCode": "UNKNOWN",
        
        # numerical: out-of-range value
        "SQLDATE": 99999999,
        "MonthYear": 999999,
        "QuadClass": 999,
        "GoldsteinScale": 999.0,
        "ActionGeo_Lat": 999.0,
        "ActionGeo_Long": 999.0,
        "NumArticles": 999  # shouldn't have missing values, but just in case
    }

    # fill missing values
    df.fillna(imputation_strategy, inplace=True)

    # Handle time and data columns
    df['date'] = pd.to_datetime(df['SQLDATE'], format='%Y%m%d')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df = df.drop(["date", "SQLDATE", "MonthYear"], axis=1)

    # Handle categorical data by numerical encoding
    if is_train == True:
        # initialize encoder
        encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        
        # fit encoder on train set and transform data
        df[categorical_columns] = encoder.fit_transform(df[categorical_columns])
        
        # save encoder if requested
        if save_data == True:
            encoder_path = 'ordinal_encoder_prototype.pkl' 
            joblib.dump(encoder, encoder_path)

            # Use existing run or create new one
            if use_active_run and mlflow.active_run() is not None:
                # Use the currently active run
                mlflow.log_artifact(encoder_path, artifact_path='preprocessing')
                os.remove(encoder_path)
            else:
                # Create new run as before
                with mlflow.start_run():
                    mlflow.log_artifact(encoder_path, artifact_path='preprocessing')
                    os.remove(encoder_path)

    elif is_train == False:
        # for inference/query mode, ALWAYS use the hardcoded encoder for consistency
        hardcoded_run_id = get_hardcoded_encoder_run_id()
        
        # load the hardcoded encoder
        try:
            artifact_path = mlflow.artifacts.download_artifacts(
                f"runs:/{hardcoded_run_id}/preprocessing/ordinal_encoder_prototype.pkl"
            )
            loaded_encoder = joblib.load(artifact_path)
            os.remove(artifact_path)
            print(f"Using hardcoded encoder from run: {hardcoded_run_id}")
        except Exception as e:
            raise ValueError(
                f"Could not load hardcoded encoder from run {hardcoded_run_id}. "
                f"Please ensure this run exists in MLflow and contains the encoder artifact. "
                f"Error: {e}"
            )
        
        # apply the encoder to categorical columns only
        df[categorical_columns] = loaded_encoder.transform(df[categorical_columns])

    # Extract features and labels
    X = df.drop(columns=[target_label])
    y = df[target_label]
    
    # Save data to parquet files if desired
    if save_data == True:
        PATH_DATA.mkdir(parents=True, exist_ok=True)
        
        if save_path_dir is not None:
            save_name = PATH_DATA / save_path_dir
        else:
            num_dirs = sum(1 for p in PATH_DATA.iterdir() if p.is_dir())
            save_name = PATH_DATA / f"gdelt_events_2024_subset_version_{num_dirs}"
            
        save_name.mkdir(parents=True, exist_ok=True)
            
        if is_train == True:
            X_name = "X_train"
            y_name = "y_train"
        else:
            X_name = "X_query"  # Changed from "X_test" to "X_query"
            y_name = "y_query"  # Changed from "y_test" to "y_query"
            
        X.to_parquet(save_name / f"{X_name}.parquet")
        save_series_as_parquet(y, save_name / f"{y_name}.parquet")

    # Return results
    if is_train == True:
        return X, y, encoder
    else:
        return X, y
    

# main function
@click.command()
@click.option("--train", is_flag=True, help="Prepare train data")
@click.option("--query", is_flag=True, help="Prepare data to query model on such as test data")
@click.option("--subset", is_flag=True, help="Prepare 1k subset of train data")
@click.option("--subset-size", type=int, default=1000, help="Size of subset to prepare")
@click.option("--data-version", type=str, default="2024_subset_10k", help="Version of data to prepare")
@click.option("--source-data-version", type=str, default=None, help="Source data version to load from (defaults to data-version)")
@click.option("--query-data-file", type=str, default=None, help="Custom query data file path (if different from test data)")
def main(train, query, subset, subset_size=1000, data_version="2024_subset_10k", source_data_version=None, query_data_file=None):
    """
    Main function to prepare data for modeling.
    
    Args:
        train: If True, prepares training data and fits encoder
        query: If True, prepares query/test data using existing encoder
        subset: If True, creates subset of training data
        subset_size: Size of the subset to create
        data_version: Name for the output data version AND default source version
        source_data_version: Version of source data to load from (overrides data_version for loading)
        query_data_file: Path to custom query data file (optional)
        
    The train and test set will be saved to whatever you specify in the
    --data-version flag. The default is 2024_subset_10k.
    If you run this in query mode, the data will also be saved to whatever you pass
    here, but it will load from the same version name unless you specify a different
    --source-data-version.
    
    Examples:
        # Use same version for input and output
        python script.py --query --data-version "my_experiment"
        # Loads from: gdelt_events_my_experiment_test.parquet
        # Saves to: data/intermediate/my_experiment/X_query.parquet
        
        # Use different source and output versions  
        python script.py --query --data-version "output_v2" --source-data-version "2024_subset_10k"
        # Loads from: gdelt_events_2024_subset_10k_test.parquet
        # Saves to: data/intermediate/output_v2/X_query.parquet
    """

    print("Setting up workspace...")

    # Paths
    PATH_REPO = Path(__file__).resolve().parent.parent
    PATH_DATA = PATH_REPO / "data" / "raw"
    
    # Determine source data version to load from
    # If source_data_version is not specified, use data_version for both loading and saving
    load_version = source_data_version if source_data_version is not None else data_version
    
    # Source data paths (what we load)
    PATH_TRAIN_SOURCE = PATH_DATA / f"gdelt_events_{load_version}_train.parquet"
    PATH_TEST_SOURCE = PATH_DATA / f"gdelt_events_{load_version}_test.parquet"  # Uses load_version
    
    # Custom query data path if provided
    if query_data_file:
        PATH_QUERY_SOURCE = Path(query_data_file)
    else:
        PATH_QUERY_SOURCE = PATH_TEST_SOURCE

    # MLFlow
    # set MLFlow tracking URI or rather: basically connect to the MLFlow server
    # Use environment variable if available, otherwise default to local
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
    mlflow.set_tracking_uri(mlflow_uri)
    print(f"Using MLflow URI: {mlflow_uri}")

    # set experiment
    #mlflow.set_experiment("gdelt-newsimpact")
    # FIXME: SET THE ACTUAL EXPERIMENT NAME INTERACTIVELY
    mlflow.set_experiment("data_preparation")

    # Check if source files exist and provide helpful error messages
    if train and not PATH_TRAIN_SOURCE.exists():
        raise FileNotFoundError(
            f"Train data not found: {PATH_TRAIN_SOURCE}\n"
            f"Available files in {PATH_DATA}:\n" + 
            "\n".join([f"  {f.name}" for f in PATH_DATA.glob("*.parquet")])
        )
    
    if query and not PATH_QUERY_SOURCE.exists():
        raise FileNotFoundError(
            f"Query data not found: {PATH_QUERY_SOURCE}\n"
            f"Available files in {PATH_DATA}:\n" + 
            "\n".join([f"  {f.name}" for f in PATH_DATA.glob("*.parquet")])
        )

    # Load data from parquet files
    print("Loading data from parquet files...")
    
    if train or subset:
        print(f"Loading train data from: {PATH_TRAIN_SOURCE}")
        df_train = pd.read_parquet(PATH_TRAIN_SOURCE)
    
    if query:
        print(f"Loading query data from: {PATH_QUERY_SOURCE}")
        df_query = pd.read_parquet(PATH_QUERY_SOURCE)

    # Prepare data
    print("Preparing data...")

    if train == True:
        # Train data and fit encoder
        print("Preparing train data...")
        with mlflow.start_run() as run:
            X_train, y_train, encoder = prepare_data(
                df=df_train,
                is_train=True,
                save_data=True,
                save_path_dir=data_version,  # Save with output version name
                path_repo=PATH_REPO,
                use_active_run=True
            )
            training_run_id = run.info.run_id
            print(f"Training completed. Run ID: {training_run_id}")
            print(f"Train data shape: {X_train.shape}")
            print(f"Saved to: data/intermediate/{data_version}/")

    if query == True:
        print("Preparing query data...")
        # Query data with auto-fetch encoder
        X_query, y_query = prepare_data(
            df=df_query,
            is_train=False,
            save_data=True,
            save_path_dir=data_version,  # Save with output version name
            path_repo=PATH_REPO
        )
        print(f"Query data shape: {X_query.shape}")
        print(f"Loaded from: {PATH_QUERY_SOURCE}")
        print(f"Saved to: data/intermediate/{data_version}/")

    if subset == True and train == True:
        # Get subset from processed train data
        print(f"Preparing {subset_size} subset...")
        subset_indices = y_train.sample(n=subset_size, random_state=42).index
        X_train_subset = X_train.loc[subset_indices]
        y_train_subset = y_train.loc[subset_indices]

        # Save subset
        PATH_INTERMEDIATE = PATH_REPO / "data" / "intermediate" / f"subset_{subset_size}_{data_version}"
        PATH_INTERMEDIATE.mkdir(parents=True, exist_ok=True)

        X_train_subset.to_parquet(PATH_INTERMEDIATE / "X_train.parquet")
        save_series_as_parquet(y_train_subset, PATH_INTERMEDIATE / "y_train.parquet")
        print(f"Subset saved to: {PATH_INTERMEDIATE}")
        print(f"Subset shape: {X_train_subset.shape}")
        
    elif subset == True and train == False:
        print("Cannot create subset without training data. Use --train --subset together.")

    print("Data preparation completed!")

# Main
# -----
if __name__ == "__main__":
    main()
