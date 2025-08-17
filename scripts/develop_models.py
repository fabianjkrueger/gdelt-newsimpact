"""
MLOps Project 2: Boosting Models Development with Cross-Validation and HPO
==========================================================================

This script performs hyperparameter optimization and model selection for
regression tasks using three boosting algorithms: XGBoost, LightGBM, and
CatBoost. It uses cross-validation for model selection and evaluates only the
best model on the test set to avoid data leakage.

Workflow:
---------
1. Loads preprocessed training and test data from parquet files
2. Runs lightweight hyperparameter optimization (50 trials) for each algorithm
using Optuna
3. Selects the best model based on 5-fold cross-validation RMSE scores
4. Trains only the winning model on the full training set
5. Evaluates the final model on the test set
6. Logs all experiments to MLflow with proper signatures and model registry
integration

The script is optimized for datasets with mostly categorical features
(8k rows, 20 features) and uses adjusted hyperparameter search spaces to prevent
overfitting on small datasets.

Data Requirements:
------------------
- X_train.parquet, y_train.parquet: Training features and targets
- X_query.parquet, y_query.parquet: Test features and targets (or X_query/y_query)

MLflow Integration:
-------------------
- Logs CV-only runs for non-winning models
- Logs full results (CV + test metrics + model artifact) for the best model
- Registers the best model to MLflow Model Registry with alias "cyber_dragon"
- Includes model signatures and input examples for deployment readiness

Coding Style:
-------------
I didn't refactor or parametrize the code in this script too much, because this
is a clearly defined experiment and I won't run this with other parameters.
It's a specialized tool rather than a general purpose pipeline, so keeping it
simple and focused is preferred.
Adding arguments or refactoring everything into functions would add complexity
without clear benefits in this case, so I will keep it like this.

Usage:
------
0. Have the data prepared and in the right place. Just follow the instructions
from the README for this
1. Ensure MLflow server is running: mlflow server --host 127.0.0.1 --port 5001
2. Run the script: `uv run scripts/develop_models.py` (no parameters)
3. Check results in MLflow UI: http://127.0.0.1:5001
"""

print("\nStarting development of boosting models...")
# Dependencies
# ------------
print("\nLoading libraries...")
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost
from mlflow.tracking import MlflowClient
from pathlib import Path
import optuna
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')
from mlflow.models.signature import infer_signature

# Configuration
# -------------
print("\nSetting up workspace...")
DATA_VERSION = "2024_subset_10k"
N_FOLDS = 5
N_TRIALS = 50  # lightweight HPO - increase for better results but longer runtime
RANDOM_STATE = 42
METRIC = 'rmse'  # primary metric for optimization

# Paths
PATH_REPO = Path(__file__).parent.parent
PATH_DATA = PATH_REPO / "data" / "intermediate" / DATA_VERSION

# MLflow setup
mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("boosting_models_cv")

# Load prepared data
# -----------------
print("\nLoading data...")
X_train = pd.read_parquet(PATH_DATA / "X_train.parquet")
y_train = pd.read_parquet(PATH_DATA / "y_train.parquet").squeeze()
X_test = pd.read_parquet(PATH_DATA / "X_query.parquet")
y_test = pd.read_parquet(PATH_DATA / "y_query.parquet").squeeze()

# briefly validate that correct data was used by printing shape
# this way, user can compare num rows to what they used as input
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# setup cross validation strategy
cv_strategy = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Lightweight hyperparameter search spaces
# ----------------------------------------
# Data has just 8k rows for training and mostly categorical features
# Use smaller space and enable categorical handling in XGBoost
print("\nSetting up hyperparameter search spaces...")
def get_xgboost_search_space(trial):
    """Optimized for 8k rows, 20 features, categorical data"""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'learning_rate': trial.suggest_float(
            'learning_rate', 0.01, 0.3, log=True,
        ),
        'subsample': trial.suggest_float('subsample', 0.8, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 3.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0, log=True),
        'random_state': RANDOM_STATE,
        'tree_method': 'hist',
        'enable_categorical': True,
    }

def get_lightgbm_search_space(trial):
    """Optimized for categorical GDELT data"""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'learning_rate': trial.suggest_float(
            'learning_rate', 0.01, 0.3, log=True,
        ),
        'num_leaves': trial.suggest_int('num_leaves', 15, 80),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'subsample': trial.suggest_float('subsample', 0.8, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 3.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0, log=True),
        'random_state': RANDOM_STATE,
        'verbose': -1,
    }

def get_catboost_search_space(trial):
    """Lightweight CatBoost hyperparameter space"""
    return {
        'iterations': trial.suggest_int('iterations', 100, 400),
        'depth': trial.suggest_int('depth', 4, 8),
        'learning_rate': trial.suggest_float(
            'learning_rate', 0.01, 0.3, log=True,
        ),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'bootstrap_type': trial.suggest_categorical(
            'bootstrap_type', ['Bayesian', 'Bernoulli'],
        ),
        'random_seed': RANDOM_STATE,
        'verbose': False,
        'allow_writing_files': False,
    }

print("\nDefining functions...")

# Cross-validation evaluation function
# -----------------------------------

def evaluate_model_cv(model, X, y, cv_strategy):
    """Evaluate model using cross-validation for regression"""
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv_strategy,
        scoring='neg_mean_squared_error',
    )
    return -scores.mean()  # return positive RMSE

# Optuna objective functions
# -------------------------

def xgboost_objective(trial):
    """Optuna objective for XGBoost"""
    params = get_xgboost_search_space(trial)
    model = xgb.XGBRegressor(**params)
    score = evaluate_model_cv(model, X_train, y_train, cv_strategy)
    return score

def lightgbm_objective(trial):
    """Optuna objective for LightGBM"""
    params = get_lightgbm_search_space(trial)
    model = lgb.LGBMRegressor(**params)
    score = evaluate_model_cv(model, X_train, y_train, cv_strategy)
    return score

def catboost_objective(trial):
    """Optuna objective for CatBoost"""
    params = get_catboost_search_space(trial)
    model = cb.CatBoostRegressor(**params)
    score = evaluate_model_cv(model, X_train, y_train, cv_strategy)
    return score

# Model training and evaluation
# -----------------------------

def train_and_evaluate_final_model(
    model_name,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
):
    """Train model and evaluate on test set"""
    
    # fit on full training set
    model.fit(X_train, y_train)
    
    # predict on test set
    y_pred = model.predict(X_test)
    
    # calculate regression metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'test_mse': mse,
        'test_rmse': rmse,
        'test_mae': mae,
        'test_r2': r2
    }
    
    print(f"{model_name} Test Results:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}\n")
    
    return model, metrics


# Cross validation for HPO and determining best model
# ---------------------------------------------------
print("\nStarting cross validation for HPO and determining best model...")

results = {}

# 1. XGBoost
print("\n❌XGBoost❌")
print("Running hyperparameter optimization...")

# minimize function, because we use regression, and metrics are good when low
study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(xgboost_objective, n_trials=N_TRIALS, show_progress_bar=True)

# store results for later use
results['XGBoost'] = {
    'cv_score': study_xgb.best_value,
    'best_params': study_xgb.best_params
}

# print best CV score and parameters for users
print(f"\nBest XGBoost CV score: {study_xgb.best_value:.4f}")
print(f"Best XGBoost params:")
for param, value in study_xgb.best_params.items():
    print(f"    {param}: {value}")


# 2. LightGBM
print("\n⚡️LightGBM⚡️")
print("Running hyperparameter optimization...")

study_lgb = optuna.create_study(direction='minimize')
study_lgb.optimize(lightgbm_objective, n_trials=N_TRIALS, show_progress_bar=True)

results['LightGBM'] = {
    'cv_score': study_lgb.best_value,
    'best_params': study_lgb.best_params
}

print(f"\nBest LightGBM CV score: {study_lgb.best_value:.4f}")
print(f"Best LightGBM params:")
for param, value in study_lgb.best_params.items():
    print(f"    {param}: {value}")


# 3. CatBoost
print("\n🐱CatBoost🐱")
print("Running hyperparameter optimization...")

study_cb = optuna.create_study(direction='minimize')
study_cb.optimize(catboost_objective, n_trials=N_TRIALS, show_progress_bar=True)

results['CatBoost'] = {
    'cv_score': study_cb.best_value,
    'best_params': study_cb.best_params
}

print(f"\nBest CatBoost CV score: {study_cb.best_value:.4f}")
print(f"Best CatBoost params:")
for param, value in study_cb.best_params.items():
    print(f"    {param}: {value}")


# Final comparison and model selection
# -----------------------------------
print("\nFinal comparison on cross validation results...")

# print all CV scores
print("\nCross-Validation RMSE:")
for model_name, result in results.items():
    print(f"  {model_name}: {result['cv_score']:.4f}")

# find best model based on CV score
best_model_name = min(results.keys(), key=lambda x: results[x]['cv_score'])
print(f"\nBest model (based on CV): {best_model_name}")
print(f"Best CV score: {results[best_model_name]['cv_score']:.4f}")

# Train and evaluate ONLY the best model on test set
# Log CV results of ALL models
# --------------------------------------------------

# model creation mapping
model_creators = {
    'XGBoost': lambda params: xgb.XGBRegressor(**params),
    'LightGBM': lambda params: lgb.LGBMRegressor(**params),
    'CatBoost': lambda params: cb.CatBoostRegressor(**params)
}

# mlflow logging functions
mlflow_loggers = {
    'XGBoost': mlflow.xgboost.log_model,
    'LightGBM': mlflow.lightgbm.log_model,
    'CatBoost': mlflow.catboost.log_model
}

# Log CV results for all models, PLUS test metrics for the best one
print("\nLogging results to MLflow...")
client = MlflowClient()

for model_name, result in results.items():
    is_best = (model_name == best_model_name)
    run_name = f"{model_name.lower()}_{'final' if is_best else 'cv_only'}"
    
    with mlflow.start_run(run_name=run_name):
        # Always log CV results
        mlflow.log_params(result['best_params'])
        mlflow.log_param("cv_folds", N_FOLDS)
        mlflow.log_param("n_trials", N_TRIALS)
        mlflow.log_metric("cv_score", result['cv_score'])
        mlflow.log_param("is_final_model", is_best)
        
        # For best model: also train and log test results + model artifact
        if is_best:
            print(f"\nTraining and evaluating final model: {model_name}")
            
            # create and train the model
            best_model = model_creators[model_name](result['best_params'])
            final_model, test_metrics = train_and_evaluate_final_model(
                model_name, best_model, X_train, y_train, X_test, y_test
            )
            
            # log test metrics
            mlflow.log_metrics(test_metrics)
            
            # create signature and input example
            signature = infer_signature(X_train, final_model.predict(X_train))
            input_example = X_train.head(3)
            
            # log and register model simultaneously
            model_info = mlflow_loggers[model_name](
                final_model, 
                name="model",
                signature=signature,
                input_example=input_example,
                registered_model_name="gdelt-event-classifier"
            )
            
            # manage mobel name, because staging is deprecated
            model_version = model_info.registered_model_version
            client.set_registered_model_alias(
                name="gdelt-event-classifier",
                alias="cyber_dragon",
                version=model_version
            )
            
            # add metadata
            client.update_model_version(
                name="gdelt-event-classifier",
                version=model_version,
                description=(
                    f"Best {model_name} model. "
                    f"CV RMSE: {result['cv_score']:.4f}, "
                    f"Test RMSE: {test_metrics['test_rmse']:.4f}"
                )
            )
            
            print(
                f"Model registered as 'gdelt-event-classifier' v{model_version} "
                f"with alias 'cyber_dragon'"
            )

print(f"\nTraining completed! Check MLflow UI at http://127.0.0.1:5001")
print(f"Model Registry: http://127.0.0.1:5001/#/models/gdelt-event-classifier")