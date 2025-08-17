# MLOps Project 2: Boosting Models with 5-Fold CV and Lightweight HPO
# ================================================================

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost
from pathlib import Path
import optuna
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

# Configuration
# -------------
print("Setting up workspace...")
DATA_VERSION = "2024_subset_10k"  # adjust to your data version
N_FOLDS = 5
N_TRIALS = 50  # lightweight HPO - increase for better results but longer runtime
RANDOM_STATE = 42
METRIC = 'rmse'  # primary metric for optimization

# Paths
PATH_REPO = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
PATH_DATA = PATH_REPO / "data" / "intermediate" / DATA_VERSION

# MLflow setup
mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("boosting_models_cv")

# Load prepared data
# -----------------
print("Loading data...")
X_train = pd.read_parquet(PATH_DATA / "X_train.parquet")
y_train = pd.read_parquet(PATH_DATA / "y_train.parquet").squeeze()
X_test = pd.read_parquet(PATH_DATA / "X_query.parquet")  
y_test = pd.read_parquet(PATH_DATA / "y_query.parquet").squeeze()

# setup cross validation strategy
cv_strategy = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Lightweight hyperparameter search spaces
# ----------------------------------------
print("Setting up hyperparameter search spaces...")
def get_xgboost_search_space(trial):
    """Lightweight XGBoost hyperparameter space"""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float(
            'learning_rate', 0.01, 0.3, log=True,
        ),
        'subsample': trial.suggest_float(
            'subsample', 0.7, 1.0,
        ),
        'colsample_bytree': trial.suggest_float(
            'colsample_bytree', 0.7, 1.0,
        ),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float(
            'reg_lambda', 1e-8, 10.0, log=True,
        ),
        'random_state': RANDOM_STATE,
    }

def get_lightgbm_search_space(trial):
    """Lightweight LightGBM hyperparameter space"""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float(
            'learning_rate', 0.01, 0.3, log=True,
        ),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float(
            'colsample_bytree', 0.7, 1.0,
        ),
        'reg_alpha': trial.suggest_float(
            'reg_alpha', 1e-8, 10.0, log=True,
        ),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'random_state': RANDOM_STATE,
        'verbose': -1
    }

def get_catboost_search_space(trial):
    """Lightweight CatBoost hyperparameter space"""
    return {
        'iterations': trial.suggest_int('iterations', 100, 500),
        'depth': trial.suggest_int('depth', 3, 8),
        'learning_rate': trial.suggest_float(
            'learning_rate', 0.01, 0.3, log=True,
        ),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'bootstrap_type': trial.suggest_categorical(
            'bootstrap_type', ['Bayesian', 'Bernoulli'],
        ),
        'random_seed': RANDOM_STATE,
        'verbose': False
    }

print("Defining functions...")

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
    print(f"\nTraining final {model_name} model...")
    
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
    print(f"  R²: {r2:.4f}")
    
    return model, metrics

# Main training pipeline
# ---------------------

print("\n" + "="*60)
print("STARTING BOOSTING MODELS TRAINING WITH 5-FOLD CV")
print("="*60)

results = {}

# 1. XGBoost
print(f"\n{'='*20} XGBOOST {'='*20}")
print("Running hyperparameter optimization...")

# minimize, because we use regression, and metrics are good when low
study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(xgboost_objective, n_trials=N_TRIALS, show_progress_bar=True)

print(f"Best XGBoost CV score: {study_xgb.best_value:.4f}")
print(f"Best XGBoost params: {study_xgb.best_params}")

# train final XGBoost model
best_params_xgb = study_xgb.best_params
best_xgb = xgb.XGBRegressor(**best_params_xgb)

with mlflow.start_run(run_name="xgboost_cv"):
    # log hyperparameters
    mlflow.log_params(best_params_xgb)
    mlflow.log_param("cv_folds", N_FOLDS)
    mlflow.log_param("n_trials", N_TRIALS)
    mlflow.log_metric("cv_score", study_xgb.best_value)
    
    # train and evaluate
    model_xgb, metrics_xgb = train_and_evaluate_final_model(
        "XGBoost",
        best_xgb,
        X_train,
        y_train,
        X_test,
        y_test,
    )
    
    # log metrics and model
    mlflow.log_metrics(metrics_xgb)
    mlflow.xgboost.log_model(model_xgb, "model")

results['XGBoost'] = {
    'cv_score': study_xgb.best_value,
    'test_metrics': metrics_xgb,
    'best_params': best_params_xgb
}

# 2. LightGBM
print(f"\n{'='*20} LIGHTGBM {'='*20}")
print("Running hyperparameter optimization...")

study_lgb = optuna.create_study(direction='minimize')
study_lgb.optimize(lightgbm_objective, n_trials=N_TRIALS, show_progress_bar=True)

print(f"Best LightGBM CV score: {study_lgb.best_value:.4f}")
print(f"Best LightGBM params: {study_lgb.best_params}")

# train final LightGBM model
best_params_lgb = study_lgb.best_params
best_lgb = lgb.LGBMRegressor(**best_params_lgb)

with mlflow.start_run(run_name="lightgbm_cv"):
    # log hyperparameters
    mlflow.log_params(best_params_lgb)
    mlflow.log_param("cv_folds", N_FOLDS)
    mlflow.log_param("n_trials", N_TRIALS)
    mlflow.log_metric("cv_score", study_lgb.best_value)
    
    # train and evaluate
    model_lgb, metrics_lgb = train_and_evaluate_final_model(
        "LightGBM",
        best_lgb,
        X_train,
        y_train,
        X_test,
        y_test,
    )
    
    # log metrics and model
    mlflow.log_metrics(metrics_lgb)
    mlflow.lightgbm.log_model(model_lgb, "model")

results['LightGBM'] = {
    'cv_score': study_lgb.best_value,
    'test_metrics': metrics_lgb,
    'best_params': best_params_lgb
}

# 3. CatBoost
print(f"\n{'='*20} CATBOOST {'='*20}")
print("Running hyperparameter optimization...")

study_cb = optuna.create_study(direction='minimize')
study_cb.optimize(catboost_objective, n_trials=N_TRIALS, show_progress_bar=True)

print(f"Best CatBoost CV score: {study_cb.best_value:.4f}")
print(f"Best CatBoost params: {study_cb.best_params}")

# train final CatBoost model
best_params_cb = study_cb.best_params
best_cb = cb.CatBoostRegressor(**best_params_cb)

with mlflow.start_run(run_name="catboost_cv"):
    # log hyperparameters
    mlflow.log_params(best_params_cb)
    mlflow.log_param("cv_folds", N_FOLDS)
    mlflow.log_param("n_trials", N_TRIALS)
    mlflow.log_metric("cv_score", study_cb.best_value)
    
    # train and evaluate
    model_cb, metrics_cb = train_and_evaluate_final_model(
        "CatBoost",
        best_cb,
        X_train,
        y_train,
        X_test,
        y_test,
    )
    
    # log metrics and model
    mlflow.log_metrics(metrics_cb)
    mlflow.catboost.log_model(model_cb, "model")

results['CatBoost'] = {
    'cv_score': study_cb.best_value,
    'test_metrics': metrics_cb,
    'best_params': best_params_cb
}

# Final comparison
# ---------------
print(f"\n{'='*20} FINAL RESULTS {'='*20}")

print("\nCross-Validation RMSE:")
for model_name, result in results.items():
    print(f"  {model_name}: {result['cv_score']:.4f}")

print("\nTest Set RMSE:")
for model_name, result in results.items():
    rmse = result['test_metrics']['test_rmse']
    print(f"  {model_name}: {rmse:.4f}")

# find best model
best_model_cv = min(results.keys(), key=lambda x: results[x]['cv_score'])
best_model_test = min(
    results.keys(),
    key=lambda x: results[x]['test_metrics']['test_rmse'],
)

print(f"\nBest model (CV): {best_model_cv}")
print(f"Best model (Test): {best_model_test}")

print(
    "\nTraining completed! ",
    "Check MLflow UI at http://127.0.0.1:5001 for detailed results.",
)