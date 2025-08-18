"""
XGBoost model serving with Flask and MLflow 3.2.
"""

import mlflow
import mlflow.xgboost
import pandas as pd
from flask import Flask, request, jsonify
import logging

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# load model from registry using alias (replaces staging in MLflow 3.x)
mlflow.set_tracking_uri("http://host.docker.internal:5001")
model_name = "gdelt-event-classifier"
alias = "cyber_dragon"  # alias replaces staging concept in MLflow 3.x
model_uri = f"models:/{model_name}@{alias}"

logger.info(f"Loading model: {model_uri}")
try:
    model = mlflow.xgboost.load_model(model_uri)
    logger.info("XGBoost model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

@app.route("/health", methods=["GET"])
def health():
    """health check endpoint"""
    return jsonify({"status": "healthy", "model": model_name, "alias": alias})

@app.route("/predict", methods=["POST"])
def predict():
    """prediction endpoint"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # convert to dataframe - handle both single dict and list of dicts
        df = pd.DataFrame([data] if isinstance(data, dict) else data)
        
        # expected features based on model signature from MLmodel file
        expected_features = [
            "QuadClass", "GoldsteinScale", "ActionGeo_Lat", "ActionGeo_Long",
            "EventCode", "EventBaseCode", "EventRootCode", "Actor1Code", 
            "Actor1Name", "Actor1CountryCode", "ActionGeo_CountryCode",
            "year", "month", "day_of_year", "day_of_week", "is_weekend"
        ]
        
        # validate features
        missing_features = [f for f in expected_features if f not in df.columns]
        if missing_features:
            return jsonify({
                "error": f"Missing required features: {missing_features}"
            }), 400
        
        # select only the expected features in correct order
        X = df[expected_features]
        
        # make predictions
        predictions = model.predict(X)
        
        # return result
        if len(predictions) == 1:
            return jsonify({"prediction": float(predictions[0])})
        else:
            return jsonify({"predictions": [float(p) for p in predictions]})
            
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/model-info", methods=["GET"])
def model_info():
    """return model information"""
    return jsonify({
        "model_name": model_name,
        "alias": alias,
        "model_uri": model_uri,
        "expected_features": [
            "QuadClass", "GoldsteinScale", "ActionGeo_Lat", "ActionGeo_Long",
            "EventCode", "EventBaseCode", "EventRootCode", "Actor1Code", 
            "Actor1Name", "Actor1CountryCode", "ActionGeo_CountryCode",
            "year", "month", "day_of_year", "day_of_week", "is_weekend"
        ]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)