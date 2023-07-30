import os
import pandas as pd
import numpy as np
import requests
from flask import Blueprint, current_app, jsonify, request
import mlflow
from mlflow.exceptions import RestException, MlflowException

from real_estate_model.utils import functions

api = Blueprint("api", __name__)

@api.route("/predict", methods=["POST"])
def predict():
    mlflow.set_tracking_uri(current_app.config["MLFLOW_TRACKING_URI"])
    try:
        model = mlflow.pyfunc.load_model("models:/real-estate-model/production")
    except (RestException, MlflowException):
        return jsonify({"error": "Model not trained yet, please train a model within prefect UI or calling /trigger-training endpoint"}), 404

    expected = ["house_age", "distance_to_the_nearest_MRT_station", "number_of_convenience_stores", "latitude", "longitude"]
    
    missing, parameters = functions.validate_headers(request, expected)

    if missing:
        return jsonify({"error": f"Missing parameters: {parameters}"}), 400
    
    data = pd.DataFrame(request.json, index=[0])
    data = data.applymap(np.float64)
    data["number_of_convenience_stores"] = data["number_of_convenience_stores"].astype(np.int32)
    y_hat = model.predict(data)

    response = {
        "input_data": request.json,
        "prediction": y_hat[0], 
        "model_metadata": {
            "run_id": model.metadata.run_id,
            "model_uuid": model.metadata.model_uuid,
        }
    }

    return jsonify(response), 200

@api.route("/trigger-batch-prediction", methods=["POST"])
def trigger_batch_prediction():
    if 's3_file_path' not in request.json:
        return jsonify({"error": "Missing parameters: s3_file_path"}), 400
    
    flow_id = functions.get_prefect_flow_id("batch_prediction")
    
    payload = {
        "state":
            {
                "type": "SCHEDULED",
                "message": "Trigger from API",
                "state_details": {}
            },
        "parameters": 
            {
                "filepath": request.json["s3_file_path"],
                "is_s3_file": True
            }
    }

    if request.json["s3_file_path"] is False:
        payload["parameters"]["filepath"] = False
        payload["parameters"]["is_s3_file"] = False

    url = current_app.config["PREFECT_DEPLOYMENT_ENDPOINT"] % flow_id
    prefect_request = requests.post(url, json=payload)

    if prefect_request.status_code != 201:
        return jsonify({"error": "Something went wrong"}), 500
    
    response = {
        "flow_run_id": prefect_request.json()["id"],
        "flow_run_name": prefect_request.json()["name"]
    }

    return jsonify(response), 200

@api.route("/trigger-training", methods=["POST"])
def trigger_training():
    if 's3_file_path' not in request.json:
        return jsonify({"error": "Missing parameters: s3_file_path"}), 400
    
    flow_id = functions.get_prefect_flow_id("model_training")
    
    payload = {
        "state":
            {
                "type": "SCHEDULED",
                "message": "Trigger from API",
                "state_details": {}
            },
        "parameters": 
            {
                "filepath": request.json["s3_file_path"],
                "is_s3_file": True
            }
    }

    if request.json["s3_file_path"] is False:
        payload["parameters"]["filepath"] = "./real_estate_model/data/Real estate.csv"
        payload["parameters"]["is_s3_file"] = False

    url = current_app.config["PREFECT_DEPLOYMENT_ENDPOINT"] % flow_id
    prefect_request = requests.post(url, json=payload)

    if prefect_request.status_code != 201:
        print(prefect_request.json())
        return jsonify({"error": "Something went wrong"}), 500
    
    response = {
        "flow_run_id": prefect_request.json()["id"],
        "flow_run_name": prefect_request.json()["name"]
    }

    return jsonify(response), 200
