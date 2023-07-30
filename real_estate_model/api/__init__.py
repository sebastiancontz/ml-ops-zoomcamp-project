import os
from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config["PREFECT_API_URL"] = os.getenv("PREFECT_API_URL")
    app.config["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
    app.config["PREFECT_API_URL"] = os.getenv("PREFECT_API_URL")
    app.config["PREFECT_DEPLOYMENT_ENDPOINT"] = f"{app.config['PREFECT_API_URL']}/deployments/%s/create_flow_run"
    
    from real_estate_model.api.prediction_api import api
    app.register_blueprint(api)
    
    return app