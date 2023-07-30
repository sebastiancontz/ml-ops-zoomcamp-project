import pandas as pd
import numpy as np
import os
from datetime import datetime
import pytz

from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
import mlflow

import awswrangler as wr
wr.config.s3_endpoint_url = os.getenv("AWS_ENDPOINT_URL")

from real_estate_model.utils import functions
from real_estate_model.flows.model_batch_metrics import generate_metrics
from real_estate_model.flows.simulate_data import simulate_data

@task(name="generate_prediction")
def generate_prediction(data: pd.DataFrame, model: mlflow.sklearn.Model):
    preds = model.predict(data)
    return preds

@task(name="upload_prediction_to_s3")
def upload_prediction_to_s3(data: pd.DataFrame, out_filepath: str):
    wr.s3.to_csv(data, out_filepath, index=False)
    return True

@flow(name="batch_prediction")
def batch_prediction(filepath: bool=False, is_s3_file: bool=False):
    if filepath is False:
        data = simulate_data()
    else:
        data = functions.read_data(filepath, is_s3_file)
    data = functions.process_csv(data, True)

    # load productive model from mlflow
    model = mlflow.sklearn.load_model("models:/real-estate-model/production")

    # make predictions
    data["price_prediction"] = generate_prediction(data, model)

    # save predictions to s3
    now = datetime.utcnow().replace(tzinfo=pytz.utc)
    now = now.strftime('%Y%m%d_%H%M')
    filename = f"predictions_{now}.csv"
    out_filepath = f"s3://predictions/{filename}"

    upload_prediction_to_s3(data, out_filepath)

    # generate report
    prediction_report = f"""# Prediction Report
    
    ## Summary

    Prediction info:
    - Process date: {now}
    - Input file: {filepath}
    - Output file: {filename}
    - Shape: {data.shape}
    - s3 location: {out_filepath}

    """ 

    create_markdown_artifact(
        key="batch-prediction-report", 
        markdown=prediction_report,
        description="Batch Prediction Report")

    # register metrics
    generate_metrics(data, datetime.strptime(now, '%Y%m%d_%H%M'))

    return True