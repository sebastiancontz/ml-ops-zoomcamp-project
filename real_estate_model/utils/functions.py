import os

import boto3
from botocore.exceptions import ClientError
import pickle
from pathlib import Path
import pandas as pd
import requests

from prefect import task, flow
import awswrangler as wr
wr.config.s3_endpoint_url = os.getenv("AWS_ENDPOINT_URL")

def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)
    
def dump_pickle(filename, obj):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)

def process_csv(dataframe: pd.DataFrame, is_prediction: bool = False):
    if is_prediction:
        dataframe.columns = ["transaction_date", "house_age", "distance_to_the_nearest_MRT_station", "number_of_convenience_stores", "latitude", "longitude"]
    else:
        dataframe.columns = ["transaction_date", "house_age", "distance_to_the_nearest_MRT_station", "number_of_convenience_stores", "latitude", "longitude", "house_price_of_unit_area"]

    dataframe.drop(columns=["transaction_date"], inplace=True)
    return dataframe.copy()

def upload_file_to_s3(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client(
        's3', 
        endpoint_url = os.getenv("AWS_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        print(e)
        return False
    return True

@task(name="read_data")
def read_data(filepath: str, is_s3_file: bool = False):
    if is_s3_file:
        data = wr.s3.read_csv(filepath)
    else:
        data = pd.read_csv(filepath)

    if 'No' in data.columns:
        data.drop(columns=['No'], inplace=True)

    data.reset_index(drop=True, inplace=True)
    return data

def validate_headers(request, expected):
    parameters = [parameter for parameter in request.json.keys()]
    missing_parameters = (list(set(expected)-set(parameters)))
    if missing_parameters != []:
        missing = True
        parameters = (", ".join(missing_parameters))
    else:
        missing = False
        parameters = None
    return (missing, parameters)

def get_prefect_flow_id(flow_name: str):
    prefect_api_url = os.getenv("PREFECT_API_URL")
    prefect_api_url = prefect_api_url[:-1] if prefect_api_url[-1] == "/" else prefect_api_url
    response = requests.get(f"{prefect_api_url}/deployments/name/{flow_name}/{flow_name}")
    flow_id = response.json()["id"]
    return flow_id

