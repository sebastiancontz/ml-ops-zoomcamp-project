
import os
from datetime import datetime
import pandas as pd
import awswrangler as wr
from sqlalchemy import create_engine
import mlflow
from prefect import task, flow
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric
from evidently.report import Report
from evidently import ColumnMapping

@task(name="load_train_data")
def load_train_data():
    training = wr.s3.read_csv(f's3://files/training_data.csv')
    model = mlflow.sklearn.load_model("models:/real-estate-model/production")
    training.drop(columns=["house_price_of_unit_area"], inplace=True)
    training["price_prediction"] = model.predict(training)
    return training

@task(name="generate_evidently_report")
def generate_evidently_report():
    num_features = ['house_age', 'distance_to_the_nearest_MRT_station', 'number_of_convenience_stores', 'latitude', 'longitude']
    cat_features = None
    column_mapping = ColumnMapping(
        target =None,
        numerical_features=num_features,
        categorical_features=cat_features,
        prediction="price_prediction"
    )

    report = Report(metrics = [
        ColumnDriftMetric(column_name='price_prediction'),
        DatasetDriftMetric()
    ])

    return report, column_mapping

@task(name="extract_batch_report_data")
def extract_report_data(batch_date, report, column_mapping, training_data, batch_data):
    report.run(reference_data=training_data, current_data=batch_data, column_mapping=column_mapping)
    drift_report = report.as_dict()["metrics"]

    drift_prediction = {
        "batch_date": batch_date,
        "drift_stat_test": drift_report[0]["result"]["stattest_name"],
        "drift_stat_threshold": drift_report[0]["result"]["stattest_threshold"],
        "drift_score": drift_report[0]["result"]["drift_score"],
        "drift_detected": drift_report[0]["result"]["drift_detected"],
    }

    drift_dataset = {
        "batch_date": batch_date,
        "drift_dataset": drift_report[1]["result"]["drift_share"],
        "number_of_columns": drift_report[1]["result"]["number_of_columns"],
        "number_of_drifted_columns": drift_report[1]["result"]["number_of_drifted_columns"],
        "share_of_drifted_columns": drift_report[1]["result"]["share_of_drifted_columns"],
        "dataset_drift": drift_report[1]["result"]["dataset_drift"]
    }

    params = {
        "user": os.getenv("POSTGRES_USER"),
        "pass": os.getenv("POSTGRES_PASSWORD"),
        "host": os.getenv("POSTGRES_HOST"),
        "port": os.getenv("POSTGRES_PORT"),
        "database": "metrics_batch"
    }
    engine = create_engine('postgresql://%(user)s:%(pass)s@%(host)s:%(port)s/%(database)s' % params)

    # insert metrics
    pd.DataFrame(drift_prediction, index=[0]).to_sql("drift_prediction", engine, if_exists="append", index=False)
    pd.DataFrame(drift_dataset, index=[0]).to_sql("drift_dataset", engine, if_exists="append", index=False)
    

@flow(name="generate_batch_metrics")
def generate_metrics(batch_data: pd.DataFrame, batch_date: datetime=datetime.now()):
    training = load_train_data()
    report, column_mapping = generate_evidently_report()
    extract_report_data(batch_date, report, column_mapping, training, batch_data)