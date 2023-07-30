
import os
from datetime import datetime
import pandas as pd
import awswrangler as wr
from sqlalchemy import create_engine
import mlflow
from prefect import task, flow
from evidently.metric_preset import RegressionPreset
from evidently.report import Report
from evidently import ColumnMapping

@task(name="load_model")
def load_model(model_uri: str="models:/real-estate-model/production"):
    model = mlflow.sklearn.load_model(model_uri)
    return model

@task(name="load_data")
def load_data(date: datetime=datetime.now()):
    extracion_date = date.strftime("%Y%m%d")
    training = wr.s3.read_csv(f's3://files/{extracion_date}/training_data.csv')
    testing = wr.s3.read_csv(f's3://files/{extracion_date}/testing_data.csv')
    
    return training, testing

@task(name="generate_evidently_report")
def generate_evidently_report():
    num_features = ['house_age', 'distance_to_the_nearest_MRT_station', 'number_of_convenience_stores', 'latitude', 'longitude']
    cat_features = None
    column_mapping = ColumnMapping(
        target ='house_price_of_unit_area',
        numerical_features=num_features,
        categorical_features=cat_features,
        prediction="price_prediction"
    )

    report = Report(metrics=[
        RegressionPreset()
    ])

    return report, column_mapping

@task(name="extract_train_report_data")
def extract_report_data(report, column_mapping, training_data, testing_data):
    report.run(reference_data=training_data, current_data=testing_data, column_mapping=column_mapping)
    report_data = report.as_dict()["metrics"][0]["result"]
    metrics = {
        "rmse": report_data["current"]["rmse"],
        "mae": report_data["current"]["mean_abs_error"],
        "r2": report_data["current"]["r2_score"],
        "error_std": report_data["current"]["error_std"],
    }

    error = report_data["error_normality"]
    params = {
        "user": os.getenv("POSTGRES_USER"),
        "pass": os.getenv("POSTGRES_PASSWORD"),
        "host": os.getenv("POSTGRES_HOST"),
        "port": os.getenv("POSTGRES_PORT"),
        "database": "metrics_training"
    }
    engine = create_engine('postgresql://%(user)s:%(pass)s@%(host)s:%(port)s/%(database)s' % params)

    # insert metrics
    pd.DataFrame(metrics, index=[0]).to_sql("metrics", engine, index=False, if_exists="replace")

    # insert error metrics
    for key, value in error.items():
        if isinstance(value, list):
            df = pd.DataFrame(value, columns=[key])
        else:
            df = pd.DataFrame([value], columns=[key])
        df.to_sql(key, engine, index=False, if_exists="replace")

    # qqdata
    qqdata = pd.concat([pd.Series(error["order_statistic_medians_x"]), pd.Series(error["order_statistic_medians_y"])], axis=1)
    qqdata.columns = ["Theorical Quantiles", "Sample Quantiles"]
    qqdata.to_sql("qqdata", engine, index=False, if_exists="replace")

@flow(name="generate_train_metrics")
def generate_train_metrics():
    model = load_model()
    training, testing = load_data()
    training["price_prediction"] = model.predict(training)
    testing["price_prediction"] = model.predict(testing)
    report, column_mapping = generate_evidently_report()
    extract_report_data(report, column_mapping, training, testing)