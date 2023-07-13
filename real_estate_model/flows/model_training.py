import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import optuna
from optuna.samplers import TPESampler

from prefect import flow, task
from prefect.deployments import Deployment

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)
    
def dump_pickle(filename, obj):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)
    
@task(name="read_data")
def read_data(filename: str):
    data = pd.read_csv(filename, index_col="No")
    data.columns = ["transaction_date", "house_age", "distance_to_the_nearest_MRT_station", "number_of_convenience_stores", "latitude", "longitude", "house_price_of_unit_area"]
    data.drop(columns=["transaction_date"], inplace=True)
    return data

@task(name="split_data")
def split_data(data: pd.DataFrame, target: str, generate_val_set: bool = True):
    if generate_val_set:
        # generate validation, train and test sets
        np.random.seed(42)
        val_index = np.random.randint(0, data.shape[0], int(data.shape[0]*0.2), dtype=np.int32)
        validation_set = data.iloc[val_index]
        X_validation = validation_set.drop(target, axis=1)
        y_validation = validation_set[target]
        train_data = data.drop(index=validation_set.index)
    else:
        train_data = data.copy()
        validation_set = None
    X_train, X_test, y_train, y_test = train_test_split(train_data.drop(target, axis=1), train_data[target], test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test, X_validation, y_validation

@task(name="train_models", log_prints=True)
def train_models(X_train, X_test, y_train, y_test, preprocessor, num_trials: int=10):

    def rf_optimize(trial):
        mlflow.set_experiment("random-forest-hyperparameter-tuning")
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 50, 1),
            'max_depth': trial.suggest_int('max_depth', 1, 20, 1),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10, 1),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4, 1),
            'random_state': 42,
            'n_jobs': -1
        }

        with mlflow.start_run():
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(**params))
            ])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mlflow.sklearn.log_model(pipeline, "rf")
            mlflow.log_params(params)
            mlflow.log_metric("test_rmse", rmse)
        return rmse

    def knn_optimize(trial):
        mlflow.set_experiment("knn-hyperparameter-tuning")
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 1, 20, 1),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'leaf_size': trial.suggest_int('leaf_size', 1, 50, 10),
            'algorithm': 'auto',
            'n_jobs': -1
        }

        with mlflow.start_run():
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', KNeighborsRegressor(**params))
            ])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mlflow.sklearn.log_model(pipeline, "knn")
            mlflow.log_params(params)
            mlflow.log_metric("test_rmse", rmse)
        return rmse
    
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(rf_optimize, n_trials=num_trials)
    study.optimize(knn_optimize, n_trials=num_trials)

@task(name="select_best_model", log_prints=True)
def select_best_model(X_train, y_train, X_validation, y_validation, preprocessor):
    client = MlflowClient(MLFLOW_TRACKING_URI)
    models = {
        "random-forest-hyperparameter-tuning": RandomForestRegressor,
        "knn-hyperparameter-tuning": KNeighborsRegressor
    }
    best_runs = []
    for experiment, model in models.items():
        experiment_metadata = client.get_experiment_by_name(experiment)
        runs = client.search_runs(
                experiment_ids=experiment_metadata.experiment_id,
                run_view_type=ViewType.ACTIVE_ONLY,
                max_results=5,
                order_by=["metrics.test_rmse ASC"]
            )
        
        for run in runs:
            best_runs.append({
                "model": model,
                "test_rmse": run.data.metrics["test_rmse"],
                "params": run.data.params
            })
    
    for run in best_runs:
        mlflow.set_experiment("best_model_selection")
        with mlflow.start_run():
            params = run["params"]
            for key, value in params.items():
                try:
                    params[key] = int(value)
                except:
                    pass

            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', run["model"](**params))
            ])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_validation)
            rmse = mean_squared_error(y_validation, y_pred, squared=False)
            mlflow.sklearn.log_model(pipeline, run["model"].__name__)
            mlflow.log_params(params)
            mlflow.log_metric("validation_rmse", rmse)

    # select the model with the lowest validation RMSE
    experiment = client.get_experiment_by_name("best_model_selection")
    best_run = client.search_runs(experiment.experiment_id, order_by=["metrics.validation_rmse ASC"])[0]
    best_run_id = best_run.info.run_id

    return model, y_pred, best_run_id

@task(name="register_model")
def register_model(X_validation, y_pred, model, best_run_id):
    # log model
    signature = infer_signature(X_validation, y_pred)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="sklearn-model",
        signature=signature,
        registered_model_name=MLFLOW_MODEL_NAME,
    )
    
    # promote to production
    client = MlflowClient(MLFLOW_TRACKING_URI)
    client.transition_model_version_stage(
        name = MLFLOW_MODEL_NAME,
        version = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["None"])[0].version,
        stage = "Production",
        archive_existing_versions=True   
    )

@flow
def model_training(data: str="./real_estate_model/data/Real estate.csv"):
    global MLFLOW_TRACKING_URI
    global MLFLOW_MODEL_NAME
    
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    data = read_data(data)
    preprocessor = ColumnTransformer(
        remainder='drop',
        transformers=[
            ('discretizer', KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform', subsample=None), ["number_of_convenience_stores"]),
            ('scaler', StandardScaler(), ["house_age", "distance_to_the_nearest_MRT_station", "latitude", "longitude"])   
        ]
    )
    X_train, X_test, y_train, y_test, X_validation, y_validation = split_data(data, target="house_price_of_unit_area", generate_val_set=True)
    train_models(X_train, X_test, y_train, y_test, preprocessor, num_trials=10)
    model, y_pred, best_run_id = select_best_model(X_train, y_train, X_validation, y_validation, preprocessor)
    register_model(X_validation, y_pred, model, best_run_id)