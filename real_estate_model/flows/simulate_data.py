import os
from prefect import task, flow
import awswrangler as wr
wr.config.s3_endpoint_url = os.getenv("AWS_ENDPOINT_URL")

from real_estate_model.utils import functions

@flow(name="simulate_data")
def simulate_data(upload_to_s3: bool = False):
    # generate simulation data
    unprocessed_data = functions.read_data("./real_estate_model/data/Real estate.csv")
    simulation_data = unprocessed_data.drop(columns=["Y house price of unit area"]).sample(100).reset_index(drop=True).copy()
    for col in simulation_data.columns[1:]:
        simulation_data[col] = unprocessed_data[col].sample(simulation_data.shape[0]).values
    if upload_to_s3:
        wr.s3.to_csv(simulation_data, "s3://files/simulation_data.csv", index=False)
    return simulation_data