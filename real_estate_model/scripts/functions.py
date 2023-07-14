import os
import pickle
import pysftp
from pathlib import Path
import pandas as pd

def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)
    
def dump_pickle(filename, obj):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)
    
def upload_file_to_sftp(filepath):
    hostname = os.getenv("STFP_HOST")
    username = os.getenv("SFTP_USER")
    password = os.getenv("SFTP_PASSWORD")
    port = int(os.getenv("SFTP_PORT"))
    cnopts = pysftp.CnOpts()
    cnopts.hostkeys = None
    filepath = Path(filepath)
    with pysftp.Connection(host=hostname, username=username, password=password, port=port, cnopts=cnopts) as sftp:
        with sftp.cd("/files"):
            sftp.put(filepath.absolute().as_posix())

def process_csv(dataframe: pd.DataFrame):
    dataframe.columns = ["transaction_date", "house_age", "distance_to_the_nearest_MRT_station", "number_of_convenience_stores", "latitude", "longitude", "house_price_of_unit_area"]
    dataframe.drop(columns=["transaction_date"], inplace=True)
    return dataframe.copy()