# haimtran 06 OCT 2022
# preprocess raw data

import argparse
import os
import numpy as np
import pandas as pd

# path
container_base_path = "/opt/ml/processing"
processed_train_dir = f"{container_base_path}/output"
os.makedirs(processed_train_dir, exist_ok=True)
local_data_path = f"{container_base_path}/data"


def test_data():
    data = np.random.randint(0, 100, (100, 4))
    pd.DataFrame(data).to_csv(
        f"{processed_train_dir}/test_data.csv", sep=",", index=False
    )
    return 1


def get_files():
    """
    os list files in a folder
    """
    files = os.listdir(local_data_path)
    files = [x for x in files if x.split(".")[-1] == "csv"]
    return files


def read_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processor", type=str, default="based processor"
    )
    parser.add_argument("--sagemaker", type=bool, default=True)
    params, _ = parser.parse_known_args()
    return params


def process_raw_data(file):
    """
    process raw data
    """
    # file path
    file_path = f"{local_data_path}/{file}"
    # read data
    df = pd.read_csv(
        file_path, header=0, decimal=",", low_memory=False
    )
    # replace nan by zero
    df.fillna(0, inplace=True)
    # extract ecg data
    data = df[["Ch5", "Ch6", "Ch7", "Ch8"]].values.astype(
        dtype=np.float32
    )
    # file name
    # save processed data
    pd.DataFrame(data).to_csv(
        f"{processed_train_dir}/clean_{file}",
        sep=",",
        index=False,
        header=False,
    )


# ============================RUN===========================
if __name__ == "__main__":
    # test data
    test_data()
    # parse arguments
    args = read_parameters()
    # parse paraemter
    if "PROCESSOR" in os.environ:
       pass
    else:
       os.environ["PROCESSOR"] = args.processor
    # process raw data
    files = get_files()
    for file in files:
       process_raw_data(file)
