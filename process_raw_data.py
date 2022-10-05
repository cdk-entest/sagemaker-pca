import argparse
import os
import numpy as np
import pandas as pd

# path
container_base_path = "/opt/ml/processing"
processed_train_dir = f"{container_base_path}/output/train"
os.makedirs(processed_train_dir, exist_ok=True)
local_data_path = f"{container_base_path}/data/171A_raw.csv"


def read_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processor", type=str, default="based processor")
    parser.add_argument("--sagemaker", type=bool, default=True)
    params, _ = parser.parse_known_args()
    return params


# ============================RUN===========================
args = read_parameters()
# parse paraemter
if "PROCESSOR" in os.environ:
    pass
else:
    os.environ["PROCESSOR"] = args.processor
# read data
df = pd.read_csv(local_data_path, header=0, decimal=",", low_memory=False)
# replace nan by zero
df.fillna(0, inplace=True)
# extract ecg data
data = df[["Ch5", "Ch6", "Ch7", "Ch8"]].values.astype(dtype=np.float32)
# save processed data
pd.DataFrame(data).to_csv(
    f'{processed_train_dir}/{os.environ["PROCESSOR"]}_clean_data.csv',
    sep=",",
    index=False,
    header=False,
)
