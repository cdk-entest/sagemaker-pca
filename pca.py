# haimtran 01 OCT 2022
# sagemaker pca test
# use sagemaker estimator base
# content_type=text/csv; label_size=0
# add processing raw data by processor

import json
import os
from time import strftime
import boto3
import sagemaker
from sagemaker import image_uris
from sagemaker.inputs import TrainingInput
from sagemaker.processing import (
    ProcessingInput,
    Processor,
    ProcessingOutput,
)


if "SAGEMAKER_ROLE" in os.environ and "BUCKET_NAME" in os.environ:
    pass
else:
    with open("config.json", "r", encoding="utf-8") as file:
        config = json.load(file)
        os.environ["SAGEMAKER_ROLE"] = config["SAGEMAKER_ROLE"]
        os.environ["SAGEMAKER_BUCKET"] = config["SAGEMAKER_BUCKET"]
        os.environ["REGION"] = config["REGION"]

container_base_path = "/opt/ml/processing"
data_input_path = f's3://{os.environ["SAGEMAKER_BUCKET"]}/pca-raw-data/171A_raw.csv'
data_output_path = f's3://{os.environ["SAGEMAKER_BUCKET"]}/pca-processed-data'
code_input_path = f's3://{os.environ["SAGEMAKER_BUCKET"]}/code/process_raw_data.py'


def process_data():
    """
    process data by sagemaker processor
    """
    image_uri = image_uris.retrieve(
        framework="sklearn",
        region=os.environ["REGION"],
        version="0.23-1",
    )
    processor = Processor(
        role=os.environ["SAGEMAKER_ROLE"],
        image_uri=image_uri,
        instance_count=1,
        instance_type="ml.m4.xlarge",
        entrypoint=[
            "python",
            f"{container_base_path}/input/process_raw_data.py",
            "--processor=base-processor",
        ],
    )
    processor.run(
        job_name=f'processor-{strftime("%Y-%m-%d-%H-%M-%S")}',
        inputs=[
            ProcessingInput(
                source=data_input_path,
                destination=f"{container_base_path}/data/",
            ),
            ProcessingInput(
                source=code_input_path,
                destination=f"{container_base_path}/input/",
            ),
        ],
        outputs=[
            ProcessingOutput(
                source=f"{container_base_path}/output/train",
                destination=f"{data_output_path}/train",
                output_name="train",
            )
        ],
    )


def train_pca_model():
    # pca image uri
    image_uri = image_uris.retrieve(
        framework="pca",
        region=os.environ["REGION"],
        instance_type="t2.medium",
    )
    # pca estimator
    pca = sagemaker.estimator.Estimator(
        role=config["SAGEMAKER_ROLE"],
        image_uri=image_uri,
        instance_count=1,
        instance_type="ml.c4.xlarge",
    )
    # set pca parameters
    pca.set_hyperparameters(
        feature_dim=4,
        num_components=3,
        mini_batch_size=200,
    )
    # train the model
    pca.fit(
        inputs={
            "train": TrainingInput(
                content_type="text/csv;label_size=0",
                s3_data=f"s3://{config['SAGEMAKER_BUCKET']}/ecg/test_small.csv",
            )
        }
    )
    # return
    return None


if __name__ == "__main__":
    # upload process_raw_data.py to s3
    boto3.resource("s3").Bucket(os.environ["SAGEMAKER_BUCKET"]).upload_file(
        "./process_raw_data.py", "code/process_raw_data.py"
    )
    # processing job
    process_data()
    # train_pca_model()
