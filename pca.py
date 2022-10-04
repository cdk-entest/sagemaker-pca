# haimtran 01 OCT 2022
# sagemaker pca test
# use sagemaker estimator base
# content_type=text/csv; label_size=0

import json
import sagemaker
from sagemaker import image_uris
from sagemaker.inputs import TrainingInput


def get_config():
    with open("role.json", 'r') as file:
        config = json.load(file)
    return config


def train_pca_model():
    # pca image uri
    image_uri = image_uris.retrieve(
        framework="pca",
        region="ap-southeast-1",
        instance_type="t2.medium",
    )
    # pca estimator
    pca = sagemaker.estimator.Estimator(
        role=config['SAGEMAKER_ROLE'],
        image_uri=image_uri,
        instance_count=1,
        instance_type='ml.c4.xlarge',
    )
    # set pca parameters
    pca.set_hyperparameters(
        feature_dim=4,
        num_components=3,
        mini_batch_size=200,
    )
    # train the model
    pca.fit(inputs={'train': TrainingInput(
        content_type="text/csv;label_size=0",
        s3_data=f"s3://{config['SAGEMAKER_BUCKET']}/ecg/test_small.csv")})
    # return
    return None


if __name__ == "__main__":
    config = get_config()
    train_pca_model()
