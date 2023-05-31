# haimtran 10 OCT 2022
# sagemaker pipeline by using stepfunctions
# debug by check workflow definition in aws
# and create a new workflow name
# this script can be run from a CodeBuild
# invoke for data format samed as for training
import os
import json
import uuid
import boto3
from sagemaker.estimator import sagemaker
import stepfunctions
from stepfunctions.inputs import ExecutionInput
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from stepfunctions.steps import ProcessingStep

# environment variable and configuration
if "SAGEMAKER_ROLE" in os.environ and "BUCKET_NAME" in os.environ:
    pass
else:
    with open("config.json", "r", encoding="utf-8") as file:
        config = json.load(file)
        os.environ["SAGEMAKER_ROLE"] = config["SAGEMAKER_ROLE"]
        os.environ["SAGEMAKER_BUCKET"] = config["SAGEMAKER_BUCKET"]
        os.environ["REGION"] = config["REGION"]
# parameter
container_base_path = "/opt/ml/processing"
input_code_uri = (
    f's3://{os.environ["SAGEMAKER_BUCKET"]}/code/process_raw_data.py'
)
input_data_uri = (
    f's3://{os.environ["SAGEMAKER_BUCKET"]}/pca-raw-data/'
)
processing_output_path = (
    f's3://{os.environ["SAGEMAKER_BUCKET"]}/pca-processed-data'
)

# execution input for the statem machine
execution_input = ExecutionInput(
    schema={
        "PreprocessingJobName": str,
        "TrainingJobName": str,
        "LambdaFunctionName": str,
        "ModelName": str,
    }
)


def create_processing_step() -> stepfunctions.steps.ProcessingStep:
    """
    create processing step which process raw data
    """
    processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type="ml.m5.xlarge",
        instance_count=1,
        base_job_name="pca-sklearn-processsor",
        role=os.environ["SAGEMAKER_ROLE"],
    )

    step_process = ProcessingStep(
        state_id="PreprocessingData",
        processor=processor,
        job_name=execution_input["PreprocessingJobName"],
        inputs=[
            ProcessingInput(
                input_name="train-data-input",
                source=input_data_uri,
                destination=f"{container_base_path}/data",
            ),
            ProcessingInput(
                input_name="train-code-input",
                source=input_code_uri,
                destination=f"{container_base_path}/input",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source=f"{container_base_path}/output",
                destination=processing_output_path,
            )
        ],
        container_entrypoint=[
            "python3",
            f"{container_base_path}/input/process_raw_data.py",
        ],
    )
    return step_process


# create training step
def create_training_step() -> stepfunctions.steps.TrainingStep:
    """
    create a training step
    """
    # get sklearn image uri
    image_uri = sagemaker.estimator.image_uris.retrieve(
        framework="pca",
        region=os.environ["REGION"],
        version="0.23-1",
        instance_type="ml.m4.xlarge",
    )
    # create an estimator
    estimator = sagemaker.estimator.Estimator(
        image_uri=image_uri,
        instance_type="ml.m4.xlarge",
        instance_count=1,
        # output_path="",
        role=os.environ["SAGEMAKER_ROLE"],
    )
    # set hyperparameter
    estimator.set_hyperparameters(
        feature_dim=4, num_components=3, mini_batch_size=200
    )
    # create a train step
    step_train = stepfunctions.steps.TrainingStep(
        job_name=execution_input["TrainingJobName"],
        state_id="PCATrainingStep",
        estimator=estimator,
        data={
            "train": sagemaker.TrainingInput(
                content_type="text/csv;label_size=0",
                s3_data=f's3://{os.environ["SAGEMAKER_BUCKET"]}/pca-processed-data',
            )
        },
    )
    # return
    return step_train


def create_model_step(
    training_step: stepfunctions.steps.TrainingStep,
) -> stepfunctions.steps.ModelStep:
    """
    create a model step
    """
    model_step = stepfunctions.steps.ModelStep(
        state_id="PcaModelStep",
        model=training_step.get_expected_model(),
        model_name=execution_input["ModelName"],
    )
    return model_step


def create_workflow() -> stepfunctions.workflow.Workflow:
    """
    create workflow by stepfunctions
    """
    # processing step
    processing_step = create_processing_step()
    # training step
    training_step = create_training_step()
    # model step
    model_step = create_model_step(training_step)
    # workflow
    definition = stepfunctions.steps.Chain(
        [processing_step, training_step, model_step]
    )
    print(os.environ["SAGEMAKER_ROLE"])
    workflow = stepfunctions.workflow.Workflow(
        name="StepFunctionWorkFlowDemo001",
        definition=definition,
        role=os.environ["SAGEMAKER_ROLE"],
        execution_input=execution_input,
    )
    return workflow


def create_end_point(model_name, endpoint_name):
    """
    create an endpoint
    """

    # sagemaker client
    sg = boto3.client("sagemaker")
    # create endpoint configuration
    endpoint_config_name = "PCAEndpointConfigName"
    #
    try:
        sg.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    "VariantName": "test",
                    "ModelName": model_name,
                    "InstanceType": "ml.m4.xlarge",
                    "InitialInstanceCount": 1,
                }
            ],
        )
    except:
        print("configuration already existed")
        pass
    # create an endpoint
    endpoint = sg.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name,
    )
    return endpoint


def test_boto3_invoke_endpoint(endpoint_name):
    """
    test the pca endpoint
    """
    client = boto3.client("sagemaker-runtime")
    resp = client.invoke_endpoint(
        EndpointName=endpoint_name,
        # same format as used for training
        ContentType="text/csv",
        Body="1,2,3,4\n1,2,3,4\n2,3,4,5",
        # Body=bytes('1,2,3,4', 'utf-8')
    )
    print(json.loads(resp["Body"].read()))
    return resp


def upload_data_code_to_s3():
    """
    """
    s3_client = boto3.client("s3")
    s3_client.upload_file(
        "./process_raw_data.py", 
        os.environ["SAGEMAKER_BUCKET"], 
        "code/process_raw_data.py"
    )
    s3_client.upload_file(
        "./171A_raw.csv",
        os.environ["SAGEMAKER_BUCKET"],
        "pca-raw-data/171A_raw.csv"
    )


if __name__ == "__main__":
    # upload data and code to s3
    # upload_data_code_to_s3()
    # create workflow 
    ml_workflow = create_workflow()
    print(ml_workflow.definition)
    # run workflow 
    ml_workflow.create()
    ml_workflow.execute(
       inputs={
           "PreprocessingJobName": f"PreprocessingJobName{uuid.uuid4()}",
           "TrainingJobName": f"TrainingJobName{uuid.uuid4()}",
           "LambdaFunctionName": "LambdaRecordModelName",
           "ModelName": f"PCAModel{uuid.uuid4()}",
       }
    )
    # ml_workflow.delete()
    # create_end_point(
    #    model_name="pcalmodel161fb49a-4865-4740-8f3f-2945e40fc8b9",
    #    endpoint_name="pca-endpoint-test",
    # )
    # test_boto3_invoke_endpoint("pca-endpoint-test")
