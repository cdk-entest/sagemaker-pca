# haimtran 10 OCT 2022
# sagemaker pipeline by using stepfunctions
# debug by check workflow definition in aws
# and create a new workflow name
import os
import json
import uuid
import boto3
import stepfunctions
from stepfunctions.inputs import ExecutionInput
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor

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

# create processing step


def create_processing_step() -> stepfunctions.steps.ProcessingStep:
    """
    create processing step which process raw data
    """
    processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type="ml.m5.xlarge",
        instance_count=1,
        base_job_name="sklearn-abalone-process",
        role=os.environ["SAGEMAKER_ROLE"],
    )

    step_process = stepfunctions.steps.ProcessingStep(
        state_id="PreprocessingData",
        processor=processor,
        job_name=execution_input["PreprocessingJobName"],
        inputs=[
            ProcessingInput(
                input_name="train-data-input",
                source=input_data_uri,
                destination="/opt/ml/processing/data",
            ),
            ProcessingInput(
                input_name="train-code-input",
                source=input_code_uri,
                destination="/opt/ml/processing/input",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/output",
                destination=processing_output_path,
            )
        ],
        container_entrypoint=[
            "python3",
            "/opt/ml/processing/input/process_raw_data.py",
        ],
    )
    return step_process


# create training step

# create model step

# deploy model as sagemaker endpoint

# create a workflow by stepfunctions
def create_workflow() -> stepfunctions.workflow.Workflow:
    """
    create workflow by stepfunctions
    """
    # processing step
    processing_step = create_processing_step()
    # workflow
    definition = stepfunctions.steps.Chain([processing_step])
    print(os.environ["SAGEMAKER_ROLE"])
    workflow = stepfunctions.workflow.Workflow(
        name="StepFunctionWorkFlowDemo001",
        definition=definition,
        role=os.environ["SAGEMAKER_ROLE"],
        execution_input=execution_input,
    )
    return workflow


if __name__ == "__main__":
    # upload code to s3
    boto3.resource("s3").Bucket(
        os.environ["SAGEMAKER_BUCKET"]
    ).upload_file("./process_raw_data.py", "code/process_raw_data.py")
    ml_workflow = create_workflow()
    print(ml_workflow.definition)
    ml_workflow.create()
    ml_workflow.execute(
        inputs={
            "PreprocessingJobName": f"PreprocessingJobName{uuid.uuid4()}",
            "TrainingJobName": f"TrainingJobName{uuid.uuid4()}",
            "LambdaFunctionName": "LambdaRecordModelName",
            "ModelName": f"ModelName{uuid.uuid4()}",
        }
    )
    # ml_workflow.delete()
