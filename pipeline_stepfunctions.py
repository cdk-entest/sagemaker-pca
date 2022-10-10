# haimtran 10 OCT 2022
# sagemaker pipeline by using stepfunctions
import os
import json
import stepfunctions
from stepfunctions.inputs import ExecutionInput
from sagemaker import image_uris
from sagemaker.processing import Processor
from sagemaker.processing import ProcessingInput, ProcessingOutput

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
input_data_uri = f's3://{os.environ["SAGEMAKER_BUCKET"]}/code/process_raw_data.py'
input_code_uri = f's3://{os.environ["SAGEMAKER_BUCKET"]}/pca-raw-data'
processing_output_path = f's3://{os.environ["SAGEMAKER_BUCKET"]}/pca-processed-data'

# execution input for the statem machine
execution_input = ExecutionInput(
    schema={
        "PreprocessingJobName": str,
        "TrainingJobName": str,
        "LambdaFunctionName": str,
        "ModelName": str
    }
)

# create processing step
def create_processing_step() -> stepfunctions.steps.ProcessingStep:
    """
    create processing step which process raw data
    """
    # image uri
    image_uri =  image_uris.retrieve(
        framework="sklearn",
        region=os.environ["REGION"],
        version="0.23-1"
    )
    processor = Processor(
        role=os.environ["SAGEMAKER_ROLE"],
        image_uri=image_uri,
        instance_count=1,
        instance_type="ml.m4.xlarge",
        entrypoint=[
            "python",
            f"{container_base_path}/input/process_raw_data.py",
        ]
    )
    step_process = stepfunctions.steps.ProcessingStep(
        state_id="PreprocessingData",
        processor=processor,
        job_name=execution_input["PreprocessingJobName"],
        inputs=[ 
           ProcessingInput(
            input_name="train-data-input",
            source=input_data_uri,
            destination="/opt/ml/processing/input"
           ),
           ProcessingInput(
            input_name="train-code-input",
            source=input_code_uri,
            destination="/opt/ml/processing/code"
           )
        ],
        outputs = [
            ProcessingOutput(
                output_name="train-data-output",
                source="/opt/ml/processiing/train",
                destination=processing_output_path
            ),
            ProcessingOutput(
                output_name="validation-data-output",
                source="/opt/ml/processing/validation",
                destination=processing_output_path
            )
        ]
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
    definition = stepfunctions.steps.Chain(
        [processing_step]
    )
    workflow = stepfunctions.workflow.Workflow(
        name="StepFunctionWorkFlowDemo",
        definition=definition,
        role=os.environ["SAGEMAKER_ROLE"],
        execution_input=execution_input
    )
    return workflow


if __name__=="__main__":
    ml_workflow = create_workflow()
    print(ml_workflow.definition)