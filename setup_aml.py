from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import (
    Environment,
    BuildContext,
    ComputeInstance,
    AmlCompute
)
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group = os.getenv("AZURE_RESOURCE_GROUP")
workspace_name = "gpu-ml-workspace"
location = "eastus"  # Change to your preferred region

def create_workspace():
    # Get credentials
    credential = DefaultAzureCredential()
    
    # Create ML Client
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
    )
    
    # Create or get workspace
    try:
        workspace = ml_client.workspaces.get(workspace_name)
        print(f"Found existing workspace: {workspace.name}")
    except Exception:
        print("Creating new workspace...")
        workspace = ml_client.workspaces.begin_create(
            workspace_name=workspace_name,
            location=location,
            display_name="GPU ML Workspace",
            description="Workspace for GPU-accelerated ML workloads"
        ).result()
        print(f"Created workspace: {workspace.name}")
    
    return ml_client

def setup_compute(ml_client):
    # Create GPU compute cluster
    gpu_compute_name = "gpu-cluster"
    
    try:
        # Check if compute exists
        ml_client.compute.get(gpu_compute_name)
        print(f"Compute {gpu_compute_name} already exists.")
    except Exception:
        print("Creating GPU compute cluster...")
        gpu_compute = AmlCompute(
            name=gpu_compute_name,
            type="amlcompute",
            size="Standard_NC6s_v3",  # 1x V100 GPU
            min_instances=0,
            max_instances=4,
            idle_time_before_scale_down=1800,  # 30 minutes
            tier="dedicated",
        )
        ml_client.compute.begin_create_or_update(gpu_compute).result()
        print(f"Created GPU compute: {gpu_compute_name}")
    
    # Create Jupyter Notebook instance
    try:
        notebook_instance_name = "jupyter-notebook"
        ml_client.compute.get(notebook_instance_name)
        print(f"Notebook instance {notebook_instance_name} already exists.")
    except Exception:
        print("Creating Jupyter Notebook instance...")
        notebook_instance = ComputeInstance(
            name=notebook_instance_name,
            type="computeinstance",
            size="Standard_DS3_v2",  # General purpose instance
            description="Jupyter Notebook instance with GPU support",
        )
        ml_client.compute.begin_create_or_update(notebook_instance).result()
        print(f"Created Jupyter Notebook instance: {notebook_instance_name}")

def create_environment(ml_client):
    env_name = "gpu-inference-env"
    
    # Create environment from Docker context
    env_docker_context = Environment(
        name=env_name,
        description="Environment for GPU-accelerated inference",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.3-cudnn8-ubuntu20.04:latest",
        conda_file="environment.yml",
    )
    
    try:
        ml_client.environments.create_or_update(env_docker_context)
        print(f"Created/updated environment: {env_name}")
    except Exception as e:
        print(f"Error creating environment: {str(e)}")

if __name__ == "__main__":
    if not all([subscription_id, resource_group]):
        print("Please set AZURE_SUBSCRIPTION_ID and AZURE_RESOURCE_GROUP environment variables.")
        print("Create a .env file or export them in your shell.")
    else:
        ml_client = create_workspace()
        setup_compute(ml_client)
        create_environment(ml_client)
        print("\nSetup complete! You can now access your Azure ML Studio at:")
        print(f"https://ml.azure.com/workspaces/{workspace_name}/overview")
