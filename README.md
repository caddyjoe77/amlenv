# Azure ML GPU Environment

A ready-to-use Azure Machine Learning environment with GPU support for both batch and real-time inference, including Jupyter notebooks.

## Prerequisites

1. Azure subscription
2. Python 3.8+
3. Azure CLI installed and logged in (`az login`)
4. Azure ML CLI extension (`az extension add -n ml`)

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Azure credentials:
   ```
   AZURE_SUBSCRIPTION_ID=your_subscription_id
   AZURE_RESOURCE_GROUP=your_resource_group
   ```

## Getting Started

1. Run the setup script to create the Azure ML workspace and resources:
   ```bash
   python setup_aml.py
   ```

2. The script will output a link to your Azure ML Studio. Open it in your browser.

3. In Azure ML Studio:
   - Go to "Compute" > "Compute Instances"
   - Start your Jupyter notebook instance
   - Upload the notebooks from the `notebooks` directory
   - Run the GPU test notebook to verify your setup

## Features

- GPU-accelerated compute cluster (NC6s_v3 with V100 GPU)
- Jupyter notebook instance for interactive development
- Pre-configured PyTorch environment with CUDA support
- Sample notebook for GPU testing

## Environment

The environment includes:
- PyTorch 2.0.1 with CUDA 11.7
- Common ML libraries (numpy, pandas, scikit-learn)
- Jupyter and IPython for interactive development
- FastAPI for serving models

## Next Steps

- Upload your training scripts to the `src` directory
- Use the GPU cluster for model training
- Deploy models as batch or real-time endpoints
- Scale your compute resources as needed

## Clean Up

To avoid unnecessary charges, remember to stop or delete resources when not in use:
```bash
az ml compute stop --name jupyter-notebook
az ml compute stop --name gpu-cluster
```

## License

MIT
