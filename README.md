# Azure ML Time-Series Clustering Model Training Pipeline Sample

Sample notebooks demonstrating how to create a model training pipeline using the Azure ML SDK. 

The model training pipeline `./ModelTrainingPipeline/AML_CreateTrainingPipeline.ipynb` loads raw data from an [AML-linked datastore](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-access-data), registers the raw dataset, splits and registers test and train datasets, trains a clustering model, and evaluates and registers the model and associated scalers. 

For demonstration purposes, we leverage a [cached time-series dataset from tslearn](https://tslearn.readthedocs.io/en/stable/gen_modules/datasets/tslearn.datasets.CachedDatasets.html) to train a k-means time series clustering model. The pipeline in this example pulls data from a registered datastore (named `timeseriesdatastore`) which retrieves the attached sample file `sample_data\time_series_data.csv`.

![Azure ML Pipeline Samples](img/AML_Pipelines.png?raw=true "Azure ML Pipeline Samples")

## Environment Setup
<b>Note:</b> Recommend running these notebooks on an Azure Machine Learning Compute Instance using the preconfigured `Python 3.6 - AzureML` environment.

To build and run the sample pipelines contained in `./ModelTrainingPipeline` and `./BatchInferencingPipeline` the following resources are required:
* Azure Machine Learning Workspace