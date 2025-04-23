# Xmartlabs Trash Classifier

Welcome to the impressive trash classifier developed by XmartLabs :-)

## Instructions

* Download the TrashNet dataset by running

```bash
./download_trashnet_dataset.sh
```

* Build the docker container:

```bash
./build.sh
```

* Start the docker container with the Jupyter Notebook:

```bash
./start.sh
```

* Follow the instructions to access the notebook on your browser


### Mac users
For Mac users you can opt to run the notebook outside Docker, using conda.
To do so, you need to install the required dependencies:

```bash
conda env create -f environment.yml
conda activate trash-classifier
```

Then you can run the notebook:

```bash
jupyter notebook
```


## Deploy

To deploy the trained model follow these steps:
* Copy the model to the GCS:
  * Run: `gsutil cp <model_path> <model_destination>`
  * model_destination should look like: `gs://<PREFIX>/garbage/<VERSION>`, where VERSION is an integer.
* Connect to the GCP Compute instance:
  * Run: `ssh <your_user>@<INSTANCE_IP>`
* Inside the GCP instance, start the TF Serving docker container pointing to the model location:
  * Run on GCP:
  ```
## Training via CLI

Once you have downloaded and prepared the TrashNet dataset, you can train the model using the provided command-line script and configuration:

```bash
python train.py --config config/config.yaml
```

This will:
- Load data and split according to `val_split`.
- Build the model architecture as configured.
- Log parameters and metrics to MLflow.
- Save the trained model under the `models/` directory.
  docker run --rm -d --name=garbage_model -it -p 8501:8501 \
        -e MODEL_NAME=garbage \
        -e MODEL_BASE_PATH="gs://<PREFIX>" \
        tensorflow/serving:2.11.0
  ```
