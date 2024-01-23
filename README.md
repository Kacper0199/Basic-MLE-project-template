# Basic-MLE-project-template
This repository provides a structured template for building a Machine Learning project. It includes a modular project structure with dedicated folders for data processing, training, inference. Below, you will find instructions on prerequisites, installation, structure, and how to run both training and inference using Docker or locally with Python.

---

# Prerequisites

Before running the code, ensure that Docker Desktop is installed on your machine. If Docker Desktop is not available, you can alternatively run the code locally using Python.

# Installation

Clone the repository:

```bash
git clone https://github.com/<YOUR-USERNAME>/{project_name}
```

To install requirements use:

```bash
cd project_name
pip install -r requirements.txt
```

# Project structure

This project follows a modular structure, with each folder dedicated to a specific functionality or task.

```
project_name
├── data                      # Data files used for training and inference, generated with data_process.py script
│   ├── data_iris.csv
│   ├── inference_iris.csv
│   └── train_iris.csv
├── data_process              # Scripts used for data uploading and splitting into training and inference parts
│   └── data_process.py         
├── inference                 # Scripts and Dockerfiles used for inference
│   ├── Dockerfile
│   └── inference.py
├── logs                      # Folder with various model version logs
│   └── model_logs
├── models                    # Folder where trained models are stored (e.g. best_model.ckpt, decode_labels.npy, NNClassifier.pth)
│   └── various model files
├── results                   # Folder where test results are stored
│   └── results.csv
├── training                  # Scripts and Dockerfiles used for training
│   ├── Dockerfile
│   └── train.py
├── README.md
├── requirements.txt          # All requirements for the project
├── settings.json             # All configurable parameters and settings
└── utils.py                  # Utility functions and classes that are used in scripts
```

# Settings

The project's configuration is managed through the settings.json file.

**General settings:**
- status: Specifies the current status, e.g. "test".
- data_url: URL for the dataset. Example includes the Iris Flower Data Set.
- data_dir: Directory to store the dataset.
- models_dir: Directory to save trained models.
- results_dir: Directory to store output results.
- logs_dir: Directory to store log files.
- decode_labels: Subdirectory for label decoding.
- target_feature: Target feature for training and inference (e.g., "species").
- random_state: Seed for random number generation.

**Training settings:**
- csv_file: Training dataset file (e.g. "train_iris.csv").
- batch_size: Batch size for training.
- hidden_layer_size: Size of the hidden layer in the neural network.
- test_size: Percentage of data to use for testing during training.
- max_epochs: Maximum number of training epochs.

**Inference Settings**
- csv_file: Inference dataset file (e.g. "inference_iris.csv").
- results_file: File to store inference results (e.g. "results.csv").
- model: Trained model file for inference (e.g. "NNClassifier.pth").

# Data preparation

To generate the data required for training the model and testing the inference, utilize the script located at `data_process/data_process.py`. This script is dedicated to handling the responsibility of data generation, following the principle of separating concerns. Execute this script to ensure the generation of data necessary for subsequent steps in the workflow.

```bash
python3 data_process/data_process.py
```

By running this command, the following actions are performed:

- The script downloads data from the webpage and saves the complete dataset into the `data` directory as a .csv file. If the `data` directory does not exist, it is created.

- Subsequently, the dataset is split into training and inference portions based on the `test_size` parameter specified in the settings.json file.

- Finally, both the training and inference datasets are saved into the `data` directory as .csv files with names defined in the `settings.json` file.

# Run training with Docker

- Build the training Docker image.

```bash
docker build -f training/Dockerfile --build-arg settings_name=settings.json -t training_image .
```

- Run the Docker container in detached mode.

```bash
docker run -dit training_image
```

- Move the necessary files from the directory inside the Docker container `/app/models` to the local machine using the following command or retrieve them from Docker Desktop. Make sure to obtain the container ID from Docker Desktop.

```bash
docker cp <container_id>:/app/models/<file_name> ./models
```

# Run training locally

Run the `train.py` script using:

```bash
python3 training/train.py
```

# Run inference with Docker

- Build the inference Docker image.

```bash
docker build -f inference/Dockerfile --build-arg settings_name=settings.json -t inference_image .
```

- Run the Docker container in detached mode.

```bash
docker run -dit inference_image
```

- Subsequently, verify that the results are located within the results directory of your inference container. Afterward, you can download that folder to the local repository.

# Run inference locally

Run the `inference.py` script using:

```bash
python3 inference/inference.py
```
