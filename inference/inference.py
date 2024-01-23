import numpy as np
import pandas as pd
import os
import sys
import json
import logging
import io
import pickle
import torch
import torchmetrics
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from time import perf_counter as stoper
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define directories
ROOT_DIR = os.path.abspath(os.path.dirname( __file__ ))
ROOT_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(ROOT_DIR)

from utils import get_project_dir, configure_logging, join_path

DATA_DIR = os.path.abspath(join_path(ROOT_DIR, "data"))
if not os.path.exists(DATA_DIR):
    raise Exception("There is no data directory in project.")

# Load configuration settings from JSON
CONF_FILE = os.path.join(ROOT_DIR, "settings.json")
logger.info("Loading configuration settings from JSON...")
with open(CONF_FILE, "r") as file:
    configuration = json.load(file)

# Define paths
logger.info("Defining paths...")
DATA_DIR = get_project_dir(configuration["general"]["data_dir"])
MODEL_DIR = os.path.join(ROOT_DIR, configuration["general"]["models_dir"])

RESULTS_DIR = os.path.abspath(join_path(ROOT_DIR, "results"))
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

INFERENCE_PATH = os.path.join(DATA_DIR, configuration["inference"]["csv_file"])
TARGET_FEATURE = configuration["general"]["target_feature"]
DECODE_LABELS = configuration["general"]["decode_labels"]
DECODE_LABELS_PATH = os.path.join(MODEL_DIR, DECODE_LABELS+".npy")\

RESULTS_PATH = os.path.join(RESULTS_DIR, configuration["inference"]["results_file"])
MODEL_PATH = os.path.join(MODEL_DIR, configuration["inference"]["model"])
INF_MODEL_PATH = os.path.join(RESULTS_DIR, configuration["inference"]["model"])


# Define the same class as in the train.py file 
# to later load the model: model = torch.load(...)
class NNClassifier(pl.LightningModule):
    def __init__(self, input_size, output_size, hidden_size, dropout_rate=0.5):
        super(NNClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.metric_training = torchmetrics.F1Score(task="multiclass", num_classes=output_size)
        self.metric_validation = torchmetrics.F1Score(task="multiclass", num_classes=output_size)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = F.relu(self.dropout1(self.bn1(self.fc1(x))))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, target)
        f1_score_train = self.metric_training(outputs.argmax(dim=1), target)
        self.log("train_loss", loss, logger=True, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_f1_score", f1_score_train, logger=True, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, target)
        f1_score_val = self.metric_validation(outputs.argmax(dim=1), target)
        self.log("val_loss", loss, logger=True, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val_f1_score", f1_score_val, logger=True, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=0.001)


def get_dataloader(path):
    """Load data from the specified CSV file and preprocess it for inference."""
    logging.info("Loading data...")
    df = pd.read_csv(path, encoding="utf-16")
    y = df[TARGET_FEATURE].astype("category").cat.codes.values
    X = df.drop(columns=TARGET_FEATURE).values
    X = torch.tensor(X, dtype=torch.float32)
    
    # Wrap X and y in a TensorDataset
    dataset = TensorDataset(X, torch.tensor(y))

    # Use DataLoader to create batches
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=1)

    return dataloader


def load_model():
    """Load the trained model and its corresponding checkpoint."""
    logging.info("Loading trained model...")

    if MODEL_DIR is None:
        raise ValueError("Please provide a valid folder_path")

    folder_path = Path(MODEL_DIR)
    
    model_files = list(folder_path.rglob("*.pth"))
    checkpoint_files = list(folder_path.rglob("*ckpt"))

    if not model_files or not checkpoint_files:
        raise FileNotFoundError("No models or checkpoints found. Train a model first")

    latest_model_path = max(model_files, key=os.path.getctime)
    latest_checkpoint_path = max(checkpoint_files, key=os.path.getctime)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if latest_model_path.parts[0].startswith("http") or latest_checkpoint_path.parts[0].startswith("http"):
        raise NotImplementedError("Only local models can be used")

    # Load files
    model = torch.load(latest_model_path, map_location=device)
    checkpoint = torch.load(latest_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    return model


def predict(model, dataloader):
    """Make predictions using the provided model on the given dataloader."""
    logging.info("Model is making predictions...")
    t_start = stoper()
    model.eval()
    prediction_list = []
    true_labels = []

    for batch in dataloader:
        inputs, labels = batch
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)
        prediction_list.extend(predictions.detach().numpy())
        true_labels.extend(labels.numpy())

    t_stop = stoper()
    logging.info(f"Inference time [s]: {(t_stop - t_start):.2f}")

    label_decode = np.load(DECODE_LABELS_PATH, allow_pickle=True).item()
    results = np.vectorize(label_decode.get)(prediction_list)
    logging.info(f"Saving results to csv...")
    pd.Series(results).to_csv(RESULTS_PATH, index=False, encoding="utf-16")

    f1 = f1_score(true_labels, prediction_list, average="micro")
    logging.info(f"Inference F1 Score: {f1:.2f}")


if __name__ == "__main__":
    model = load_model()
    dataloader = get_dataloader(INFERENCE_PATH)
    predict(model, dataloader)
