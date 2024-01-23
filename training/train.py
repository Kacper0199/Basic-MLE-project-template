import numpy as np
import pandas as pd
import os
import sys
import json
import logging
import hashlib
import pickle
import requests
import torch
import torchmetrics
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from time import perf_counter as stoper
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define directories
# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
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
TRAIN_PATH = os.path.join(DATA_DIR, configuration["train"]["csv_file"])
MODEL_DIR = os.path.join(ROOT_DIR, configuration["general"]["models_dir"])
MODEL_FILE = os.path.join(MODEL_DIR, configuration["inference"]["model"])
LOGS_DIR = os.path.join(ROOT_DIR, configuration["general"]["logs_dir"])
if not os.path.exists(os.path.abspath(LOGS_DIR)):
    os.makedirs(LOGS_DIR)

# Define training parameters
TARGET_FEATURE = configuration["general"]["target_feature"]
RANDOM_STATE = configuration["general"]["random_state"]
DECODE_LABELS = configuration["general"]["decode_labels"]
HIDDEN_LAYER_SIZE = configuration["train"]["hidden_layer_size"]
MAX_EPOCHS = configuration["train"]["max_epochs"]
TEST_SIZE = configuration["train"]["test_size"]
BATCH_SIZE = configuration["train"]["batch_size"]


class Training():
    """
    A class for handling data preparation, label encoding, and splitting for training.

    Attributes:
    - df: pandas DataFrame, the main dataset.
    - label_mapping: dict, mapping of encoded labels to original labels.
    - model_input_len: int, length of the input features for the model.
    - model_output_len: int, number of unique classes in the target variable.

    Methods:
    - data_preparation(train_path): Reads data from a CSV file, encodes labels, and prepares DataLoader objects for training and validation.
    - label_handler(): Encodes labels using sklearn's LabelEncoder and saves the label mapping for future use.
    - split_train(data): Splits the dataset into training and testing sets, and sets the lengths for model input and output.

    """
    
    def __init__(self):
        self.df = None
        self.label_mapping = {}
        self.model_input_len = -1
        self.model_output_len = -1

    def data_preparation(self, train_path):
        logging.info("Starting data preparation...")
        self.df = pd.read_csv(train_path, encoding="utf-16")
        self.label_handler()
        X_train, X_test, y_train, y_test = self.split_train(self.df)

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)

        X_val_t = torch.tensor(X_test, dtype=torch.float32)
        y_val_t = torch.tensor(y_test, dtype=torch.long)

        train_loader = DataLoader(dataset=TensorDataset(X_train_t, y_train_t), 
                                  batch_size=BATCH_SIZE, 
                                  shuffle=True, 
                                  num_workers=2, 
                                  persistent_workers=True)
        val_loader = DataLoader(dataset=TensorDataset(X_val_t, y_val_t), 
                                batch_size=BATCH_SIZE, 
                                num_workers=2, 
                                persistent_workers=True)
    
        return train_loader, val_loader

    def label_handler(self):
        logging.info("Handling labels...")
        label_encoder = LabelEncoder()
        self.df[TARGET_FEATURE] = label_encoder.fit_transform(self.df[TARGET_FEATURE])

        # Map original label to the new encoded label
        for label, original_label in zip(range(len(label_encoder.classes_)), 
                                         label_encoder.classes_):
            self.label_mapping[label] = original_label

        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        
        # Save mapped labels for further purpose
        decode_labels = os.path.join(MODEL_DIR, DECODE_LABELS)
        np.save(decode_labels, self.label_mapping)

    def split_train(self, data):
        logging.info("Getting train and test data...")
        X = data.drop(columns=TARGET_FEATURE).values
        y = data[TARGET_FEATURE].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, 
                                                            stratify=y, random_state=RANDOM_STATE)
        self.model_input_len = X_train.shape[1]
        self.model_output_len = len(np.unique(y))

        return X_train, X_test, y_train, y_test


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


def train_model(train_loader, val_loader, hidden_layer_size, max_epochs, input_size, output_size):
    """Train a neural network model on Iris dataset."""
    logging.info("Training...")
    t_start = stoper()
    model = NNClassifier(input_size=input_size, 
                         output_size=output_size, 
                         hidden_size=hidden_layer_size)

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Create callbacks
    early_stop_callback = EarlyStopping(monitor="val_loss",
                                        patience=3,
                                        verbose=True,
                                        mode="min")

    checkpoint_callback = ModelCheckpoint(monitor="val_loss",
                                          dirpath=MODEL_DIR,
                                          filename="best_model",
                                          save_top_k=1,
                                          mode="min")

    model_logs = os.path.abspath(join_path(LOGS_DIR, "model_logs"))
    if not os.path.exists(model_logs):
        logging.info("Creating model logs directory...")
        os.makedirs(model_logs)

    # Use TensorBoardLogger for logging
    logger = pl.loggers.TensorBoardLogger(LOGS_DIR, name="model_logs")

    trainer = pl.Trainer(max_epochs=max_epochs,
                         callbacks=[early_stop_callback, checkpoint_callback],
                         log_every_n_steps=4,
                         logger=logger)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    t_stop = stoper()
    logging.info(f"Training time [s]: {(t_stop - t_start):.2f}")

    # Save model
    torch.save(model, MODEL_FILE)
    logging.info("Model saved.")

    return model


def evaluation_results(model, dataloader):
    """Evaluate the performance of the model on the provided dataloader."""
    logging.info("Evaluating results...")
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets_batch = batch
            outputs = model(inputs)
            _, predictions_batch = torch.max(outputs, 1)
            
            all_predictions.extend(predictions_batch.cpu().numpy())
            all_targets.extend(targets_batch.cpu().numpy())

    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    results = f1_score(all_targets, all_predictions, average="micro")
    logging.info(f"Evaluation f1_score: {results:.2f}")
    
    return all_predictions, all_targets


if __name__ == "__main__":
    trainer = Training()
    train_loader, val_loader = trainer.data_preparation(train_path=TRAIN_PATH)
    model = train_model(train_loader=train_loader, 
                             val_loader=val_loader, 
                             hidden_layer_size=HIDDEN_LAYER_SIZE,
                             max_epochs = MAX_EPOCHS,
                             input_size=trainer.model_input_len,
                             output_size=trainer.model_input_len)
    evaluation_results(model, val_loader)
