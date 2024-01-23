import pandas as pd
import logging
import os
import sys
import json
import requests
from bs4 import BeautifulSoup
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
    os.makedirs(DATA_DIR)

# Load configuration settings from JSON
CONF_FILE = os.path.join(ROOT_DIR, "settings.json")
# CONF_FILE = os.getenv('CONF_PATH') # second option with env variable
logger.info("Loading configuration settings from JSON...")
with open(CONF_FILE, "r") as file:
    configuration = json.load(file)

# Define paths
logger.info("Defining paths...")
DATA_DIR = get_project_dir(configuration["general"]["data_dir"])
TRAIN_PATH = os.path.join(DATA_DIR, configuration["train"]["csv_file"])
INFERENCE_PATH = os.path.join(DATA_DIR, configuration["inference"]["csv_file"])

# Load train_test_split parameters
TEST_SIZE = configuration["train"]["test_size"]
RANDOM_STATE = configuration["general"]["random_state"]

# define target
TARGET_FEATURE = configuration["general"]["target_feature"]

# Load dataset url
DATA_URL = configuration["general"]["data_url"]


class DataProcessor():
    """Class for dowloading and saving data to csv files."""
    def __init__(self):
        self.df = None
        self.data_arr = []
        self.response = None

    def get_dataframe(self):
        """Method to download data from specified url and return dataframe."""
        logger.info("Downloading data...")
        self.response = requests.get(DATA_URL)

        if self.response.status_code == 200:
            soup = BeautifulSoup(self.response.content, "html.parser")
            table = soup.find("table", {"class": "wikitable"})

            for row in table.find_all("tr")[1:]:
                self.data_arr.append([col.get_text(strip=True) for col in row.find_all(["th", "td"])])

            df_cols = ["id", "sepal_length", "sepal_width", 
                       "petal_length", "petal_width", "species"]
            df = pd.DataFrame(self.data_arr, columns=df_cols)
            df.set_index('id', inplace=True)
        else:
            error_message = f"Error when downloading data. Status code: {self.response.status_code}"
            logger.error(error_message)
            raise Exception(error_message)
        
        return df

    def save_to_csv(self):
        """Method to save dataframe to csv in the specified directory."""
        logger.info("Saving Iris Dataset to csv...")
        df = self.get_dataframe()
        raw_file_path = os.path.join(DATA_DIR, "data_iris.csv")
        df.to_csv(raw_file_path, encoding="utf-16", index=False)
        self.df = df

        return df

    def save_train_test_to_csv(self):
        """Train Test Split data and save to csv."""
        logger.info("Train Test Split...")
        df = self.df

        # drop target feature from X and bind to y
        X = df.drop(columns=TARGET_FEATURE)
        y = pd.DataFrame(df[TARGET_FEATURE])

        # Make Train Test Split with stratification and specified test size
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, 
                                                            random_state=RANDOM_STATE, stratify=y)

        df_train = X_train.join(y_train)
        df_train.to_csv(TRAIN_PATH, encoding="utf-16", index=False)
        df_inf = X_test.join(y_test)
        df_inf.to_csv(INFERENCE_PATH, encoding="utf-16", index=False)

# Main execution
if __name__ == "__main__":
    configure_logging()
    logger.info("Starting Data Initialization:")
    process = DataProcessor()
    process.save_to_csv()
    process.save_train_test_to_csv()
    logger.info("Data Initialization completed successfully.")
