###os & sys → for file handling and system operations.

###pandas → used to read and process CSV data.

###train_test_split → splits data into training and testing parts.

###dataclass → helps define configuration classes easily.

###CustomException → a user-defined error handler for clear debugging.

###logging → logs messages about each step in the pipeline.
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from source.exception import CustomException
from source.logger import logging


###config setup
# This dataclass stores file paths where data will be saved:

##train.csv → Training data

##test.csv → Testing data

##data.csv → Raw/original dataset
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


###This initializes the ingestion process using the configuration above.
##When this class is used, it already knows where to save train/test/raw data.


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion component")

        try:
            #read csv
            data_path = r"C:\Users\L E N O V O\Downloads\mlproject-main+(2)\mlproject-main\notebook\data\stud.csv"

            print(f"Reading dataset from: {data_path}")
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"❌ File not found at {data_path}")

            df = pd.read_csv(data_path)
            print(f" Dataset loaded successfully. Shape: {df.shape}")

            #  Create artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            print(f" Artifacts directory ensured at: {os.path.dirname(self.ingestion_config.train_data_path)}")

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            print(" Raw data saved.")

            #  Split and save train/test
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            print(" Train/test data saved inside artifacts folder.")
            logging.info("Ingestion of data completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
