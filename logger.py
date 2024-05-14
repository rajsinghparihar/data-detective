import logging
import os
from dotenv import load_dotenv
from pathlib import Path
from pymongo import MongoClient
from datetime import datetime


load_dotenv(override=True)

LOGS_DIR = Path(
    os.path.join(os.getenv("MODEL_DATA_DIR"), "logs")
)  # Replace with your actual log directory path
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
MONGO_URI = os.getenv("MONGO_URI")


class CustomLogger:
    def __init__(self):
        self.logger = None

    def configure_logger(self, process_id: str = "", log_file_name=None):
        logs_dir = LOGS_DIR
        logs_dir.mkdir(
            parents=True, exist_ok=True
        )  # Create the directory if it doesn't exist

        # Get the file name from the calling module
        if not log_file_name:
            import inspect

            calling_module = inspect.stack()[1].filename
            log_file_name = Path(
                calling_module
            ).stem  # Use the module name as the log file name

        # Configure the logger
        self.logger = logging.getLogger(log_file_name)
        self.logger.setLevel(LOG_LEVEL)
        self.logger.handlers.clear()
        # self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        mongo_logger = MongoLogger(
            process_id=process_id,
            uri=MONGO_URI,
            database_name="document-processor-storage",
            collection_name="dp_logs",
        )

        # Add the MongoLogger to the root logger
        self.logger.addHandler(mongo_logger)

        file_handler = logging.FileHandler(logs_dir / "app.log")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

        return self.logger


class MongoLogger(logging.Handler):
    """
    A logging handler that writes logs to a MongoDB collection.
    """

    def __init__(
        self,
        process_id,
        uri,
        database_name,
        collection_name,
        level=logging.INFO,
        timestamp_format="%H:%M:%S  %d-%m-%y",
    ):
        """
        Initializes the MongoLogger.

        Args:
            uri (str): The MongoDB connection URI.
            database_name (str): The name of the database to store logs.
            collection_name (str): The name of the collection to store logs.
            level (int, optional): The logging level. Defaults to logging.INFO.
        """
        super().__init__(level)
        self.client = MongoClient(uri)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        self.timestamp_format = timestamp_format
        self.process_id = process_id

    def emit(self, record):
        """
        Emits a log record to the MongoDB collection.

        Args:
            record (logging.LogRecord): The log record to be emitted.
        """
        log_message = self.format(record)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_data = {
            "timestamp": timestamp,
            "level": record.levelname,
            "name": record.name,
            "message": log_message,
            "process_id": self.process_id,
        }
        self.collection.insert_one(log_data)
