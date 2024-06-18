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
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")


class Logger(object):
    def __init__(self, logger_name):
        self.logger = None
        self.logger_name = logger_name
        self.configure_logger()

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            # Log function call with arguments
            self.logger.info(
                f"Calling function: {func.__name__} with args: {args}, kwargs: {kwargs}"
            )
            # Call the original function
            result = func(*args, **kwargs)
            # Log function return value
            self.logger.info(f"Function: {func.__name__} returned: {result}")
            return result

        return wrapper

    def configure_logger(self, process_id: str = ""):
        logs_dir = LOGS_DIR
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Get the file name from the calling module
        if not self.logger_name:
            import inspect

            calling_module = inspect.stack()[1].filename
            self.logger_name = Path(calling_module).stem

        # Configure the logger
        self.logger = logging.getLogger(name=self.logger_name)
        self.logger.setLevel(LOG_LEVEL)
        self.logger.handlers.clear()

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        mongo_logger = MongoLogger(
            process_id=process_id,
            uri=MONGO_URI,
            database_name=MONGO_DB_NAME,
            collection_name="dp_logs",
            level=LOG_LEVEL,
        )
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
