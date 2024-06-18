import logging
import os
from pathlib import Path
from pymongo import MongoClient
from src.config import ConfigManager
import time


class CustomLogger(object):
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

    def configure_logger(self):
        self.cm = ConfigManager()
        os.makedirs(self.cm.LOGS_DIR, exist_ok=True)

        # Get the file name from the calling module
        if not self.logger_name:
            import inspect

            calling_module = inspect.stack()[1].filename
            self.logger_name = Path(calling_module).stem

        # Configure the logger
        self.logger = logging.getLogger(name=self.logger_name)
        self.logger.setLevel(self.cm.LOG_LEVEL)
        self.logger.handlers.clear()

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        file_handler = logging.FileHandler(f"{self.cm.LOGS_DIR}/app.log")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        return self.logger


class MongoLogWriter:
    """
    A class that helps push logs to a MongoDB collection.
    """

    def __init__(
        self, uri, database_name, collection_name, timestamp_format="%Y-%m-%d %H:%M:%S"
    ):
        """
        Initializes the MongoLogWriter.

        Args:
            uri (str): The MongoDB connection URI.
            database_name (str): The name of the database to store logs.
            collection_name (str): The name of the collection to store logs.
            timestamp_format (str, optional): The format for timestamps in logs. Defaults to "%Y-%m-%d %H:%M:%S" (hh-mm-ss-dd-mm-yy).
        """
        self.client = MongoClient(uri)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        self.timestamp_format = timestamp_format

    def push_log(self, level, name, message, process_id):
        """
        Pushes a log message to the MongoDB collection.

        Args:
            level (str): The logging level (e.g., INFO, ERROR).
            name (str): The name of the logger.
            message (str): The log message.
        """
        timestamp = time.time()
        formatted_timestamp = time.strftime(
            self.timestamp_format, time.localtime(timestamp)
        )
        log_data = {
            "timestamp": formatted_timestamp,
            "level": level,
            "name": name,
            "message": message,
            "process_id": process_id,
        }
        self.collection.insert_one(log_data)
