import logging
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(override=True)

LOGS_DIR = Path(
    os.path.join(os.getenv("MODEL_DATA_DIR"), "logs")
)  # Replace with your actual log directory path
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


class CustomLogger:
    def __init__(self):
        self.logger = None

    def configure_logger(self, log_file_name=None):
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
        # self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        file_handler = logging.FileHandler(logs_dir / f"app.log")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

        return self.logger
