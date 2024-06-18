import os
import csv
from pymongo import MongoClient
from src.logger import CustomLogger
from src.utils import Utils
from src.logger import MongoLogWriter
from src.config import ConfigManager


class MongoUtils:
    def __init__(self, collection_name, process_id):
        self.mongo_uri = os.getenv("MONGO_URI")
        self.db_name = os.getenv("MONGO_DB_NAME")
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[collection_name]
        self.config_manager = ConfigManager()
        self.mongo_logger = MongoLogWriter(
            uri=self.config_manager.MONGO_URI,
            database_name=self.config_manager.MONGO_DB_NAME,
            collection_name="dp_logs",
        )
        self.logger = CustomLogger(__name__).configure_logger()
        log_msg = f"Initializing Class {__name__}.{self.__class__.__qualname__}"
        self.logger.debug(log_msg)
        self.mongo_logger.push_log(
            level="DEBUG",
            name=str(__name__),
            message=log_msg,
            process_id=process_id,
        )
        self.utils = Utils()

    def read_csv(self, file_path, delimiter=";"):
        data = []
        with open(file_path, "r") as file:
            reader = csv.DictReader(file, delimiter=delimiter)
            for row in reader:
                data.append(row)
        return data

    def push_to_mongo(self, data):
        if len(data) == 1:
            # Use insert_one for a single document
            self.collection.insert_one(data[0])
        else:
            # Use insert_many for multiple documents
            self.collection.insert_many(data)

    def close(self):
        self.client.close()

    def run(self, csv_file):
        data = self.read_csv(csv_file)
        self.push_to_mongo(data)

    def update_mongo_status(
        self, filename, process_id, id=None, success=False, start=True
    ):
        if start:
            file_record_intial = {
                "filename": filename,
                "process_id": process_id,
                "start_time": self.utils.get_timestamp(),
                "end_time": None,
                "status": "processing",
                "success": success,
            }
            result = self.collection.insert_one(file_record_intial)
            return result.inserted_id
        else:
            self.collection.update_one(
                {"_id": id},
                {
                    "$set": {
                        "status": "completed",
                        "success": success,
                        "end_time": self.utils.get_timestamp(),
                    }
                },
            )
        log_msg = f"File {filename} processing status updated to Mongo successfully."
        self.logger.info(log_msg)
        self.mongo_logger.push_log(
            level="INFO",
            name=str(__name__),
            message=log_msg,
            process_id=process_id,
        )
