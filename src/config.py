import os
import json
from pathlib import Path
from pymongo import MongoClient, DESCENDING
from datetime import datetime


class ConfigManager:
    def __init__(self) -> None:
        self.PYTHON_ENV = os.getenv("python_env", "local")
        self.structure_mongo_cm = ConfigManagerMongo(collection_name="dp_vendors")
        self.mongo_cm = ConfigManagerMongo(collection_name="dp_config_2")
        self.config = self.get_base_config()
        self.SRC_FILES_DIR = self.config.get("SRC_FILES_DIR")
        self.DATA_DIR = self.config.get("DATA_DIR")
        self.MODELS_DIR = os.path.join(self.SRC_FILES_DIR, "models_dir")

        self.LLM_MODEL_NAME = self.config.get("LLM_MODEL_NAME")
        self.EMBEDDING_MODEL_NAME = self.config.get("EMBEDDING_MODEL_NAME")
        self.RERANKING_MODEL_NAME = self.config.get("RERANKING_MODEL_NAME")

        self.LLM_MODEL_PATH = os.path.join(self.MODELS_DIR, "llms", self.LLM_MODEL_NAME)
        self.EMBEDDING_MODEL_PATH = os.path.join(
            self.MODELS_DIR, "embedding_models", self.EMBEDDING_MODEL_NAME
        )
        self.RERANKING_MODEL_PATH = os.path.join(
            self.MODELS_DIR, "reranking_models", self.RERANKING_MODEL_NAME
        )

        self.LOGS_DIR = Path(os.path.join(os.getenv("DATA_DIR"), "logs"))
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

        self.MONGO_URI = os.getenv("MONGO_URI")
        self.MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
        self.PORT = os.getenv("PORT")
        self.BASE_PATH = os.getenv("BASE_PATH")
        self.CONFIGS_DIR = os.getenv("CONFIGS_DIR")
        self.INPUT_DIR = os.path.join(self.DATA_DIR, "inputs")
        self.OUTPUT_DIR = os.path.join(self.DATA_DIR, "outputs")
        self.INTER_DIR = os.path.join(self.DATA_DIR, "intermediate")
        self.STRUCTURE_CONFIG_FILEPATH = os.path.join(
            self.CONFIGS_DIR, "structure.json"
        )
        self.PROMPTS_CONFIG_FILEPATH = os.path.join(
            self.CONFIGS_DIR, "prompts_config.json"
        )
        self.ENTITY_CONFIG_FILEPATH = os.path.join(
            self.CONFIGS_DIR, "entity_config.json"
        )
        self.entity_config_data = self.get_entity_config()
        self.structure_config = self.get_structure_config()

    def read_config(self, filepath):
        try:
            with open(filepath, "r") as file:
                config_data = json.load(file)
            return config_data
        except FileNotFoundError:
            print("Config file not found.")
            return None
        except json.JSONDecodeError:
            print("Error decoding JSON in config file.")
            return None

    def write_config(self, config_data):
        with open(self.config_filepath, "w") as file:
            json.dump(config_data, file, indent=4)

    def get_entity_config(self):
        return self.read_config(filepath=self.ENTITY_CONFIG_FILEPATH)

    def get_prompts_config(self):
        return self.read_config(filepath=self.PROMPTS_CONFIG_FILEPATH)

    def get_structure_config(self):
        return self.structure_mongo_cm.get_config()

    def get_base_config(self):
        config = self.mongo_cm.get_config()
        config["_id"] = str(config.get("_id"))
        return config

    def update_structure_config(self):
        try:
            structure_config = self.read_config(filepath=self.STRUCTURE_CONFIG_FILEPATH)
            self.structure_mongo_cm.update_structure_config(data=structure_config)
            return True, "Structure Updated Successfully"
        except Exception as e:
            print(f"Error updating structure config: {e}")
            return False, e

    def get_fields_from_filetype(self, filetype):
        if self.entity_config_data is None:
            self.entity_config_data = self.get_entity_config()
        configs = self.entity_config_data["configs"]
        for config in configs:
            if config["type"] == filetype:
                return config["fields"]
        print("No config found for this filetype.")
        return []


class ConfigManagerMongo:
    def __init__(self, collection_name="dp_config"):
        self.mongo_uri = os.getenv("MONGO_URI")
        self.db_name = os.getenv("MONGO_DB_NAME")
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[collection_name]
        self.collection_name = collection_name

    def update_structure_config(self, data: dict):
        data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.collection.drop()
        self.collection.insert_one(data)

    def get_config(self):
        if self.collection.count_documents(filter={}) == 0:
            if self.collection_name == "dp_config":
                self.create_inital_config()
            elif self.collection_name == "dp_vendors":
                self.create_inital_struct_config()
        for document in (
            self.collection.find().sort([("timestamp", DESCENDING)]).limit(1)
        ):
            return document

    def create_inital_config(self):
        # Read the .env file and create a dictionary
        env_vars = {}
        env_path = os.path.join("./src/.env")  # TODO: Fix this for non local envs
        with open(env_path, "r") as f:
            for line in f:
                res = line.strip().split("=")
                value = "=".join(res[1:])
                key = res[0]
                env_vars[key] = value

        env_vars["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Insert the dictionary into the MongoDB collection
        self.collection.insert_one(env_vars)

    def create_inital_struct_config(self):
        filepath = "configs/structure.json"
        with open(filepath, "r") as file:
            config_data = json.load(file)
        self.update_structure_config(data=config_data)
