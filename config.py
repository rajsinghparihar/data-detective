import json
import os
from logger import Logger

logger = Logger(__name__).logger


class ConfigManager:
    def __init__(self) -> None:
        logger.debug("Initializing ConfigManager")
        self.models_dir = os.getenv("MODEL_DATA_DIR")
        self.config_filepath = os.path.join(
            self.models_dir, "models/configs/config.json"
        )
        self.config_data = self.read_config()

    def read_config(self):
        logger.debug("Reading config file")
        try:
            with open(self.config_filepath, "r") as file:
                config_data = json.load(file)
            return config_data
        except FileNotFoundError:
            print("Config file not found.")
            return None
        except json.JSONDecodeError:
            print("Error decoding JSON in config file.")
            return None

    def write_config(self, config_data):
        logger.debug("Writing updated config to file")
        self.config_data = config_data
        with open(self.config_filepath, "w") as file:
            json.dump(config_data, file, indent=4)

    def update_config(self, filetype, fields):
        logger.debug("Updating configuration for type: %s", filetype)
        # Check if the type already exists
        for config in self.config_data["configs"]:
            if config["type"] == filetype:
                # Update existing fields
                existing_fields = set(config["fields"])
                fields = set(fields)
                config["fields"] = list(existing_fields.union(fields))
                return self.config_data
        # If type does not exist, add new configuration
        logger.debug("Adding new configuration for type: %s", filetype)
        self.config_data["configs"].append({"type": filetype, "fields": fields})
        return self.config_data

    def update_config_with_new_fields(self, filetype, fields):
        logger.debug("Updating config with new fields for type: %s", filetype)
        self.config_data = self.read_config()
        if self.config_data is None:
            return
        self.config_data = self.update_config(filetype, fields)
        self.write_config(self.config_data)

    def get_fields_from_filetype(self, filetype):
        logger.debug("Getting fields for type: %s", filetype)
        if self.config_data is None:
            self.config_data = self.read_config()
        configs = self.config_data["configs"]
        for config in configs:
            if config["type"] == filetype:
                return config["fields"]
        logger.error("No config found for this filetype.")
        return []

    def get_db_table_info(self):
        logger.debug("Getting database table information")
        document_types = []
        for data in self.config_data["configs"]:
            document_types.append((data["type"].title(), data["fields"]))
        return document_types


class PromptManager:
    def __init__(self, filetype) -> None:
        logger.debug("Initializing PromptManager")
        self.models_dir = os.getenv("MODEL_DATA_DIR")
        self.prompt_config_filepath = os.path.join(
            self.models_dir, "models/configs/prompts.json"
        )
        self.prompts = self.read_config()
        self.filetype = filetype

    def read_config(self) -> dict:
        logger.debug("Reading prompt configuration file")
        try:
            with open(self.prompt_config_filepath, "r") as file:
                config_data = json.load(file)
            return config_data
        except FileNotFoundError:
            logger.error("Prompt Config file not found.")
            return None
        except json.JSONDecodeError:
            logger.error("Error decoding JSON in config file.")
            return None

    def get_prompt(self) -> str:
        if self.filetype == "invoice":
            logger.debug("Getting invoice prompt")
            return self.prompts["invoice_prompt"]
        logger.debug("Getting general prompt")
        return self.prompts["general_prompt"]
