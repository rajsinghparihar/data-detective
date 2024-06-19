import os
import re
import uuid
import json
import requests
import shutil
import tempfile
import pandas as pd
from datetime import datetime
from src.config import ConfigManager
from src.logger import CustomLogger, MongoLogWriter


class Utils:
    def __init__(self) -> None:
        self.cm = ConfigManager()
        self.data_dir = self.cm.INTER_DIR
        self.logger = CustomLogger(__name__).logger
        self.mongo_logger = MongoLogWriter(
            uri=self.cm.MONGO_URI,
            database_name=self.cm.MONGO_DB_NAME,
            collection_name="dp_logs",
        )
        self.datetime_format = "%Y-%m-%d %H:%M:%S"

    def download_and_save_file(self, source, save_dir):
        """
        Downloads a file from a source local ,remote, url.
        and saves it to the specified directory.

        Args:
            source (str): The URL, local path, or remote file path.
            save_dir (str): The directory where the file should be saved.

        Raises:
            ValueError: If the source is not a valid URL or local path.
            OSError: If there's an error while downloading or saving the file.
        """
        # Check if source is a URL
        if source.startswith("http"):
            try:
                response = requests.get(source, stream=True)
                # Raise exception for non-2xx response codes
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error downloading file from URL: {source}")
                raise ValueError(
                    f"Error downloading file from URL: {source}, \
                    {e}"
                )

            # Get filename from URL or response headers
            filename = os.path.basename(source)
            content_disposition = response.headers.get("content-disposition")
            if content_disposition:
                filename = re.findall(r"filename=(.*)", content_disposition)[0]
        else:
            # Assume local path or remote file path (not starting with http)
            if not os.path.exists(source):
                self.logger.error(f"File not found: {source}")
                raise ValueError(f"File not found: {source}")
            filename = os.path.basename(source)

        # Construct the full save path within the directory
        save_path = os.path.join(save_dir, filename)

        # Open the save file for writing in binary mode
        with open(save_path, "wb") as f_out:
            if source.startswith("http"):
                # Write downloaded content in chunks
                for chunk in response.iter_content(1024):
                    f_out.write(chunk)
            else:
                # Copy local/remote file content in chunks
                with open(source, "rb") as f_in:
                    for chunk in iter(lambda: f_in.read(4096), b""):
                        f_out.write(chunk)

        # Print confirmation message
        self.logger.debug(
            f"File '{filename}' downloaded from '{source}' and saved to '{save_dir}'."
        )
        log_msg = f"File '{filename}' Downloaded!"
        self.logger.info(log_msg)
        self.mongo_logger.push_log(
            level="INFO",
            name=str(__name__),
            message=log_msg,
            process_id="",
        )

    def clear_dir(self, dir):
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.mkdir(dir)

    def create_temp_txt_file(self, text_to_save):
        """
        Creates a temporary text file and saves the given text in it.
        Args:
            text_to_save (str): The text to save in the temporary file.
        Returns:
            str: The path to the temporary text file, None otherwise.
        """
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
                # Encode text for binary writing
                temp_file.write(text_to_save.encode())
                return temp_file.name
        except Exception as e:
            print(f"Error creating temporary file: {e}")
            return None

    def save_output(self, filename, output_text):
        output_filepath = os.path.join(
            self.data_dir, "".join(filename.split(".")[:-1]) + ".csv"
        )
        with open(output_filepath, "w+") as f:
            f.write(output_text)
        output_filepath = os.path.abspath(output_filepath)

        return output_filepath

    def save_output_df(self, df, filename):
        output_filepath = os.path.join(
            self.data_dir, "".join(filename.split(".")[:-1]) + ".csv"
        )
        output_filepath = os.path.abspath(output_filepath)
        df.to_csv(output_filepath, index=False, sep=";")
        return output_filepath

    def postprocess_output(self, csv_filepath, fields, filename, delimiter=";"):
        df = pd.read_csv(csv_filepath, delimiter=delimiter)
        if df.shape[0] == 0:
            df.loc[1, :] = df.columns.values
            df.columns = fields
        df["timestamp"] = datetime.now().strftime(self.datetime_format)
        df["filename"] = filename
        df.to_csv(csv_filepath, index=False, sep=delimiter)

        return df

    def generate_unique_process_id(self):
        """Generates a unique request ID using UUID."""
        return str(uuid.uuid1())

    def replace_keys(self, data, keys):
        """
        Replaces keys in a dictionary with a list of given keys.

        Args:
            data: The dictionary to modify.
            keys: A list of keys to replace the original keys with.

        Returns:
            A new dictionary with the replaced keys.
        """
        if keys == [] or len(data.keys()) != len(keys):
            return data
        new_data = {}
        for key, value in data.items():
            new_key = keys.pop(0)
            new_data[new_key] = value
        return new_data

    def postprocess_json_string(self, json_string: str) -> dict:
        json_string = json_string.replace("'", '"')
        json_string = json_string[json_string.rfind("{") : json_string.rfind("}") + 1]
        try:
            json_data = json.loads(json_string)
        except Exception as e:
            print("Error parsing output, invalid json format", e)
        return json_data

    def postprocess_json_data(
        self, json_data: dict, filename: str, process_id: str
    ) -> dict:
        json_data["timestamp"] = datetime.now().strftime(self.datetime_format)
        json_data["filename"] = filename
        json_data["process_id"] = process_id
        return json_data

    def json_to_str(self, data, indent):
        return json.dumps(data, indent=indent)

    def get_timestamp(self):
        return datetime.now().strftime(self.datetime_format)

    def break_invoce_string(self, json_data):
        invoice_str = json_data["Invoice_Number"]
        parts = (
            invoice_str.replace("-", " ").replace("_", " ").replace("/", " ").split()
        )
        if len(parts) < 3 or not parts:
            json_data["Billing Operator"] = None
            json_data["Franchies"] = None
        else:
            json_data["Billing Operator"] = parts[0]
            json_data["Franchies"] = parts[1]
        return json_data


class DataSanityCheck:
    def __init__(self, data: dict, process_id: str) -> None:
        self.data = data
        self.process_id = process_id
        self.cm = ConfigManager()
        self.logger = CustomLogger(__name__).logger
        self.mongo_logger = MongoLogWriter(
            uri=self.cm.MONGO_URI,
            database_name=self.cm.MONGO_DB_NAME,
            collection_name="dp_logs",
        )
        log_msg = "Running Sanity Checks..."
        self.logger.debug(log_msg)
        self.mongo_logger.push_log(
            level="DEBUG",
            name=str(__name__),
            message=log_msg,
            process_id=self.process_id,
        )

    def contains_missing_value(self):
        for k, v in self.data.items():
            if v == "" or v is None:
                return True
        return False

    def is_not_missing_value(self):
        log_msg = "Running missing value sanity check..."
        self.logger.debug(log_msg)
        self.mongo_logger.push_log(
            level="DEBUG",
            name=str(__name__),
            message=log_msg,
            process_id=self.process_id,
        )
        count_missing = 0
        for key, val in self.data.items():
            val = val.strip()
            if val == "":
                count_missing += 1
        print(count_missing / len(self.data))
        return count_missing / len(self.data) < 0.1

    def is_amount_float(self):
        log_msg = "Running amount sanity check..."
        self.logger.debug(log_msg)
        self.mongo_logger.push_log(
            level="DEBUG",
            name=str(__name__),
            message=log_msg,
            process_id=self.process_id,
        )
        for key, val in self.data.items():
            if key.lower().__contains__("amount"):
                try:
                    _ = float(val)
                    return True
                except Exception as e:
                    log_msg = f"Error in {__class__.__name__}.check_amount_value: {e}"
                    self.logger.error(log_msg)
                    self.mongo_logger.push_log(
                        level="ERROR",
                        name=str(__name__),
                        message=log_msg,
                        process_id=self.process_id,
                    )
                    return False

    def run(self):
        """
        returns True if all sanity checks are passing.
        """
        check_labels = ["missing_value_check", "amount_check"]
        checks = [self.is_not_missing_value(), self.is_amount_float()]
        for i, check in enumerate(checks):
            if not check:
                log_msg = f"Check {check_labels[i]} Failed"
                self.logger.debug(log_msg)
                self.mongo_logger.push_log(
                    level="DEBUG",
                    name=str(__name__),
                    message=log_msg,
                    process_id=self.process_id,
                )

                log_msg = "Data sanity checks completed. Some Checks failed. Data not correctly parsed."
                self.logger.debug(log_msg)
                self.mongo_logger.push_log(
                    level="DEBUG",
                    name=str(__name__),
                    message=log_msg,
                    process_id=self.process_id,
                )
                return False
        log_msg = "Data sanity checks completed. All checks passed. Data is correct."
        self.logger.debug(log_msg)
        self.mongo_logger.push_log(
            level="DEBUG",
            name=str(__name__),
            message=log_msg,
            process_id=self.process_id,
        )
        return True

    def run_llm(self):
        # works differently than the above run function
        return self.contains_missing_value()
