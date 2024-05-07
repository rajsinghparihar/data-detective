from llama_index import (
    ServiceContext,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.response_synthesizers import get_response_synthesizer
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from llama_index.embeddings import HuggingFaceEmbedding
from pdf2image import convert_from_path
import fitz
from PIL import Image
import numpy as np
import pandas as pd
from paddleocr import PaddleOCR as ppocr
from img2table.ocr import PaddleOCR
from img2table.document import PDF
import os
import zipfile
import shutil
import json
from dotenv import load_dotenv
from pymongo import MongoClient
import csv
from logger import CustomLogger
from datetime import datetime
logger = CustomLogger().configure_logger()

load_dotenv()


class TextExtractor:
    def __init__(self, filepath) -> None:
        self._filepath = filepath
        self._text_filepath = "./temp/invoice_text.txt"
        self._header_filepath = "./temp/vendor_information.txt"
        self.ppocr_obj = ppocr(
            use_angle_cls=True, lang="en", use_gpu=True, verbose=False
        )
        logger.debug("Initialized TextExtractor with filepath: %s", filepath)

    def is_file_readable(self):
        total_words = 0
        doc = fitz.open(self._filepath)
        for page in doc:
            text = page.get_text()
            total_words += len(text.split())

        if total_words < 100:
            return False
        logger.debug("File is computer readable with %d words", total_words)
        return True

    def extract_and_save_txt(self, filetype_keyword: str):
        """
        - filetype_keyword: str
            - examples: ["Invoice", "Customer", etc.]
            only used when file is not readable
            and values must be extracted using ocr
        """
        text_file = open(self._text_filepath, "w+")
        if not self.is_file_readable():
            logger.info("File is not readable. Using OCR to extract text.")
            ocr_text = []
            pages = convert_from_path(self._filepath)
            for page in pages:
                image = np.array(page)
                result = self.ppocr_obj.ocr(img=image, cls=True)[0]
                text = "\n".join([line[1][0] for line in result])
                if text.lower().__contains__(filetype_keyword.lower()):
                    ocr_text.append(text)
                    break
            text_file.writelines(ocr_text)
        text_file.close()

    def extract_header_ocr(self):
        doc = fitz.open(self._filepath)
        page = doc[0]

        header_y = int(page.mediabox[3] / 4)  # Assuming header is in top quarter

        pix = page.get_pixmap(matrix="RGB")
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = np.array(img)
        logger.info("Extracting header text using OCR.")
        header_result = self.ppocr_obj.ocr(img=img[:header_y, :, :], cls=True)[0]
        header_text = "\n".join([line[1][0] for line in header_result])
        with open(self._header_filepath, "w+") as header_file:
            header_file.write(header_text.strip())


class LLMUtils:
    def __init__(self) -> None:
        logger.debug("Initializing LLMUtils")
        self.models_dir = os.getenv("MODEL_DATA_DIR")
        self.model_name = os.getenv("MODEL_NAME")
        logger.debug("Model name: %s", self.model_name)
        self.model_path = os.path.join(self.models_dir, "models/" + self.model_name)
        self._llm = self.load_llm()
        self._embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
        self.service_context = ServiceContext.from_defaults(
            llm=self._llm,
            embed_model=self._embed_model,
        )

    def load_llm(self):
        logger.info(f"Loading LLM model: {self.model_name.split('.')[0]}")
        llm = LlamaCPP(
            model_path=self.model_path,
            temperature=0,
            max_new_tokens=512,
            context_window=3900,
            generate_kwargs={},
            model_kwargs={"n_gpu_layers": 8},
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            verbose=False,
        )
        logger.info("Model loaded!")
        return llm


class ConfigManager:
    def __init__(self) -> None:
        logger.debug("Initializing ConfigManager")
        self.config_filepath = "./temp/config.json"
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
    def __init__(self) -> None:
        logger.debug("Initializing PromptManager")
        self.prompt_config_filepath = "./temp/prompts.json"
        self.prompts = self.read_config()

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

    def get_general_prompt(self) -> str:
        logger.debug("Getting general prompt")
        return self.prompts["general_prompt"]


class Summarizer(TextExtractor):
    def __init__(self, filepath: str, filetype: str) -> None:
        """
        - filetype: str
            - examples: ["Invoice", "CAF", etc.]
        used to select correct fields to summarize from configfile.
        """
        logger.debug("Initializing Summarizer")
        super().__init__(filepath)
        self.prompt_manager = PromptManager()
        self.prompt = self.prompt_manager.get_general_prompt()

        self.config_manager = ConfigManager()
        # always using filetype.lower() assuming that keys in the config file will always be lowercase strings
        self.fields = self.config_manager.get_fields_from_filetype(
            filetype=filetype.lower()
        )
        documents = SimpleDirectoryReader(input_files=[filepath]).load_data()
        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize",
            use_async=True,
        )
        logger.debug("Creating VectorStoreIndex from documents")
        self.index = VectorStoreIndex.from_documents(
            documents=documents,
            response_synthesizer=response_synthesizer,
        )
        logger.debug("Creating query engine from index")
        self.query_engine = self.index.as_query_engine(
            response_mode="tree_summarize", use_async=True, streaming=False
        )

    def csv_formatting(self, csv_file_path, delimiter=";"):
        logger.debug("Formatting CSV file")
        df = pd.read_csv(csv_file_path, delimiter=delimiter)
        if df.shape[0] == 0:
            logger.info("No header in CSV file. Using header as default values")
            df.loc[1, :] = df.columns.values
            df.columns = self.fields
            df.to_csv(csv_file_path, index=False, sep=delimiter)
        else:
            logger.debug("Formatting CSV file with existing data")
            df.to_csv(csv_file_path, index=False, sep=delimiter)

    def summarize(self):
        logger.info("Extracting required fields using LLM call, wait...")
        response = self.query_engine.query(
            self.prompt.format(fields=", ".join(self.fields))
        )
        return response.response


class DocumentsProcessor:
    def __init__(self, zip_filepath, data_dir) -> None:
        logger.debug("Initializing Document Processor")
        self.ocr = PaddleOCR(lang="en", kw={"use_gpu": True})
        self.zip_filepath = zip_filepath
        self.zip_files_dir = os.path.dirname(self.zip_filepath)
        self.extracted_data_dir = data_dir

    def extract_files_from_zip(self):
        logger.debug("Extracting files from zip archive")
        with zipfile.ZipFile(self.zip_filepath, "r") as zip_ref:
            zip_ref.extractall(self.zip_files_dir)

    def extract_data_from_files(self):
        """
        Example:

        - filename = "442.pdf"
        - filepath = "./temp/uploaded_files/442.pdf"
        - output_filename = "442.xlsx"
        - input_filepath = "./temp/uploaded_files/files.zip" -> extracted to "./temp/uploaded_files/"
        - destination_dir = "./temp/extracted_files"
        - output_filepath = "./temp/extracted_files/442.xlsx"
        """
        logger.debug("Extracting data from files")
        for filename in os.listdir(self.zip_files_dir):
            filepath = os.path.join(self.zip_files_dir, filename)
            if filename.endswith("pdf"):
                doc = PDF(src=filepath)
                output_filename = filename.replace(".pdf", ".xlsx")

                output_filepath = os.path.join(self.extracted_data_dir, output_filename)
                logger.debug(f"Converting {filename} to Excel file")
                doc.to_xlsx(
                    dest=output_filepath,
                    ocr=self.ocr,
                    implicit_rows=False,
                    borderless_tables=False,
                    min_confidence=50,
                )
            elif filename.endswith("xlsx") or filename.endswith("csv"):
                # move file to destination directory
                logger.debug(f"Moving {filename} to extracted data directory")
                shutil.copy(filepath, self.extracted_data_dir)
            else:
                continue

    def save_extracted_data(self, save_filepath):
        logger.debug("Saving extracted data as zip archive")
        # extract files from zip archive
        self.extract_files_from_zip()
        # extract data
        self.extract_data_from_files()
        # save files as zip in save_filepath
        shutil.make_archive(save_filepath, "zip", self.extracted_data_dir)
        logger.info(f"Saved extracted data to {save_filepath}")
        return True


class DuplicateChecker:
    def __init__(self, files: list[str], unique_col) -> None:
        self.ocr = ppocr(use_angle_cls=True, lang="en", use_gpu=True, verbose=False)
        self.files = files
        self.unique_col = unique_col

    def extract_unique_ids_from_pdf(self):
        sap_ids = []
        for filepath in self.files:
            last_page = None
            if filepath.endswith("pdf"):
                images = convert_from_path(filepath, dpi=100)
                for i, image in enumerate(images):
                    image = np.array(image)
                    result = self.ocr.ocr(image, cls=True)[0]
                    txts = [line[1][0] for line in result]
                    if txts.__contains__("Work Completion Certificate"):
                        last_page = i
                        for txt in txts:
                            if txt.__contains__("ENB"):
                                sap_ids.append(txt)
                    else:
                        if last_page is not None:
                            break
            else:
                continue
        return sap_ids

    def extract_unique_ids_from_csv(self):
        result_df = pd.DataFrame()
        for filepath in self.files:
            if filepath.endswith("csv"):
                df = pd.read_csv(filepath)
            elif filepath.endswith("xlsx"):
                df = pd.read_excel(filepath)
            else:
                continue
            if df.columns.__contains__(self.unique_col):
                result_df = pd.concat([result_df, df], ignore_index=True)
        if result_df.empty:
            return []
        return result_df[self.unique_col].to_list()

    def find_duplicates(self, strings):
        counts = {}
        duplicates = []
        for string in strings:
            counts[string] = counts.get(string, 0) + 1
        for string, count in counts.items():
            if count > 1:
                duplicates.append(string)
        return duplicates

    def check_duplicates(self):
        sap_ids_from_csv = self.extract_unique_ids_from_csv()
        sap_ids_from_pdf = self.extract_unique_ids_from_pdf()

        total_sap_ids = sap_ids_from_csv + sap_ids_from_pdf
        ret = len(total_sap_ids) != len(set(total_sap_ids))
        duplicate_sap_ids = self.find_duplicates(total_sap_ids)

        return ret, duplicate_sap_ids


# extracts tabular data from single pdf and saves it to single xlsx file
class TabularDataExtractor:
    def __init__(self, filepath: str) -> None:
        logger.debug("Initializing Document Tabular Data Extractor")
        self.ocr = PaddleOCR(lang="en", kw={"use_gpu": True})
        self.filepath = filepath

    def extract_and_save_data(self) -> None:
        doc = PDF(src=self.filepath)
        output_filepath = self.filepath.replace(".pdf", ".xlsx")
        logger.debug("Convertingto Excel file")
        doc.to_xlsx(
            dest=output_filepath,
            ocr=self.ocr,
            implicit_rows=False,
            borderless_tables=False,
            min_confidence=50,
        )
        return output_filepath


# csv to mongoDB data push:
class CSVToMongo:
    def __init__(self, collection_name):
        logger.debug("Initializing CSV to MongoDB processor")
        self.mongo_uri = os.getenv("MONGO_URI")
        self.db_name = os.getenv("MONGO_DB_NAME")
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[collection_name]
    
    def update_mongo_status(self,filename,id = None, success=False,start=True):
        if start:
            file_record_intial = {
                    "filename": filename,
                    "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time":None,
                    "status": "processing",
                    "success": success
                }
            result = self.collection.insert_one(file_record_intial)
            self.client.close()
            return result.inserted_id
        else:
            self.collection.update_one(
                {"_id": id},
                {"$set": {"status": "completed", "success": success,"end_time":datetime.now().strftime("%Y-%m-%d %H:%M:%S")}}
            )
            self.client.close()
        logger.info("file processing status update to Mango is successful.")
                    
    def push_raw_data(self,raw_text,filename):
        data = {"raw_text":raw_text,
        "filename":filename,
        "timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        self.collection.insert_one(data)
        self.client.close()
        logger.info("Raw data upload to Mango is successful.")
        
    def read_csv(self, file_path, delimiter=";"):
        logger.info(f"Reading CSV file {file_path}")
        data = []
        with open(file_path, "r") as file:
            reader = csv.DictReader(file, delimiter=delimiter)
            for row in reader:
                data.append(row)
        return data

    def push_to_mongo(self, data):
        logger.debug("Pushing data to MongoDB")
        if len(data) == 1:
            # Use insert_one for a single document
            self.collection.insert_one(data[0])
        else:
            # Use insert_many for multiple documents
            self.collection.insert_many(data)
        self.client.close()

    def run(self, csv_file):
        data = self.read_csv(csv_file)
        self.push_to_mongo(data)
        logger.info("Extrected data upload to Mongo is successful.")
