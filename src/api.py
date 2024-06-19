import os

from src.text_ext_llm import LLMEntityExtractor
from src.mongo import MongoUtils
from src.utils import Utils, DataSanityCheck
from src.text_ext_ocr import OCREntityExtractor
from typing import Optional
import copy
from glob import glob

from src.logger import MongoLogWriter, CustomLogger


class ExtractionAPI:
    def __init__(self, rerank) -> None:
        self.logger_instance = CustomLogger(__name__)
        self.rerank = rerank
        self.utils = Utils()
        self.config_manager = self.utils.cm
        self.mongo_logger = MongoLogWriter(
            uri=self.config_manager.MONGO_URI,
            database_name=self.config_manager.MONGO_DB_NAME,
            collection_name="dp_logs",
        )
        log_msg = f"Initializing Class {__name__}.{self.__class__.__qualname__}"
        self.mongo_logger.push_log(
            level="DEBUG",
            name=str(__name__),
            message=log_msg,
            process_id="",
        )
        self.logger_instance.logger.debug(log_msg)

    def configure(
        self,
        document_type: str,
        struct_type: Optional[str] = "",
        document_path: Optional[str] = "",
        document_dir: Optional[str] = "",
    ):
        self.process_id = self.utils.generate_unique_process_id()
        self.logger = self.logger_instance.configure_logger()
        log_msg = f"Configuring Class {__name__}.{self.__class__.__qualname__}"
        self.logger_instance.logger.debug(log_msg)
        self.mongo_logger.push_log(
            level="DEBUG",
            name=str(__name__),
            message=log_msg,
            process_id=self.process_id,
        )
        self.document_path = document_path
        self.document_dir = document_dir
        self.document_type = document_type
        self.struct_type = struct_type

        self.fields = self.config_manager.get_fields_from_filetype(
            filetype=self.document_type
        )
        print("Reconfiguring ExtractionAPI:", self.fields)

        self.mongo_utils = MongoUtils(
            collection_name=self.document_type, process_id=self.process_id
        )
        self.mongo_status_utils = MongoUtils(
            collection_name="dp_status", process_id=self.process_id
        )

    def get_entities(self, document_path=None):
        # processing for single file for sure
        print("Processing File:", self.document_path, document_path)

        # self.fields is getting modified since the utils.replace_keys function uses pop on the passed list.
        # so we pass a deepcopy of the variable (python has no concept of passed variables by value, always passed as reference)
        fields_copy = copy.deepcopy(self.fields)
        print("Processing Fields:", fields_copy)
        if document_path:
            self.document_path = document_path
        self.document_name = os.path.basename(self.document_path)
        log_msg = f"Processing file: {self.document_name}, with struct_type: {self.struct_type} and document_type: {self.document_type}"
        self.logger.debug(log_msg)
        self.mongo_logger.push_log(
            level="DEBUG",
            name=str(__name__),
            message=log_msg,
            process_id=self.process_id,
        )
        mongo_record_id = self.mongo_status_utils.update_mongo_status(
            filename=self.document_name, process_id=self.process_id, start=True
        )
        if self.struct_type != "":
            """
            - get data, flag from ocr_extractor
            - data should be in json format 
            - e.g. {
                "Document Number": "BGDGP-INDJO-2024-03-00001-INV-24-03", 
                "Total Invoice Due Amount": "5.65", 
                "Amount Currency": "USD", 
                "Invoice Description or Details": "Invoice for Roaming Traffic", 
                "Jio/Reliance Entity Name": "Reliance Jio Infocomm Limited", 
                "Vendor Name": "GrameenPhone Ltd", 
                "Vendor Country": "Bangladesh", 
                "Invoice Date": "10.04.2024"
            }
            - [ DEPRECIATED ] e.g. "INV-1234;1200;USD;Desc" or "Invoice Number;Amount;Currency;Description\nINV-1234;1200;USD;Desc"
            """
            self.ocr_extractor = OCREntityExtractor(
                document_path=self.document_path,
                document_type=self.document_type,
                struct_type=self.struct_type,
                process_id=self.process_id,
            )
            ret, data = self.ocr_extractor.extract()
            if ret:
                # post process data and push to mongo
                print("Output from OCREntityExtractor: ", data)
                # json_data = self.utils.postprocess_json_string(json_string=data)
                # if output is from ocr extractor, keys may increase e.g. invoice_amount_2, currency_2 etc.
                # so we don't try to replace them using the replace keys function
                if not DataSanityCheck(
                    data=data,
                    process_id=self.process_id,
                ).run():
                    log_msg = f"""Entity extraction for {self.document_name} and struct_type {self.struct_type} failed. Some Data Sanity Checks did not pass, 
possible issue could be mismatch in document's struct_type and the user provided struct_type passed in request"""
                    self.logger.error(log_msg)
                    self.mongo_logger.push_log(
                        level="ERROR",
                        name=str(__name__),
                        message=log_msg,
                        process_id=self.process_id,
                    )

                    self.mongo_status_utils.update_mongo_status(
                        filename=self.document_name,
                        process_id=self.process_id,
                        id=mongo_record_id,
                        success=False,
                        start=False,
                    )
                    return
                log_msg = f"Entity extraction for {self.document_name} completed successfully."
                self.logger.info(log_msg)
                self.mongo_logger.push_log(
                    level="INFO",
                    name=str(__name__),
                    message=log_msg,
                    process_id=self.process_id,
                )

                self.utils.postprocess_json_data(
                    json_data=data,
                    filename=self.document_name,
                    process_id=self.process_id,
                )
                log_msg = f"Data post processing for {self.document_name} completed sucessfully."
                self.logger.info(log_msg)
                self.mongo_logger.push_log(
                    level="INFO",
                    name=str(__name__),
                    message=log_msg,
                    process_id=self.process_id,
                )
                data = self.utils.break_invoce_string(data)
                self.mongo_utils.push_to_mongo(data=[data])
                self.mongo_status_utils.update_mongo_status(
                    filename=self.document_name,
                    process_id=self.process_id,
                    id=mongo_record_id,
                    success=True,
                    start=False,
                )
                log_msg = f"Processed data pushed to MongoDB for {self.document_name} sucessfully."
                self.logger.info(log_msg)
                self.mongo_logger.push_log(
                    level="INFO",
                    name=str(__name__),
                    message=log_msg,
                    process_id=self.process_id,
                )

            else:
                # call LLMEntityExtractor (ret is false means that either struct_type not found or extracted data is incorrect)
                self.llm_extractor = LLMEntityExtractor(
                    filepath=self.document_path,
                    filetype=self.document_type,
                    rerank=self.rerank,
                )
                data = self.llm_extractor.extract()
                # post process data and push to mongo
                print(
                    "OCR did not work, unknown struct type, getting output from LLMEntityExtractor: ",
                    data,
                )
                json_data = self.utils.postprocess_json_string(json_string=data)
                # replace keys to keep things consistent
                try:
                    json_data = self.utils.replace_keys(
                        data=json_data, keys=fields_copy
                    )
                    json_data = self.utils.postprocess_json_data(
                        json_data=json_data,
                        filename=self.document_name,
                        process_id=self.process_id,
                    )
                    self.mongo_utils.push_to_mongo(data=[json_data])
                    self.mongo_status_utils.update_mongo_status(
                        filename=self.document_name,
                        process_id=self.process_id,
                        id=mongo_record_id,
                        success=True,
                        start=False,
                    )
                    log_msg = f"Processed data pushed to MongoDB for {self.document_name} sucessfully."
                    self.logger.info(log_msg)
                    self.mongo_logger.push_log(
                        level="INFO",
                        name=str(__name__),
                        message=log_msg,
                        process_id=self.process_id,
                    )
                except Exception as e:
                    print("Error in utils.replace_keys: ", e)
                    self.mongo_status_utils.update_mongo_status(
                        filename=self.document_name,
                        process_id=self.process_id,
                        id=mongo_record_id,
                        success=False,
                        start=False,
                    )
                    log_msg = f"Entity extraction failed for {self.document_name}."
                    self.logger.error(log_msg)
                    self.mongo_logger.push_log(
                        level="ERROR",
                        name=str(__name__),
                        message=log_msg,
                        process_id=self.process_id,
                    )

        else:
            # call LLMEntityExtractor (since struct_type is empty means that it was never passed)
            self.llm_extractor = LLMEntityExtractor(
                filepath=self.document_path,
                filetype=self.document_type,
                rerank=self.rerank,
            )
            data = self.llm_extractor.extract()
            # post process data and push to mongo
            print(
                "struct_type not passed in request, getting output from LLMEntityExtractor: ",
                data,
            )
            json_data = self.utils.postprocess_json_string(json_string=data)
            # replace keys to keep things consistent
            try:
                json_data = self.utils.replace_keys(data=json_data, keys=fields_copy)
                json_data = self.utils.postprocess_json_data(
                    json_data=json_data,
                    filename=self.document_name,
                    process_id=self.process_id,
                )
                self.mongo_status_utils.update_mongo_status(
                    filename=self.document_name,
                    process_id=self.process_id,
                    id=mongo_record_id,
                    success=True,
                    start=False,
                )
                log_msg = f"Processed data pushed to MongoDB for {self.document_name} sucessfully."
                self.logger.info(log_msg)
                self.mongo_logger.push_log(
                    level="INFO",
                    name=str(__name__),
                    message=log_msg,
                    process_id=self.process_id,
                )
                self.mongo_utils.push_to_mongo(data=[json_data])
            except Exception as e:
                print("Error in utils.replace_keys: ", e)
                self.mongo_status_utils.update_mongo_status(
                    filename=self.document_name,
                    process_id=self.process_id,
                    id=mongo_record_id,
                    success=False,
                    start=False,
                )
                log_msg = f"Entity extraction failed for {self.document_name}."
                self.logger.error(log_msg)
                self.mongo_logger.push_log(
                    level="ERROR",
                    name=str(__name__),
                    message=log_msg,
                    process_id=self.process_id,
                )

    def get_entities_from_dir(self):
        log_msg = f"Starting to process files in directory: {self.document_dir}"
        self.logger.debug(log_msg)
        self.mongo_logger.push_log(
            level="DEBUG",
            name=str(__name__),
            message=log_msg,
            process_id=self.process_id,
        )
        files = glob(f"{self.document_dir}/**/*.pdf", recursive=True)
        for filepath in files:
            if not filepath.endswith(".pdf"):
                continue
            try:
                self.get_entities(document_path=filepath)
            except Exception as e:
                # TODO: Mongo status update as completed, success = false
                print(e)
        self.mongo_utils.close()
