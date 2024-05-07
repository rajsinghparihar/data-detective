from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import Optional
import os
from api import Summarizer, LLMUtils, TabularDataExtractor, CSVToMongo, ConfigManager
from llama_index import set_global_service_context
import requests
import re
from dotenv import load_dotenv
from paddleocr import PaddleOCR as ppocr
from pdf2image import convert_from_path
import numpy as np
import tempfile
import pandas as pd
from datetime import datetime
from logger import CustomLogger
from tqdm import tqdm
from pathlib import Path
import shutil

load_dotenv()
logger = CustomLogger().configure_logger()

app = FastAPI()
llm_module = LLMUtils()
set_global_service_context(service_context=llm_module.service_context)
config_manager = ConfigManager()

MODEL_DATA_DIR = os.getenv("MODEL_DATA_DIR")
INPUT_DATA_DIR = os.getenv("INPUT_DATA_DIR")
data_dir = os.path.join(MODEL_DATA_DIR, "outputs/")


class Document(BaseModel):
    file_path: Optional[str] = ""
    document_type: Optional[str] = ""
    document_dir: Optional[str] = ""


class ProcessResponse(BaseModel):
    response: str
    output_filepath: str


def download_and_save_file(source, save_dir):
    """
    Downloads a file from a source (local path, remote URL, or remote file path)
    and saves it to the specified directory.

    Args:
        source (str): The URL, local path, or remote file path of the file to download.
        save_dir (str): The directory where the file should be saved.

    Raises:
        ValueError: If the source is not a valid URL or local path.
        OSError: If there's an error while downloading or saving the file.
    """

    # Check if source is a URL
    if source.startswith("http"):
        try:
            response = requests.get(source, stream=True)
            response.raise_for_status()  # Raise exception for non-2xx response codes
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading file from URL: {source}")
            raise ValueError(f"Error downloading file from URL: {source}") from e

        # Get filename from URL or response headers
        filename = os.path.basename(source)
        content_disposition = response.headers.get("content-disposition")
        if content_disposition:
            filename = re.findall(r"filename=(.*)", content_disposition)[0]
    else:
        # Assume local path or remote file path (not starting with http)
        if not os.path.exists(source):
            logger.error(f"File not found: {source}")
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
    logger.debug(f"File '{filename}' downloaded from '{source}' and saved to '{save_dir}'.")
    logger.info(f"File '{filename}' Downloaded!")

def extract_page_texts(pdf_path):
    """
    Checks if a PDF is readable and extracts text using the appropriate method.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        list: A list containing extracted text from each page (might be empty strings).
    """
    extracted_text = []
    ocr = ppocr(use_angle_cls=True, lang="en", use_gpu=True, verbose=False)
    try:
        pages = convert_from_path(pdf_path)
        # Use PaddleOCR for scanned images
        for page in pages:
            image = np.array(page)
            result = ocr.ocr(img=image, cls=True)[0]
            text = "\n".join([line[1][0] for line in result])
            extracted_text.append(text)
    except Exception as e:
        logger.error(f"Error opening PDF: {e}")
        return []

    return extracted_text


def create_temp_txt_file(text_to_save):
    """
    Creates a temporary text file and saves the given text in it.

    Args:
        text_to_save (str): The text to save in the temporary file.

    Returns:
        str: The path to the temporary text file (if successful), None otherwise.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
            temp_file.write(text_to_save.encode())  # Encode text for binary writing
            logger.debug(f"Temporary text file created at {temp_file.name}.")
            return temp_file.name
    except Exception as e:
        logger.error(f"Error creating temporary file: {e}")
        return None


def check_page_relevance(page_text, fields, thresh=0.1):
    """
    Checks if a page of text is relevant to given fields based on keyword matching.

    Args:
        page_text (str): Text content of the page.
        fields (list): List of field names or keywords.
        thresh (float): Threshold value between 0 and 1, where relevance score > thresh means relevant.

    Returns:
        bool: Whether the page text is deemed relevant to at least one field based on keyword matching.
    """

    try:
        # Convert page text and fields to lowercase for case-insensitive matching

        page_text_list = page_text.lower().split()
        page_text_set = set(page_text_list)
        keywords_list = []
        for field in fields:
            keywords_list.extend(field.lower().split())

        keywords_set = set(keywords_list)
        num_matched_fields = len(page_text_set.intersection(keywords_set))
        total_fields = len(keywords_set)
        return num_matched_fields >= thresh * total_fields

    except Exception as e:
        logger.error(f"Error calculating page relevance: {e}")
        return None


# Function to simulate raw data processing (replace with your actual logic)
def process_raw_data(file_path):
    try:
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        filename = os.path.basename(file_path)
        download_and_save_file(file_path, data_dir)
        local_filepath = os.path.join(data_dir, filename)

        extractor = TabularDataExtractor(filepath=local_filepath)
        output_filepath = extractor.extract_and_save_data()
        output_filepath = os.path.abspath(output_filepath)
        logger.info(
            f"Raw data processing successful. Output file saved at {output_filepath}."
        )
        return ProcessResponse(response="success", output_filepath=output_filepath)
    except Exception as e:
        logger.error(f"Error processing raw data: {e}")
        return None


# Function to simulate entity extraction (replace with your actual logic)
def get_entities(file_path, document_type):
    try:
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        os.mkdir(data_dir)
        filename = os.path.basename(file_path)
        csvs_folder = Path(os.path.join(MODEL_DATA_DIR,'tempCSV'))
        if not os.path.exists(csvs_folder):
            os.mkdir(csvs_folder)
        output_csv_filepath = os.path.join(csvs_folder, "".join(filename.split(".")[:-1]) + ".csv")
        output_csv_filepath = os.path.abspath(output_csv_filepath)
        
        
        mongo_record_id = CSVToMongo('dp_status').update_mongo_status(filename=filename,start=True)
        download_and_save_file(file_path, data_dir)
        local_filepath = os.path.join(data_dir, filename)
        output_filepath = os.path.join(data_dir, "".join(filename.split(".")[:-1]) + ".csv")
        output_filepath = os.path.abspath(output_filepath)
        extracted_texts = extract_page_texts(local_filepath)
        relevant_pages = []
        fields = config_manager.get_fields_from_filetype(document_type)
        
        for i, text in enumerate(extracted_texts):
            if check_page_relevance(page_text=text, fields=fields):
                relevant_pages.append(i)
        if not relevant_pages:
            logger.error(f"Provided document {filename} contains no relevant pages.")
            CSVToMongo('dp_status').update_mongo_status(filename=filename,id=mongo_record_id,success=False,start=False)
            raise Exception("Provided document contains no relevant pages.")

        final_df = pd.DataFrame()
        csv_text_all = ""
        formattinf_flag =False
        
        for relevant_page in relevant_pages:
            temp_filename = create_temp_txt_file(extracted_texts[relevant_page])
            summarizer = Summarizer(filepath=temp_filename, filetype=document_type)
            csv_text = summarizer.summarize()
            with open(output_filepath, "w+") as f:
                f.write(csv_text)
            with open(output_csv_filepath, "w+") as f:
                csv_text = csv_text.strip()
                csv_text += ";"+filename+";"+datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(csv_text)
                
            try:
                summarizer.csv_formatting(csv_file_path=output_filepath)
                df = pd.read_csv(output_filepath, delimiter=";")
                df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                df["filename"] = filename
                final_df = pd.concat([final_df, df])
            except:
                formattinf_flag =True
                csv_text_all += f"\n{csv_text}"
        if formattinf_flag:
            csv_to_mongo = CSVToMongo(document_type).push_raw_data(filename=filename,raw_text=csv_text_all)
            CSVToMongo('dp_status').update_mongo_status(filename=filename,id=mongo_record_id,success=False,start=False)
            return csv_text_all
        else:
            logger.debug(f"Writing final DataFrame to CSV {output_filepath}")
            final_df.to_csv(output_filepath, index=False, sep=";")
            logger.debug(f"Writing final DataFrame to Mango {output_filepath}")
            csv_to_mongo = CSVToMongo(document_type)
            csv_to_mongo.run(output_filepath)
            CSVToMongo('dp_status').update_mongo_status(filename=filename,id=mongo_record_id,success=True,start=False)
            return ProcessResponse(response="success", output_filepath=output_filepath)
    except Exception as e:
        logger.error(f"Error in getting entites: {e}")
        return None
        
    
# Health Check API
@app.get("/document_processor/api/health")
async def health_check():
    logger.info("health check request recevied.")
    try:
        logger.info("health check request successful")
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Error in health check : {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


# Raw Data Processing API
@app.post("/document_processor/api/process_raw_data", response_model=ProcessResponse)
async def process_raw_data_api(data: Document):
    try:
        return process_raw_data(data.file_path)
    except Exception as e:
        logger.error(f"Error during health check: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


# Processed Data (Entity Extraction) API
@app.post("/document_processor/api/get_entities", response_model=ProcessResponse)
async def get_entities_api(data: Document):
    try:
        logger.debug("Starting entity extraction process.")
        result = get_entities(data.file_path, data.document_type)
        logger.info("Entity extraction completed successfully.")
        return result
    except Exception as e:
        logger.error("Error occurred during entity extraction: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


# Function to simulate entity extraction (replace with your actual logic)
def get_entities_from_dir(document_type, document_dir):
    logger.debug("Starting to process files in directory: %s", document_dir)
    try:
        document_folder_full_path = os.path.join(INPUT_DATA_DIR, document_dir)
        
        csvs_folder = Path(os.path.join(MODEL_DATA_DIR,'tempCSV'))
        if os.path.exists(csvs_folder):
            shutil.rmtree(csvs_folder)
            os.mkdir(csvs_folder)
        files = os.listdir(Path(document_folder_full_path))
        for filename in tqdm(files):
            filepath = os.path.join(document_folder_full_path, filename)
            logger.info("Processing file: %s", filename)
            result = get_entities(file_path=filepath, document_type=document_type)
            try:
                if type(result)== str:
                    file_name = document_dir + ".csv"
                    result += (result +";"+filename+";"+str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                    output_filepath = os.path.join(MODEL_DATA_DIR,"raw_data",file_name)
                    with open(output_filepath, "w+") as f:
                        f.write(result)
                    logger.info("File processed with error, outputs are in csv for the file: %s", filename)
                else:
                    logger.info("File processed successfully, %s", filename)
            except Exception as e:
                logger.error(
                    "Error occurred while processing file: %s - %s", filepath, str(e)
                )
        # Combine all text files into one
        combined_content = []
        for file_name in os.listdir(csvs_folder):
            if file_name.endswith(".csv"):
                file_path = os.path.join(csvs_folder, file_name)
                with open(file_path, "r") as f:
                    content = f.readlines()
                    combined_content.extend(content)
        raw_data_dir = Path(os.path.join(MODEL_DATA_DIR,'raw_data'))
        if not os.path.exists(raw_data_dir):
            os.mkdir(raw_data_dir)
        combined_file_path = os.path.join(raw_data_dir, f"{document_dir}.csv")
        
        with open(combined_file_path, "w") as f:
            combined_content_txt = "\n".join(combined_content)
            f.writelines(combined_content_txt)
            
        return ProcessResponse(
            response="success",
            output_filepath=f"Succesfully processed all files in {document_folder_full_path}",
        )
    except Exception as e:
        logger.error(
            "An error occurred while processing files in directory: %s - %s",
            document_dir,
            str(e),
        )


# Processed Data (Entity Extraction) API
@app.post(
    "/document_processor/api/get_entities_from_dir", response_model=ProcessResponse
)
async def get_entities_from_dir_api(data: Document):
    try:
        logger.info(f"Starting to process file from directory: {data.document_dir} for document type: {data.document_type}.")

        result = get_entities_from_dir(
            document_type=data.document_type, document_dir=data.document_dir
        )
        logger.info(f"Succesfully processed all files in {data.document_type}")
        return result
    except Exception as e:
        logger.error("Error occurred during file processing: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
