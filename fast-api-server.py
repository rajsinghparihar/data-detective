from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import Optional
import os
from api import Summarizer, LLMUtils, TabularDataExtractor, CSVToMongo, ConfigManager
from llama_index import set_global_tokenizer, set_global_service_context
from transformers import AutoTokenizer
import requests
import re
import shutil
from dotenv import load_dotenv
from paddleocr import PaddleOCR as ppocr
from pdf2image import convert_from_path
import numpy as np
import tempfile
import pandas as pd
from datetime import datetime

load_dotenv()


app = FastAPI()
llm_module = LLMUtils()
set_global_tokenizer(
    AutoTokenizer.from_pretrained("macadeliccc/laser-dolphin-mixtral-2x7b-dpo").encode
)
set_global_service_context(service_context=llm_module.service_context)
config_manager = ConfigManager()

MODEL_DATA_DIR = os.getenv("MODEL_DATA_DIR")
INPUT_DATA_DIR = os.getenv("INPUT_DATA_DIR")
data_dir = os.path.join(MODEL_DATA_DIR, "outputs/")


class Document(BaseModel):
    file_path: Optional[str] = ""
    document_type: Optional[str] = ""


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
            raise ValueError(f"Error downloading file from URL: {source}") from e

        # Get filename from URL or response headers
        filename = os.path.basename(source)
        content_disposition = response.headers.get("content-disposition")
        if content_disposition:
            filename = re.findall(r"filename=(.*)", content_disposition)[0]
    else:
        # Assume local path or remote file path (not starting with http)
        if not os.path.exists(source):
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
    print(f"File '{filename}' downloaded from '{source}' and saved to '{save_dir}'.")


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
        print(f"Error opening PDF: {e}")
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
            return temp_file.name
    except Exception as e:
        print(f"Error creating temporary file: {e}")
        return None


def check_page_relevance(page_text, fields, thresh=0.1):
    page_text_list = page_text.lower().split()
    page_text_set = set(page_text_list)
    keywords_list = []
    for field in fields:
        keywords_list.extend(field.lower().split())

    keywords_set = set(keywords_list)
    num_matched_fields = len(page_text_set.intersection(keywords_set))
    total_fields = len(keywords_set)
    return num_matched_fields >= thresh * total_fields


# Function to simulate raw data processing (replace with your actual logic)
def process_raw_data(file_path):
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.mkdir(data_dir)
    filename = os.path.basename(file_path)
    download_and_save_file(file_path, data_dir)
    local_filepath = os.path.join(data_dir, filename)
    extractor = TabularDataExtractor(filepath=local_filepath)
    output_filepath = extractor.extract_and_save_data()
    output_filepath = os.path.abspath(output_filepath)
    return ProcessResponse(response="success", output_filepath=output_filepath)


# Function to simulate entity extraction (replace with your actual logic)
def get_entities(file_path, document_type):
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.mkdir(data_dir)
    filename = os.path.basename(file_path)
    download_and_save_file(file_path, data_dir)
    local_filepath = os.path.join(data_dir, filename)
    extracted_texts = extract_page_texts(local_filepath)
    relevant_pages = []
    fields = config_manager.get_fields_from_filetype(document_type)
    for i, text in enumerate(extracted_texts):
        if check_page_relevance(page_text=text, fields=fields):
            relevant_pages.append(i)
    if relevant_pages == []:
        raise Exception("Provided document contains no relevant pages")
    final_df = pd.DataFrame()
    for relevant_page in relevant_pages:
        temp_filename = create_temp_txt_file(extracted_texts[relevant_page])
        summarizer = Summarizer(filepath=temp_filename, filetype=document_type)
        csv_text = summarizer.summarize()
        output_filepath = os.path.join(
            data_dir, "".join(filename.split(".")[:-1]) + ".csv"
        )
        with open(output_filepath, "w+") as f:
            f.write(csv_text)
        output_filepath = os.path.abspath(output_filepath)
        summarizer.csv_formatting(csv_file_path=output_filepath)

        df = pd.read_csv(output_filepath, delimiter=";")
        df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df["filename"] = filename

        final_df = pd.concat([final_df, df])
    try:
        output_filepath = os.path.join(
            data_dir, "".join(filename.split(".")[:-1]) + ".csv"
        )
        output_filepath = os.path.abspath(output_filepath)
        final_df.to_csv(output_filepath, index=False, sep=";")
        csv_to_mongo = CSVToMongo(document_type)
        csv_to_mongo.run(output_filepath)

        return ProcessResponse(response="success", output_filepath=output_filepath)
    except Exception as e:
        return ProcessResponse(response="error", output_filepath=e)


# Health Check API
@app.get("/document_processor/api/health")
async def health_check():
    try:
        return {"status": "healthy"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


# Raw Data Processing API
@app.post("/document_processor/api/process_raw_data", response_model=ProcessResponse)
async def process_raw_data_api(data: Document):
    try:
        return process_raw_data(data.file_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


# Processed Data (Entity Extraction) API
@app.post("/document_processor/api/get_entities", response_model=ProcessResponse)
async def get_entities_api(data: Document):
    try:
        return get_entities(data.file_path, data.document_type)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


# Function to simulate entity extraction (replace with your actual logic)
def get_entities_from_dir(document_type):
    print(INPUT_DATA_DIR)
    print(os.listdir(INPUT_DATA_DIR))
    # return "successfull call"
    for filename in os.listdir(INPUT_DATA_DIR):
        filepath = os.path.join(INPUT_DATA_DIR, filename)
        try:
            get_entities(file_path=filepath, document_type=document_type)
        except Exception as e:
            print(e)
            pass
    return ProcessResponse(
        response="success",
        output_filepath=f"Succesfully processed all files in {INPUT_DATA_DIR}",
    )


# Processed Data (Entity Extraction) API
@app.post(
    "/document_processor/api/get_entities_from_dir", response_model=ProcessResponse
)
async def get_entities_from_dir_api(data: Document):
    try:
        print(data)
        return get_entities_from_dir(data.document_type)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
