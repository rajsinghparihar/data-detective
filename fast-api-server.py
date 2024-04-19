from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import Optional
import os
from api import (
    Summarizer,
    LLMUtils,
    TabularDataExtractor,
)
from llama_index import set_global_tokenizer, set_global_service_context
from transformers import AutoTokenizer
import requests
import re
import shutil

from dotenv import load_dotenv

load_dotenv()


app = FastAPI()
llm_module = LLMUtils()
set_global_tokenizer(
    AutoTokenizer.from_pretrained("macadeliccc/laser-dolphin-mixtral-2x7b-dpo").encode
)
set_global_service_context(service_context=llm_module.service_context)

DATA_DIR = os.getenv("DATA_DIR")
data_dir = os.path.join(DATA_DIR, "outputs/")
print(data_dir)


class Document(BaseModel):
    file_path: str
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
    summarizer = Summarizer(filepath=local_filepath, filetype=document_type)
    csv_text = summarizer.summarize()
    output_filepath = os.path.join(data_dir, filename.replace(".pdf", ".csv"))
    with open(output_filepath, "w+") as f:
        f.write(csv_text)
    output_filepath = os.path.abspath(output_filepath)
    summarizer.csv_formatting(csv_file_path=output_filepath)
    return ProcessResponse(response="success", output_filepath=output_filepath)


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
