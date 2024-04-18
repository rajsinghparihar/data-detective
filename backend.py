from api import (
    Summarizer,
    LLMUtils,
    DuplicateChecker,
    DocumentsProcessor,
    TabularDataExtractor,
)
from llama_index import set_global_tokenizer, set_global_service_context
from transformers import AutoTokenizer
from flask import Flask, request, jsonify
import shutil
import os
import pandas as pd

app = Flask(__name__)
llm_module = LLMUtils()
set_global_tokenizer(
    AutoTokenizer.from_pretrained("macadeliccc/laser-dolphin-mixtral-2x7b-dpo").encode
)
set_global_service_context(service_context=llm_module.service_context)

temp_filepath = "./temp/temp.pdf"
support_directory = "./temp/support/"
temp_zip_filepath = "./temp/uploaded_files/files.zip"
extracted_data_dir = "./temp/extracted_data"
extracted_data_filepath = "./temp/extracted_data"
invoice_csv_filepath = "./temp/invoice.csv"

files_dir = "./v2"

summarized_pdf_csv_filepath = "./v2/file.csv"
extracted_data_dir_v2 = "./v2/extracted_data"
uploaded_filepath = "./v2/file.pdf"
db_dir = "./v2/db_dir"

for dir in [files_dir,extracted_data_dir_v2,db_dir]:
    if not os.path.exists(dir):
        os.mkdir(dir)


@app.route("/", methods=["GET"])
def index():
    return """<!DOCTYPE html>
  <html>
  <body>
    <a href='http://127.0.0.1:5000/streamlit'>Go to prediction app</a>
  </body>
  </html>"""


@app.route("/summarize_invoice", methods=["GET"])
def summarize_invoice():
    uploaded_files_directory = os.path.dirname("./temp/uploaded_files/files.zip")
    filenames = os.listdir(uploaded_files_directory)
    if len(filenames) == 0:
        return jsonify({"error": "No documents uploaded!"})
    response = ""
    for filename in filenames:
        if filename.endswith("pdf"):
            summarizer = Summarizer(
                filepath=os.path.join(uploaded_files_directory, filename)
            )
            response += (
                f"\n### Here is the invoice summary for {filename}:\n"
                + summarizer.summarize()
            )

    if response == "":
        return jsonify({"error": "Uploaded files contain no Invoice pdfs"})

    return jsonify({"response": response})


@app.route("/process_documents", methods=["POST"])
def process_docs():
    shutil.rmtree(os.path.dirname(temp_zip_filepath))
    os.mkdir(os.path.dirname(temp_zip_filepath))
    shutil.rmtree(extracted_data_dir)
    os.mkdir(extracted_data_dir)
    if os.path.exists(extracted_data_dir + ".zip"):
        os.remove(extracted_data_dir + ".zip")
    if "files" not in request.files:
        return jsonify({"error": "No zip file uploaded!"})

    zip_fs = request.files["files"]

    zip_fs.save(temp_zip_filepath)
    doc_processor = DocumentsProcessor(
        zip_filepath=temp_zip_filepath, data_dir=extracted_data_dir
    )

    # save extracted data from the document processor
    ret = doc_processor.save_extracted_data(extracted_data_filepath)

    if ret:
        return jsonify(
            {
                "response": "Data extracted successfully",
                "output_filepath": extracted_data_filepath + ".zip",
            }
        )


@app.route("/invoice/summarize_invoice", methods=["POST"])
def summarize_invoice_pdf():
    if "files" not in request.files:
        return jsonify({"error": "No pdf uploaded!"})

    invoice_fs = request.files["invoice"]

    invoice_fs.save(temp_filepath)
    summarizer = Summarizer(filepath=temp_filepath)
    csv_text = summarizer.summarize()
    with open(invoice_csv_filepath, "w+") as f:
        f.write(csv_text)

    return jsonify({"output_filepath": invoice_csv_filepath})


@app.route("/v2/summarize_data", methods=["POST"])
def summarize_document():
    request_data = request.get_json()
    filetype = request_data["filetype"]
    summarizer = Summarizer(filepath=uploaded_filepath, filetype=filetype)
    csv_text = summarizer.summarize()
    with open(summarized_pdf_csv_filepath, "w+") as f:
        f.write(csv_text)

    return jsonify({"output_filepath": summarized_pdf_csv_filepath})


@app.route("/v2/extract_data", methods=["POST"])
def extract_data():
    if "files" not in request.files:
        return jsonify({"error": "No pdf uploaded!"})

    document_fs = request.files["files"]
    document_fs.save(uploaded_filepath)

    extractor = TabularDataExtractor(filepath=uploaded_filepath)
    output_filepath = extractor.extract_and_save_data()
    return jsonify({"output_filepath": output_filepath})


@app.route("/check_duplicates", methods=["GET"])
def check_duplicates():
    uploaded_files_directory = os.path.dirname("./temp/uploaded_files/files.zip")
    filepaths = os.listdir(uploaded_files_directory)
    filepaths = [
        os.path.join(uploaded_files_directory, filepath) for filepath in filepaths
    ]
    if len(filepaths) == 0:
        return jsonify({"error": "No documents uploaded!"})

    duplicate_checker = DuplicateChecker(
        files=filepaths,
        unique_col="SAP ID",
    )

    ret, sap_ids = duplicate_checker.check_duplicates()
    sap_ids_df = pd.DataFrame({"SAP ID": sap_ids})
    duplicates_filepath = "./temp/duplicates_dir/duplicates.csv"
    sap_ids_df.to_csv(duplicates_filepath, index=False)
    if ret:
        response = "### Annexure contains duplicate records!"
        return jsonify({"response": response, "output_filepath": duplicates_filepath})
    else:
        response = "### Annexure Contains No Duplicates!"
        return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True)
