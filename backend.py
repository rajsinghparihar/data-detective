from api import Summarizer, LLMUtils, DuplicateChecker, DocumentsProcessor
from llama_index import set_global_tokenizer, set_global_service_context
from transformers import AutoTokenizer
from flask import Flask, request, jsonify
import shutil
import os

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
    if ret:
        response = f"### Annexure contains duplicate records!\n- SAP IDs: {sap_ids}"
    else:
        response = "### Annexure Contains No Duplicates!"

    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True)
