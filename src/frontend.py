import streamlit as st
import requests
from config import ConfigManager
from dotenv import load_dotenv
import os
import shutil
import zipfile
import json
from streamlit_pdf_viewer import pdf_viewer
from glob import glob
from pymongo import MongoClient
from bson.objectid import ObjectId
import pandas as pd


st.set_page_config(layout="wide")
load_dotenv(dotenv_path="./src/.env", override=True)
cm = ConfigManager()


class StreamlitMongoClient:
    def __init__(self, collection_name) -> None:
        self.mongo_uri = os.getenv("MONGO_URI")
        self.db_name = os.getenv("MONGO_DB_NAME")
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[collection_name]

    def get_data(self, query, projection={}):
        documents = self.collection.find(query, projection=projection)
        return documents

    def push_data(self, id: str, data):
        id = ObjectId(id)
        self.collection.find_one_and_replace(
            filter={"_id": id},
            replacement=data,
        )


mongo_status_utils = StreamlitMongoClient(collection_name="dp_status")
st.session_state["process_id"] = ""

BACKEND_URL = "http://localhost:8501/"


TEMP_FILES_DIR = cm.INPUT_DIR


def save_file(file):
    with open(os.path.join(TEMP_FILES_DIR, file.name), "wb") as f:
        f.write(file.getvalue())


def get_filetypes():
    filetypes = []
    for obj in cm.entity_config_data["configs"]:
        filetypes.append(obj["type"])

    st.session_state["filetypes"] = filetypes


def get_fields_from_filetype(filetype):
    for obj in cm.entity_config_data["configs"]:
        if obj["type"] == filetype:
            return obj["fields"]


def extract_zip(file):
    with zipfile.ZipFile(file, "r") as zip_ref:
        zip_ref.extractall(TEMP_FILES_DIR)

    folders = []
    for filename in os.listdir(TEMP_FILES_DIR):
        if not filename.endswith(".pdf") and not filename.endswith(".zip"):
            folders.append(filename)

    files = []
    for folder_name in folders:
        for filename in os.listdir(f"{TEMP_FILES_DIR}/{folder_name}/"):
            if filename.startswith("."):
                continue
            elif filename.endswith(".pdf"):
                files.append(f"{TEMP_FILES_DIR}/{folder_name}/{filename}")
    return files


def save_new_entity_config(filetype, fields):
    data = cm.entity_config_data
    update = False
    for i, entities_obj in enumerate(data["configs"]):
        if entities_obj["type"] == filetype:
            update = True
            index = i
            break

    if update:
        data["configs"][index]["fields"] = fields
    else:
        data["configs"].append({"type": filetype, "fields": fields})

    data_str = json.dumps(data, indent=4)
    with open(cm.ENTITY_CONFIG_FILEPATH, "w") as f:
        f.write(data_str)


@st.experimental_fragment
def output_section_fragment():
    pdf_column, data_column = st.columns([0.7, 0.3])
    files = glob(f"{TEMP_FILES_DIR}/**/*.pdf", recursive=True)

    with pdf_column:
        with st.container(border=True):
            st.header("Files Section")
            filepath = st.selectbox(label="Select File", options=files)
            process_id = st.text_input(label="Enter the process id")
            if filepath:
                pdf_viewer(input=filepath)

    with data_column:
        with st.container(border=True):
            st.header("Data Section")
            data_documents = mongo_utils.get_data(
                query={
                    "process_id": process_id,
                    "filename": os.path.basename(filepath),
                }
            )
            df = pd.DataFrame(data=[doc for doc in data_documents])

            df = df.transpose()
            edited_df = st.data_editor(data=df, use_container_width=True)
            # st.write("Edited DF: ", edited_df)

            submit_col, reset_col = st.columns(2)
            with submit_col:
                if st.button("Update Data"):
                    edited_json = edited_df.to_dict()

                    edited_json = edited_json["0"]
                    object_id = edited_json["_id"]
                    del edited_json["_id"]

                    mongo_utils.push_data(id=object_id, data=edited_json)
                    st.success("Data Updated in MongoDB Successfully.")

            # with reset_col:
            #     if st.button("Reset"):
            #         st.rerun()


@st.experimental_fragment
def input_section_fragment():
    global mongo_utils, filepaths
    new_config_column, submit_job_column = st.columns([0.4, 0.6])

    with submit_job_column:
        with st.container(border=True):
            st.header("Input Section")
            st.write("Upload a document or a zipped archive to process")
            get_filetypes()
            col1, col2 = st.columns(2)
            with col1:
                filetype = st.selectbox(
                    "Select the filetype", options=st.session_state["filetypes"]
                )
                mode = st.radio(
                    label="Mode", options=["Single", "Batch"], horizontal=True
                )
                mongo_utils = StreamlitMongoClient(collection_name=filetype)
                if filetype:
                    st.write(
                        "Extractable Entities for the selected document type:",
                        get_fields_from_filetype(filetype=filetype),
                    )
            with col2:
                uploaded_pdf_file = st.file_uploader(
                    "Choose a file", type=["pdf", "zip"]
                )

            if uploaded_pdf_file is not None and st.button("Submit Job"):
                # clear dir for processing new request
                if os.path.exists(TEMP_FILES_DIR):
                    shutil.rmtree(TEMP_FILES_DIR)
                os.mkdir(TEMP_FILES_DIR)

                filename = uploaded_pdf_file.name
                save_file(uploaded_pdf_file)
                if filename.endswith(".pdf"):
                    filepath = os.path.join(TEMP_FILES_DIR, filename)
                else:
                    filepath = os.path.join(TEMP_FILES_DIR, filename)
                    filepaths = extract_zip(file=filepath)

                with st.status("Submitting Job...") as status:
                    try:
                        if mode == "Single":
                            request_data = {
                                "document_type": filetype,
                                "document_path": filepath,
                                "document_dir": "",
                                "struct_type": "",
                            }
                            response = requests.post(
                                BACKEND_URL + "invoice-processing/api/get_entities",
                                json=request_data,
                            )
                        else:
                            request_data = {
                                "document_type": filetype,
                                "document_dir": TEMP_FILES_DIR,
                            }
                            response = requests.post(
                                BACKEND_URL
                                + "invoice-processing/api/get_entities_from_dir",
                                json=request_data,
                            )
                        result = response.json()
                        if result.keys().__contains__("detail"):
                            error_msg = result["detail"]
                            status.update(
                                label="Job Submission Failed!",
                                state="error",
                                expanded=True,
                            )
                            st.error(f"Error: {error_msg}")
                            st.write("Request Params:", request_data)
                        else:
                            process_id = result["process_id"]
                            st.write(result)
                            st.code(body=process_id)
                            st.session_state["process_id"] = process_id
                            status.update(
                                label="Job Submitted Successfully",
                                state="complete",
                                expanded=True,
                            )
                    except Exception as e:
                        status.update(
                            label="Job Submission Failed!",
                            state="error",
                            expanded=True,
                        )
                        st.error(f"Error: {e}")
                        st.write("Request Params:", request_data)
                        # notify(f"Error: {e}")

    with new_config_column:
        with st.container(border=True):
            st.header("Config Section")
            new_filetype = st.text_input(label="Specify Document Type")
            new_fields = st.text_input(
                label="Specify the fields to be extracted from document, shoud be comma separated"
            )

            new_fields = [field.strip() for field in new_fields.split(",")]

            if st.button(label="Submit Config"):
                save_new_entity_config(new_filetype, new_fields)
                get_filetypes()
                st.rerun()


@st.experimental_fragment()
def status_fragment():
    with st.container(border=True):
        st.header("Status Section")
        process_id_or_query = st.text_input(
            label="Enter the process id of the job or MongoDB query"
        )

        if process_id_or_query:
            if process_id_or_query.startswith("{"):
                query = json.loads(process_id_or_query)
                print(query)
                documents = mongo_status_utils.get_data(
                    query=query, projection={"_id": 0}
                )
            else:
                documents = mongo_status_utils.get_data(
                    query={"process_id": process_id_or_query}, projection={"_id": 0}
                )
        else:
            documents = mongo_status_utils.get_data(query={}, projection={"_id": 0})
        st.dataframe(data=documents, use_container_width=True)


@st.experimental_fragment
def start_service_fragment():
    start_service_btn = st.button(
        "Start Service", type=st.session_state.get("submit_btn_type", "primary")
    )

    if start_service_btn:
        with st.status("Starting Service...") as service_status:
            try:
                response = requests.get(BACKEND_URL + "invoice-processing/api/start")
                service_status.update(
                    label="Service Started Successfully!", state="complete"
                )
                st.session_state["submit_btn_type"] = "secondary"
                st.success(response)
            except Exception as e:
                service_status.update(
                    label="Service Initialization Failed.", state="error"
                )
                st.session_state["submit_btn_type"] = "primary"
                st.error(e)


st.title("Document Processor")
start_service_fragment()

input_section, status_section, output_section = st.tabs(
    ["Input Section", "Status", "Output Section"]
)


filepaths = []

with input_section:
    input_section_fragment()
with status_section:
    status_fragment()
with output_section:
    output_section_fragment()
