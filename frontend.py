import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:5000/"


@st.experimental_fragment
def show_download_button(output_path, label, filename):
    with open(output_path, "rb") as zip_file_bytes:
        st.download_button(
            label=label,
            data=zip_file_bytes,
            file_name=filename,
        )


@st.experimental_fragment
def invoice_processing():
    st.title("Invoice Analysis")
    st.write("Upload a invoice pdf to get extracted info")
    uploaded_pdf_file = st.file_uploader("Choose a file", type=["pdf"])
    if uploaded_pdf_file is not None and st.button("Process Invoice"):
        files = {
            "files": (
                uploaded_pdf_file.name,
                uploaded_pdf_file.read(),
            )
        }
        filename = uploaded_pdf_file.name
        with st.status("Summarizing Invoice...") as status:
            response = requests.post(
                BACKEND_URL + "/invoice/summarize_invoice", files=files
            )
            result = response.json()
            if "error" in result:
                status.update(
                    label="Invoice Summarization Failed!",
                    state="error",
                    expanded=True,
                )
                st.error(result["error"])
            else:
                status.update(
                    label="Invoice Summarization completed!",
                    state="complete",
                    expanded=True,
                )
                output_path = result["output_filepath"]
                show_download_button(
                    output_path=output_path,
                    label="Download Invoice Info. CSV file",
                    filename=filename.replace("pdf", "csv"),
                )


@st.experimental_fragment
def duplicate_detection():
    st.title("Duplicate Analysis")
    st.write("Upload a zip file containing all the documents")
    uploaded_zip_file = st.file_uploader("Choose a zip file", type=["zip"])
    if uploaded_zip_file is not None and st.button("Process"):
        files = {
            "files": (
                uploaded_zip_file.name,
                uploaded_zip_file.read(),
            )
        }
        with st.status("Extracting data...") as status:
            response = requests.post(
                BACKEND_URL + "process_documents",
                files=files,
            )
            status.update(
                label="Data Extracted Successfully!",
                state="complete",
                expanded=True,
            )
            result = response.json()
            if "error" in result:
                st.error(result["error"])
            else:
                output_path = result["output_filepath"]
                show_download_button(
                    output_path=output_path,
                    label="Download Zipped Archive",
                    filename="extracted_data.zip",
                )

        with st.status("Checking Duplicates...") as status:
            response = requests.get(BACKEND_URL + "check_duplicates")
            status.update(
                label="Duplicates checking completed!",
                state="complete",
                expanded=True,
            )
            result = response.json()
            if "error" in result:
                st.error(result["error"])
            else:
                if "output_filepath" in result:
                    output_path = result["output_filepath"]
                    show_download_button(
                        output_path=output_path,
                        label="Download Duplicates CSV file",
                        filename="duplicates.csv",
                    )
                    st.markdown(result["response"])
                else:
                    st.markdown(result["response"])


# st.set_page_config(layout="wide")
# invoice_summary_col, duplicate_detection_col = st.columns(2)
# with invoice_summary_col:
#     invoice_processing()
# with duplicate_detection_col:
#     duplicate_detection()

st.title("Document Processor")
st.write("Upload a pdf to process")
filetype = st.selectbox(
    "Select the filetype", ["Invoice", "SalarySlip", "Work Completion Certificate"]
)
st.write("Upload a pdf to process")
uploaded_pdf_file = st.file_uploader("Choose a file", type=["pdf"])
if uploaded_pdf_file is not None and st.button("Process"):
    files = {
        "files": (
            uploaded_pdf_file.name,
            uploaded_pdf_file.read(),
        )
    }
    filename = uploaded_pdf_file.name
    with st.status("Extracting data...") as status:
        response = requests.post(
            BACKEND_URL + "v2/extract_data",
            files=files,
        )
        status.update(
            label="Data Extracted Successfully!",
            state="complete",
            expanded=True,
        )
        result = response.json()
        if "error" in result:
            st.error(result["error"])
        else:
            output_path = result["output_filepath"]
            show_download_button(
                output_path=output_path,
                label="Download raw data",
                filename=uploaded_pdf_file.name.replace("pdf", "xlsx"),
            )
    with st.status("Summarizing File...") as status:
        response = requests.post(
            BACKEND_URL + "v2/summarize_data", json={"filetype": filetype}
        )
        result = response.json()
        if "error" in result:
            status.update(
                label="Summarization Failed!",
                state="error",
                expanded=True,
            )
            st.error(result["error"])
        else:
            status.update(
                label="Summarization completed!",
                state="complete",
                expanded=True,
            )
            output_path = result["output_filepath"]
            show_download_button(
                output_path=output_path,
                label="Download formatted data",
                filename=filename.replace("pdf", "csv"),
            )
