import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:5000/"


@st.experimental_fragment
def show_download_button(output_path):
    with open(output_path, "rb") as zip_file_bytes:
        st.download_button(
            label="Download Zipped Archive",
            data=zip_file_bytes,
            file_name="extracted_data.zip",
        )


st.title("Document Processor")
st.write("Upload a zip file containing all the documents")
uploaded_zip_file = st.file_uploader("Choose a zip file", type=["zip"])
if uploaded_zip_file is not None and st.button("Process"):
    with open(uploaded_zip_file.name, "wb") as f:
        f.write(uploaded_zip_file.getbuffer())
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
            label="Data Extracted Successfully!", state="complete", expanded=True
        )
        result = response.json()
        if "error" in result:
            st.error(result["error"])
        else:
            st.success("Prediction complete!")
            output_path = result["output_filepath"]
            show_download_button(output_path=output_path)

    with st.status("Checking Duplicates...") as status:
        response = requests.get(BACKEND_URL + "check_duplicates")
        status.update(
            label="Duplicates checking completed!", state="complete", expanded=True
        )
        result = response.json()
        if "error" in result:
            st.error(result["error"])
        else:
            st.success("Prediction complete!")
            st.markdown(result["response"])

    with st.status("Summarizing Invoices...") as status:
        response = requests.get(BACKEND_URL + "summarize_invoice")

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
            st.success("Prediction complete!")
            st.markdown(result["response"])
