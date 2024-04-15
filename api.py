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
import pytesseract
from PIL import Image
import numpy as np
import pandas as pd
from paddleocr import PaddleOCR as ppocr
from img2table.ocr import PaddleOCR
from img2table.document import PDF
import os
import zipfile
import shutil


class TextExtractor:
    def __init__(self, filepath) -> None:
        self._filepath = filepath
        self._text_filepath = "./temp/invoice_text.txt"
        self._header_filepath = "./temp/vendor_information.txt"
        self.ppocr_obj = ppocr(
            use_angle_cls=True, lang="en", use_gpu=True, verbose=False
        )

    def is_file_readable(self):
        total_words = 0
        doc = fitz.open(self._filepath)
        for page in doc:
            text = page.get_text()
            total_words += len(text.split())

        if total_words < 100:
            return False
        return True

    def extract_and_save_txt(self):
        text_file = open(self._text_filepath, "w+")
        if not self.is_file_readable():
            ocr_text = []
            pages = convert_from_path(self._filepath)
            for page in pages:
                # text = pytesseract.image_to_string(page)
                image = np.array(page)
                result = self.ppocr_obj.ocr(img=image, cls=True)[0]
                text = "\n".join([line[1][0] for line in result])
                if text.lower().__contains__("invoice"):
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
        header_text = pytesseract.image_to_string(img[:header_y, :, :])
        with open(self._header_filepath, "w+") as header_file:
            header_file.write(header_text.strip())


class LLMUtils:
    def __init__(self) -> None:
        self._llm = self.load_llm()
        self._embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
        self.service_context = ServiceContext.from_defaults(
            llm=self._llm,
            embed_model=self._embed_model,
        )

    def load_llm(self):
        return LlamaCPP(
            model_path="../models/laser-dolphin-mixtral-2x7b-dpo.Q4_K_M.gguf",
            temperature=0,
            max_new_tokens=512,
            context_window=3900,
            generate_kwargs={},
            model_kwargs={"n_gpu_layers": 8},
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            verbose=True,
        )


class Summarizer(TextExtractor):
    def __init__(self, filepath) -> None:
        super().__init__(filepath)
        is_file_readable = self.is_file_readable()
        if is_file_readable:
            self.extract_header_ocr()
            documents = SimpleDirectoryReader(
                input_files=[filepath, self._header_filepath]
            ).load_data()
        else:
            self.extract_and_save_txt()
            documents = SimpleDirectoryReader(
                input_files=[self._text_filepath]
            ).load_data()
        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize",
            use_async=True,
        )
        self.index = VectorStoreIndex.from_documents(
            documents=documents,
            response_synthesizer=response_synthesizer,
        )
        self.query_engine = self.index.as_query_engine(
            response_mode="tree_summarize", use_async=True, streaming=False
        )

    def summarize(self):
        response = self.query_engine.query(
            "Using the information in the provided documents, What is the purchase order number (sometimes also called work order number), the invoice number (sometimes also called debit note number), the invoice amount, the invoice currency, the invoice description or details, the Jio/Reliance entity name, the Vendor name, the Vendor country and invoice date? Please format the response in bullet points."
        )
        return response.response


class DocumentsProcessor:
    def __init__(self, zip_filepath, data_dir) -> None:
        self.ocr = PaddleOCR(lang="en", kw={"use_gpu": True})
        self.zip_filepath = zip_filepath
        self.zip_files_dir = os.path.dirname(self.zip_filepath)
        self.extracted_data_dir = data_dir

    def extract_files_from_zip(self):
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
        for filename in os.listdir(self.zip_files_dir):
            filepath = os.path.join(self.zip_files_dir, filename)
            if filename.endswith("pdf"):
                doc = PDF(src=filepath)
                output_filename = filename.replace("pdf", "xlsx")

                output_filepath = os.path.join(self.extracted_data_dir, output_filename)
                doc.to_xlsx(
                    dest=output_filepath,
                    ocr=self.ocr,
                    implicit_rows=False,
                    borderless_tables=False,
                    min_confidence=50,
                )
            elif filename.endswith("xlsx") or filename.endswith("csv"):
                # move file to destination directory
                shutil.copy(filepath, self.extracted_data_dir)
            else:
                continue

    def save_extracted_data(self, save_filepath):
        # extract files from zip archive
        self.extract_files_from_zip()
        # extract data
        self.extract_data_from_files()
        # save files as zip in save_filepath
        shutil.make_archive(save_filepath, "zip", self.extracted_data_dir)

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
