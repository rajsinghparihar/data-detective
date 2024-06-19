from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.postprocessor import SentenceTransformerRerank
from llama_index.response_synthesizers import get_response_synthesizer
from src.prompts import (
    general_prompt,
    invoice_prompt,
    mobile_invoice_prompt,
)
from src.config import ConfigManager
from src.utils import Utils
from PIL import Image
from paddleocr import PaddleOCR as ppocr
from pdf2image import convert_from_path
import os
import fitz
import numpy as np
from typing import List, Optional
import tabula
from unstructured.partition.pdf import partition_pdf


class TextExtractor:
    def __init__(self, filepath) -> None:
        self._filepath = filepath
        self.cm = ConfigManager()
        self.ppocr_obj = ppocr(
            use_angle_cls=True,
            lang="en",
            use_gpu=True,
            verbose=False,
            det_model_dir=os.path.join(self.cm.MODELS_DIR, "ocr/det/"),
            rec_model_dir=os.path.join(self.cm.MODELS_DIR, "ocr/rec/"),
            cls_model_dir=os.path.join(self.cm.MODELS_DIR, "ocr/cls/"),
        )
        self.pdf_elements = partition_pdf(self._filepath, strategy="hi_res")

    def is_file_readable(self):
        total_words = 0
        doc = fitz.open(self._filepath)
        for page in doc:
            text = page.get_text()
            total_words += len(text.split())

        if total_words < 100:
            return False
        return True

    def get_tabular_data_json_str(self):
        # ASSUMPTION [Page 1 contains the invoice]
        dfs = tabula.read_pdf(self._filepath, pages=1)
        required_df = None
        for df in dfs:
            if not df.empty:
                required_df = df
        try:
            df_json = required_df.to_json(orient="records", indent=4)
        except Exception as e:
            print("Error in TextExtractor.get_tabular_data_json_str", e)
            df_json = ""
        return df_json

    def get_tabular_data_raw_text(self):
        def conditions(el):
            el_text = str(el).lower()
            # ASSUMPTION [tabular data or invoice amount, currency data contains these keywords]
            return el_text.__contains__("total") or el_text.__contains__("amount")

        raw_text = "\n".join([str(el) for el in self.pdf_elements if conditions(el)])
        return raw_text

    def get_address_info(self):
        def conditions(el):
            el_text = str(el).lower()
            # ASSUMPTION [address information or TO or FROM parts of the pdf contain these keywords]
            return el_text.__contains__("address") or el_text.__contains__("from")

        address_info = "\n".join(
            [str(el) for el in self.pdf_elements if conditions(el)]
        )
        return address_info

    def extract_and_save_txt(self, filetype_keyword: str):
        """
        - filetype_keyword: str
            - examples: ["Invoice", "Customer", etc.]
            only used when file is not readable
            and values must be extracted using ocr
        """
        text_file = open(self._text_filepath, "w+")
        if not self.is_file_readable():
            ocr_text = []
            pages = convert_from_path(self._filepath)
            for page in pages:
                image = np.array(page)
                result = self.ppocr_obj.ocr(img=image, cls=True)[0]
                text = "\n".join([line[1][0] for line in result])
                if text.lower().__contains__(filetype_keyword.lower()):
                    ocr_text.append(text)
                    break
            text_file.writelines(ocr_text)
        text_file.close()

    def extract_header_ocr(self):
        doc = fitz.open(self._filepath)
        page = doc[0]
        # Assuming header is in top quarter
        header_y = int(page.mediabox[3] / 4)
        pix = page.get_pixmap(matrix="RGB")
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = np.array(img)
        header_result = self.ppocr_obj.ocr(img=img[:header_y, :, :], cls=True)[0]
        header_text = "\n".join([line[1][0] for line in header_result])
        with open(self._header_filepath, "w+") as header_file:
            header_file.write(header_text.strip())

    def extract_page_texts(self):
        """
        Checks if a PDF is readable and extracts text.
        Args:
            pdf_path (str): Path to the PDF file.
        Returns:
            list: A list containing extracted text from each page, [].
        """
        extracted_text = []
        ocr = self.ppocr_obj
        try:
            pages = convert_from_path(self._filepath)
            # Use PaddleOCR for scanned images
            for page in pages:
                image = np.array(page)
                result = ocr.ocr(img=image, cls=True)[0]
                if result:
                    text = "\n".join([line[1][0] for line in result])
                    extracted_text.append(text)
        except Exception as e:
            print(f"Error opening PDF: {e}")
            return []

        return extracted_text

    def check_page_relevance(self, page_text, fields, thresh=0.15):
        page_text_list = page_text.lower().split()
        page_text_set = set(page_text_list)
        keywords_list = []
        for field in fields:
            keywords_list.extend(field.lower().split())

        keywords_set = set(keywords_list)
        num_matched_fields = len(page_text_set.intersection(keywords_set))
        total_fields = len(keywords_set)
        return num_matched_fields >= thresh * total_fields

    def get_relevant_pages(self, extracted_texts, document_type):
        relevant_pages = []
        fields = self.cm.get_fields_from_filetype(document_type)
        for i, text in enumerate(extracted_texts):
            if self.check_page_relevance(page_text=text, fields=fields):
                relevant_pages.append(i)
        if relevant_pages == []:
            raise NoRelevantPagesException(
                "Provided document contains no relevant pages"
            )
        return relevant_pages

    def count_pdf_pages(self):
        """
        This function counts the number of pages in a PDF file using fitz.

        Args:
            filepath (str): The path to the PDF file.

        Returns:
            int: The number of pages in the PDF file.

        Raises:
            Exception: If the file cannot be opened or is not a PDF file.
        """
        try:
            doc = fitz.open(self._filepath)
            # logger.debug(f"File:: {self._filepath} has {doc.page_count} pages")
            return doc.page_count
        except Exception as e:
            # logger.error("Error opening PDF file:: %s", self._filepath)
            raise NoRelevantPagesException(
                f"Error opening PDF file: {self._filepath}. Reason: {e}"
            )


class NoRelevantPagesException(Exception):
    """Raised when the provided document contains no relevant pages."""

    pass


class RAG:
    def __init__(
        self, filepaths: List[str], rerank: Optional[SentenceTransformerRerank] = None
    ) -> None:
        documents = SimpleDirectoryReader(input_files=filepaths).load_data()
        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize",
            use_async=True,
        )
        self.index = VectorStoreIndex.from_documents(
            documents=documents,
            response_synthesizer=response_synthesizer,
        )
        if not rerank:
            self.query_engine = self.index.as_query_engine(
                response_mode="tree_summarize",
                use_async=True,
                streaming=False,
                similarity_top_k=10,
            )
        else:
            self.query_engine = self.index.as_query_engine(
                response_mode="tree_summarize",
                use_async=True,
                streaming=False,
                similarity_top_k=10,
                node_postprocessors=[rerank],
            )

    def run_query_engine(self, prompt):
        response = self.query_engine.query(prompt)
        return response.response


class LLMEntityExtractor(RAG):
    def __init__(
        self,
        rerank: Optional[SentenceTransformerRerank] = None,
        filepath: Optional[str] = "",
        filetype: Optional[str] = "",
    ) -> None:
        self.rerank = rerank
        filetype = filetype.lower()
        self.document_path = filepath
        self.document_type = filetype

        # initialize the RAG Module
        super().__init__(filepaths=[self.document_path], rerank=self.rerank)

        # initialize prompt and entity configs
        self.prompt = ""
        self.cm = ConfigManager()

        # initialize utils
        self.utils = Utils()

        # set query from prompt template and fields
        self.fields = self.cm.get_fields_from_filetype(filetype=filetype)
        # self.query = self.prompt.format(fields=", ".join(self.fields))

        # initialize text extractor
        self.text_extractor = None

    def readable_extract(self):
        return self.run_query_engine(prompt=self.prompt)

    def non_readable_extract(self):
        if not self.text_extractor:
            self.text_extractor = TextExtractor(self.document_path)
        extracted_texts = self.text_extractor.extract_page_texts()
        if self.document_type == "mobile_invoice":
            # iterative entity extraction
            filepaths = []
            unstructured_extracted_textfile = self.utils.create_temp_txt_file(
                "\n\n".join(
                    [str(element) for element in self.text_extractor.pdf_elements]
                )
            )
            ppocr_extracted_textfile = self.utils.create_temp_txt_file(
                "\n\n".join([text for text in extracted_texts])
            )
            filepaths.append(unstructured_extracted_textfile)
            filepaths.append(ppocr_extracted_textfile)
        else:
            relevant_pages = self.text_extractor.get_relevant_pages(
                extracted_texts=extracted_texts, document_type=self.document_type
            )

            # iterative entity extraction
            filepaths = []
            for relevant_page in relevant_pages:
                temp_filename = self.utils.create_temp_txt_file(
                    extracted_texts[relevant_page]
                )
                filepaths.append(temp_filename)
        print(filepaths)
        super().__init__(filepaths=filepaths, rerank=self.rerank)
        print(self.prompt)
        llm_response = self.run_query_engine(self.prompt)

        return llm_response

    def extract(self):
        self.utils.clear_dir(dir=self.cm.OUTPUT_DIR)
        filename = os.path.basename(self.document_path)
        self.utils.download_and_save_file(
            source=self.document_path, save_dir=self.cm.OUTPUT_DIR
        )
        local_filepath = os.path.join(self.cm.OUTPUT_DIR, filename)
        if not self.text_extractor:
            self.text_extractor = TextExtractor(filepath=local_filepath)

        # Creating prompt based on document_type
        if self.document_type == "invoice":
            print("Getting invoice prompt")
            print("Extracting Tabular Data and address info from pdf")
            tabular_data = self.text_extractor.get_tabular_data_json_str()
            address_info = self.text_extractor.get_address_info()

            print("Passing address_info and tabular_data to prompt")
            self.prompt = invoice_prompt(data=tabular_data, address_info=address_info)
        elif self.document_type == "mobile_invoice":
            print("Getting mobile invoice prompt")
            self.prompt = mobile_invoice_prompt(fields=self.fields)
        else:
            print("Getting general prompt")
            self.prompt = general_prompt(fields=self.fields)

        if self.text_extractor.is_file_readable():
            if self.text_extractor.count_pdf_pages() < 15:
                print("File is readable")
                print("Extracting entities")
                llm_response = self.readable_extract()
            else:
                print("File has a lot of pages! Choosing not to process.")
                llm_response = ""
        else:
            # extract texts, get relevant texts and save to txt file and run rag again
            print("File is not readable, Using OCR Based Text Extraction")
            llm_response = self.non_readable_extract()

        return llm_response
