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
    quotation_prompt,
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
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import cv2
from jdeskew.estimator import get_angle
from jdeskew.utility import rotate
from unstructured.partition.pdf import partition_pdf


class TextExtractor:
    def __init__(self, filepath, use_trocr: Optional[bool] = False) -> None:
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
            use_space_char=True,
        )
        self.use_trocr = use_trocr
        self.trocr_model = None
        self.trocr_processor = None

        if use_trocr:
            self.trocr_processor = TrOCRProcessor.from_pretrained(
                "microsoft/trocr-large-printed"
            )
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained(
                "microsoft/trocr-large-printed"
            )
            self.trocr_model.config.eos_token_id = 2
        self.pdf_elements = partition_pdf(self._filepath, strategy="hi_res")

    def is_file_readable(self):
        total_words = 0
        try:
            doc = fitz.open(self._filepath)
        except Exception as e:
            print("Could not open file: Corrupt or Empty file", e)
            return None
        for page in doc:
            text = page.get_text()
            total_words += len(text.split())

        if total_words < 10:
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

    def get_texts_trocr(self, image, boxes):
        def model_inference(input_image):
            pixel_values = self.trocr_processor(
                input_image, return_tensors="pt"
            ).pixel_values
            generated_ids = self.trocr_model.generate(pixel_values)
            generated_text = self.trocr_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            return generated_text

        box = np.array(boxes).astype(np.int32).reshape(-1, 2)

        height = image.shape[0]
        width = image.shape[1]

        mask = np.zeros((height, width), dtype=np.uint8)
        new_list = boxes.copy()

        text = ""
        for boxs in new_list:
            box = np.array(boxs).astype(np.int32).reshape(-1, 2)
            points = np.array([box])
            cv2.fillPoly(mask, points, (255))
            res = cv2.bitwise_and(image, image, mask=mask)
            rect = cv2.boundingRect(points)  # returns (x,y,w,h) of the rect
            cropped = res[rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]]
            text += model_inference(input_image=cropped)
            text += "\n"

        return text

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
            pages = convert_from_path(self._filepath, dpi=400)
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

    def autocrop(self, image: np.array, thresh=0.05):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 15
        )
        for start in range(0, binary.shape[1], 2):
            row = binary[:, start : start + 2] < 127
            if np.sum(row) >= thresh * len(row[0]) * len(
                row
            ):  # > 5% of the pixels are black
                print("start: ", start)
                break
        for end in range(binary.shape[1] - 2, 0, -2):
            row = binary[:, end : end + 2] < 127
            if np.sum(row) >= thresh * len(row[0]) * len(
                row
            ):  # > 5% of the pixels are black
                print("end: ", end)
                break

        border = binary.shape[1] // 20
        start = max(0, start - border)
        end = min(binary.shape[1], end + border)
        if image is not None:
            return image[:, start:end]
        return binary[:, start:end]

    def extract_page_texts(self):
        """
        Checks if a PDF is readable and extracts text.
        Args:
            pdf_path (str): Path to the PDF file.
        Returns:
            list: A list containing extracted text from each page, [].
        """
        extracted_text = []
        try:
            pages = convert_from_path(self._filepath, dpi=400)
            # Use PaddleOCR for scanned images
            for page in pages:
                # crop page to only include the text and remove additional whitespaces
                image = np.array(page)
                cropped_image = self.autocrop(image=image)
                # remove skew
                angle = get_angle(cropped_image)
                cropped_image = rotate(cropped_image, angle)
                result = self.ppocr_obj.ocr(
                    img=cropped_image, rec=True, det=True, cls=False, bin=True
                )[0]
                boxes = [line[0] for line in result]
                paddleocr_text = "\n".join([line[1][0] for line in result])
                print(self.use_trocr)
                if self.use_trocr is True:
                    print("Using TrOCR")
                    trocr_text = self.get_texts_trocr(image=cropped_image, boxes=boxes)
                    extracted_text.append(trocr_text)
                else:
                    print("Using PaddleOCR")
                    extracted_text.append(paddleocr_text)
        except Exception as e:
            print(f"Error opening PDF: {e}")
            return []

        return extracted_text

    def check_page_relevance(self, page_text, fields, thresh=0.05):
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
                streaming=True,
                similarity_top_k=10,
            )
        else:
            self.query_engine = self.index.as_query_engine(
                response_mode="tree_summarize",
                use_async=True,
                streaming=True,
                similarity_top_k=10,
                node_postprocessors=[rerank],
            )

    def run_query_engine(self, prompt):
        response = self.query_engine.query(prompt)
        response.print_response_stream()
        return str(response)


class LLMEntityExtractor(RAG):
    def __init__(
        self,
        rerank: Optional[SentenceTransformerRerank] = None,
        filepath: Optional[str] = "",
        filetype: Optional[str] = "",
        use_trocr: Optional[bool] = False,
    ) -> None:
        self.rerank = rerank
        filetype = filetype.lower()
        self.document_path = filepath
        self.document_type = filetype
        self.use_trocr = use_trocr
        print(self.use_trocr, use_trocr)

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
            self.text_extractor = TextExtractor(
                self.document_path, use_trocr=self.use_trocr
            )
        extracted_texts = self.text_extractor.extract_page_texts()
        if self.document_type == "mobile_invoice":
            # iterative entity extraction
            filepaths = []
            # unstructured_extracted_textfile = self.utils.create_temp_txt_file(
            #     "\n\n".join(
            #         [str(element) for element in self.text_extractor.pdf_elements]
            #     )
            # )
            ppocr_extracted_textfile = self.utils.create_temp_txt_file(
                "\n\n".join([text for text in extracted_texts])
            )
            # filepaths.append(unstructured_extracted_textfile)
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
            self.text_extractor = TextExtractor(
                filepath=local_filepath, use_trocr=self.use_trocr
            )

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
        elif self.document_type == "quotation":
            print("Getting quotation prompt")
            self.prompt = quotation_prompt(fields=self.fields)
        else:
            print("Getting general prompt")
            self.prompt = general_prompt(fields=self.fields)

        if self.text_extractor.is_file_readable() is True:
            if self.text_extractor.count_pdf_pages() < 15:
                print("File is readable")
                print("Extracting entities")
                llm_response = self.readable_extract()
            else:
                print("File has a lot of pages! Choosing not to process.")
                llm_response = ""
        elif self.text_extractor.is_file_readable() is False:
            # extract texts, get relevant texts and save to txt file and run rag again
            print("File is not readable, Using OCR Based Text Extraction")
            llm_response = self.non_readable_extract()
        else:  # returns None
            print("Could Not Process File")
            llm_response = ""

        return llm_response
