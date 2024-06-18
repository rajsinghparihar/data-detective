import pypdfium2 as pdfium
import os
import cv2
from paddleocr import PaddleOCR
from src.config import ConfigManager
from src.logger import CustomLogger, MongoLogWriter


class OCREntityExtractor:
    def __init__(
        self, document_path: str, document_type: str, struct_type: str, process_id: str
    ) -> None:
        self.document_path = document_path
        self.document_type = document_type
        self.struct_type = struct_type
        self.logger = CustomLogger(__name__).configure_logger()
        self.cm = ConfigManager()
        self.mongo_logger = MongoLogWriter(
            uri=self.cm.MONGO_URI,
            database_name=self.cm.MONGO_DB_NAME,
            collection_name="dp_logs",
        )
        log_msg = f"Initializing Class {__name__}.{self.__class__.__qualname__}"
        self.logger.debug(log_msg)
        self.mongo_logger.push_log(
            level="DEBUG",
            name=str(__name__),
            message=log_msg,
            process_id=process_id,
        )
        self.structure_config = self.cm.structure_config
        self.ppocr_instance = PaddleOCR(
            use_angle_cls=False,
            det=False,
            cls=False,
            lang="en",
            use_gpu=True,
            verbose=False,
            det_model_dir=os.path.join(self.cm.MODELS_DIR, "ocr/det/"),
            rec_model_dir=os.path.join(self.cm.MODELS_DIR, "ocr/rec/"),
            cls_model_dir=os.path.join(self.cm.MODELS_DIR, "ocr/cls/"),
        )

    def pdf_to_image(self):
        pg = 0
        pdf = pdfium.PdfDocument(self.document_path)
        n_pages = len(pdf)
        img_paths = []
        for page_number in range(n_pages):
            page = pdf.get_page(page_number)
            pil_image = page.render(scale=2, rotation=0, crop=(0, 0, 0, 0))
            image_path = f"{self.cm.INTER_DIR}/{os.path.basename(self.document_path).strip('.pdf')}_image_{pg+1}.png"
            pil_image.to_pil().save(image_path)
            img_paths.append(image_path)
            pg += 1
        return img_paths

    def crop_image(self, image, center_x, center_y, width, height):
        """Crops an image given center values and width and height.

        Args:
        image: The image to crop.
        center_x: The x-coordinate of the center of the crop.
        center_y: The y-coordinate of the center of the crop.
        width: The width of the crop.
        height: The height of the crop.

        Returns:
        The cropped image.
        """

        # Get the dimensions of the image.
        image_width, image_height = image.shape[1], image.shape[0]
        center_x = image_width * center_x
        center_y = image_height * center_y
        width = image_width * width
        height = image_height * height
        # Calculate the start and end coordinates of the crop.
        start_x = int(center_x - width / 2)
        start_y = int(center_y - height / 2)
        end_x = int(center_x + width / 2)
        end_y = int(center_y + height / 2)

        # Crop the image.
        cropped_image = image[start_y:end_y, start_x:end_x]

        return cropped_image

    def extract(self) -> tuple[bool, dict]:
        """Extracts data from a PDF document based on structure configuration.

        Returns:
            A tuple containing a flag indicating success (bool) and extracted data (dict).
        """

        # Check for invalid conditions: missing structure or non-PDF document
        if not self._validate_document():
            return False, {}

        # Extract images from the PDF
        img_paths = self.pdf_to_image()

        # Process images based on document type
        data = self._process_images(img_paths)

        # Return results
        return True, data if data else {}

    def _validate_document(self):
        """Checks if the document is valid for processing."""
        return self.structure_config.keys().__contains__(
            self.struct_type
        ) and self.document_path.endswith(".pdf")

    def _process_images(self, img_paths):
        """Processes images based on document type and structure configuration.

        Args:
            img_paths: A list of paths to extracted images.

        Returns:
            A dictionary containing extracted data or None if no data was found.
        """
        if self.document_type == "invoice":
            for img_path in img_paths:
                image = cv2.imread(img_path)
                label = {}
                for box in self.structure_config[self.struct_type]:
                    label = self._process_image_box(image, box, label)
                break  # Only process one invoice image
            return label if label else None
        else:
            # Handle other document types here (if needed)
            return None

    def _process_image_box(self, image, box, label):
        """Processes a single image box and extracts data."""
        img = self.crop_image(
            image, box["centerX"], box["centerY"], box["width"], box["height"]
        )
        result = self.ppocr_instance.ocr(img, det=False, cls=True)
        for idx in range(len(result)):
            text = result[idx][0][0]
            try:
                num = float(text)
                if num < 0:
                    num *= -1
                text = str(num)
            except Exception:
                pass

        key = box["labels"][0]
        if key in label.keys():
            label[box["labels"][0] + "_2"] = text
        else:
            label[box["labels"][0]] = text

        return label
