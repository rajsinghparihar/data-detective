import outlines.prompts
from src.config import ConfigManager
import outlines


class PromptManager:
    def __init__(self) -> None:
        self.cm = ConfigManager()
        self.prompts = self.cm.get_prompts_config()

    def get_prompt(self, document_type) -> str:
        document_type = "".join([keyword.lower() for keyword in document_type.split()])
        return self.prompts.get(
            f"{document_type}_prompt", self.prompts["general_prompt"]
        )


@outlines.prompt
def general_prompt(fields):
    """
    You are an entity extractor.
    Using the information in the provided documents,
    What are the values of the following, {{ fields }}?
    Please format the response in JSON format, with these descriptors as keys and their values as values.
    In the records, make sure to only include the values of the descriptors without any descriptor names.
    Avoid keywords like <<SYS>> or [SYS] or [INST] in the final response.
    """


@outlines.prompt
def invoice_prompt(data: str, address_info: str):
    """
    You are an entity extractor. Using the information in the provided document,
    Complete the given tasks:
    1. Answer the following question,
    What is the Full Document Number (Invoice Number), the Invoice Description or Details, the Jio/Reliance Entity Name, and the Invoice Date?

    2. Given this additional INVOICE_AMOUNT_DATA:
    {{ data }}
    What is the the Invoice Total Amount (or Invoice Total) and the Amount Currency? Do not include all the values ONLY include the Invoice Total value.

    3. Given this ADDRESS_INFO:
    {{ address_info }}
    What is the Vendor Name and the Vendor Country in this ADDRESS_INFO?

    4. Based on the result of the previous task, please combine the outputs of 1., 2. and 3. and format the final output in JSON format and follow the following Instructions.

    In the output, make sure to include ONLY the values of the given 8 descriptors.
    Make sure to get the CORRECT currency and the CORRECT corresponding total invoice amount details from the given INVOICE_AMOUNT_DATA (which is passed in the prompt as a list of json objects). Also make sure to get the correct vendor name and country from the ADDRESS_INFO.
    The final response should contain valid JSON, AVOID keywords like <<SYS>> or [SYS] or [INST] in the final response.

    Here is an example of the expected output:
    {
        'Invoice Number': 'KWTMT/INDJ8/00062',
        'Invoice Description': 'GSM International Roaming Charges for using the net work of ZAIN,KUWAIT',
        'Reliance Entity Name': 'RELIANCE JIO INFOCOMM - JK',
        'Vendor Name': 'ZAIN - KUWAIT',
        'Vendor Country': 'DUBAI - UAE',
        'Invoice Date': '02-03-2024',
        'Invoice Amount': '5.58',
        'Currency': 'SDR'
    }
    """


@outlines.prompt
def mobile_invoice_prompt(fields):
    """
    You are an entity extractor. Using the information in the provided document,
    Complete the given tasks:

    1. What are the values of the following, {{ fields }}? Answer for each of the fields one by one step by step.

    2. Based on the result of the previous task, please combine the outputs and format the final output in JSON format with the following Instructions.

    In the output, make sure to include ONLY the values of the given 8 descriptors.
    The final response should contain the valid JSON response, Each value in the JSON response should be a single value. AVOID keywords like <<SYS>> or [SYS] or [INST] in the final response.

    Here is an example of the expected output:
    {
        'Customer Name': '<customer-name>',
        'Seller Name': '<seller-name>',
        'Seller GSTN': '<seller-GST-number>',
        'Invoice Number': '<invoice-number>',
        'Bill Date': '<bill-date>',
        'Amount': '<amount>',
        'Item Name': '<item-name>',
        'IMEI Number': '<IMEI-number>'
    }
    """
