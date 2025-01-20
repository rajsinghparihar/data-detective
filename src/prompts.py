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
    Using the information in the provided documents, use your deep understanding of documents and complete the following tasks.
    1. Answer the question, What are the values of the following, {{ fields }}?
    2. Print the answers against each field in a step by step approach.
    3. After you have all the answers ready, Please format the response in JSON format, with these fields as keys and their answers as values.

    Make sure to follow the Instructions below.
    1. In the records, make sure to include the values of the descriptors without any descriptor names.
    2. Avoid keywords like <<SYS>> or [SYS] or [INST] in the final response.
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

    Some Additional Info to help get the information more accurately:
    1. Seller Name is usually found in the start of the document, The seller could be a company or an individual.
    2. Item Name is the name of the product sold. It could be a smartphone so check for strings that look like smartphone model names.
    3. IMEI Number should be a 15 Digit Number.
    4. The Seller GSTN should be a 15 Character Alpha-Numeric String.

    Final Instructions:
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


@outlines.prompt
def quotation_prompt(fields):
    """
    You are an entity extractor. Using the information in the provided document,
    Complete the given tasks:

    1. What are the values of the following, {{ fields }}? Answer for each of the fields one by one step by step.

    2. Based on the result of the previous task, please combine the outputs and format the final output in JSON format with the following Instructions.

    Some Additional Info to help get the information more accurately:
    1. offer_reference_number is a.k.a. quotation number or enquiry number, it is the identifier for the quotation.
    2. payment_terms is a.k.a. terms of delivery or in general it is the agreed terms of delivery for the quotation.
    3. delivery_terms means the location of inspection or if it's not mentioned then leave it.
    4. The gst_number should be a 15 Character Alpha-Numeric String.
    5. If any field is unavailable, the value in the json output must be "Not Found"

    Final Instructions:
    In the output, make sure to include ONLY the values of the given 8 descriptors.
    The final response should contain the valid JSON response, Each value in the JSON response should be a single value. AVOID keywords like <<SYS>> or [SYS] or [INST] in the final response.

    Here is an example of the expected output:
    {
        'offer_reference_number': '<offer-reference-number>',
        'payment_terms': '<payment-terms>',
        'delivery_terms': '<delivery-terms>',
        'vendor_delivery_date': '<vendor-delivery-date>',
        'offer_validity_period': '<offer-validity-period>',
        'currency': '<currency>',
        'gst_number': '<gst-number>',
        'total_amount': '<total-amount>'
    }
    """
