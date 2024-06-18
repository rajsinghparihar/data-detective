from pydantic import BaseModel
from typing import Optional


class InputSchema(BaseModel):
    """Defines the expected format for input data.

    This schema is used to validate the data received by the application
    for processing documents.

    Args:
        document_path (Optional[str]): Path to the document file. Defaults to "".
        document_type (str): Required field specifying the type of document.
        document_dir (Optional[str]): Optional directory containing the document. Defaults to "".
        struct_type (Optional[str]): Optional field specifying the structure type of the document. Defaults to "".
    """

    document_path: Optional[str] = ""
    document_type: str
    document_dir: Optional[str] = ""
    struct_type: Optional[str] = ""


class OutputSchema(BaseModel):
    """Defines the format of the response data.

    This schema specifies the structure of the data returned by the application
    after processing a document.

    Args:
        response (str): Main response message containing the processing result.
        message (str):  Message providing additional information.
        process_id (Optional[str]): Optional identifier for the processing task. Defaults to "".
    """

    response: str
    message: str
    process_id: Optional[str] = ""
