import os
from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from llama_index import set_global_service_context

from src.llms import LLMUtils
from src.config import ConfigManager
from src.schema import InputSchema, OutputSchema
from src.utils import Utils
from src.api import ExtractionAPI
from src.logger import CustomLogger, MongoLogWriter
from anyio.lowlevel import RunVar
from anyio import CapacityLimiter


config_manager = ConfigManager()
logger_instance = CustomLogger(__name__)
logger = logger_instance.logger
mongo_logger = MongoLogWriter(
    uri=config_manager.MONGO_URI,
    database_name=config_manager.MONGO_DB_NAME,
    collection_name="dp_logs",
)

# Get allowed origins from environment variable
ALLOWED_ORIGINS_STR = os.environ.get("ALLOWED_ORIGINS")
if ALLOWED_ORIGINS_STR:
    ALLOWED_ORIGINS = ALLOWED_ORIGINS_STR.split(",")
else:
    ALLOWED_ORIGINS = []

# Get other environment variables
BASE_PATH = config_manager.BASE_PATH
VERSION = "1.0.0"

app = FastAPI(
    title="Document Processing",
    summary="Documents processing and data extraction.",
    version=VERSION,
    root_path=f"{BASE_PATH}",
)
logger.debug(BASE_PATH)
# Allow CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.debug(f"Model path: {config_manager.LLM_MODEL_PATH}")
logger.debug(f"Data Dir path: {config_manager.DATA_DIR}")
llm_module = None
utils = Utils()

extraction_api = None

MODELS_DIR = config_manager.MODELS_DIR
RunVar("_default_thread_limiter").set(CapacityLimiter(1))


@app.get("/api/start")
def start_service():
    global llm_module, extraction_api
    mongo_logger.push_log(
        level="INFO",
        name=str(__name__),
        message="Starting Document Processing Service...",
        process_id="",
    )
    logger.info("Starting Service...")
    if llm_module:
        return HTMLResponse(content="Models Active!", status_code=status.HTTP_200_OK)
    llm_module = LLMUtils(init_reranker=False)
    logger.info("Setting global service context with LLM and Embedding Model")
    set_global_service_context(service_context=llm_module.service_context)
    extraction_api = ExtractionAPI(rerank=None)
    return HTMLResponse(content="Models Loaded!", status_code=status.HTTP_200_OK)


@app.get(
    "/api/version",
    summary="Outputs the version of the API server",
    response_model=str,
    response_class=HTMLResponse,
    responses={
        200: {
            "description": "Version output successful",
            "content": {"text/html": {"example": VERSION}},
        }
    },
)
def version() -> HTMLResponse:
    """
    Version endpoint for the Document Processing API.
    """
    # TODO: Update the version using SemVer
    return HTMLResponse(content=VERSION, status_code=status.HTTP_200_OK)


# Health Check API
@app.get("/api/health", summary="Verify the health of API server")
def health_check():
    """
    Health check endpoint for the Document Processor API.
    """
    try:
        return HTMLResponse(content="OK", status_code=status.HTTP_200_OK)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@app.get("/api/get_config")
def get_config():
    return HTMLResponse(
        content=utils.json_to_str(data=str(vars(config_manager)), indent=4),
        status_code=status.HTTP_200_OK,
    )


# Processed Data (Entity Extraction) API
@app.post("/api/get_entities", response_model=OutputSchema)
async def get_entities_api(data: InputSchema, background_tasks: BackgroundTasks):
    if not llm_module:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str("Model Not Initialized"),
        )
    try:
        extraction_api.configure(
            document_type=data.document_type,
            struct_type=data.struct_type,
            document_path=data.document_path,
        )
        logger = logger_instance.configure_logger()
        document_name = os.path.basename(data.document_path)
        logger.info(
            f"Extraction Process for {document_name} started in the backend with process_id: {extraction_api.process_id}"
        )
        mongo_logger.push_log(
            level="INFO",
            name=str(__name__),
            message=f"Extraction Process for {document_name} started in the backend with process_id: {extraction_api.process_id}",
            process_id=extraction_api.process_id,
        )
        background_tasks.add_task(
            extraction_api.get_entities,
        )
        return OutputSchema(
            response="success",
            message=f"Started extraction process in the backend for file: {data.document_path} with process id: {extraction_api.process_id}",
            process_id=extraction_api.process_id,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


# Processed Data (Entity Extraction) API
@app.post("/api/get_entities_from_dir", response_model=OutputSchema)
async def get_entities_from_dir_api(
    data: InputSchema, background_tasks: BackgroundTasks
):
    if not llm_module:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str("Model Not Initialized"),
        )
    try:
        extraction_api.configure(
            document_type=data.document_type,
            struct_type=data.struct_type,
            document_dir=data.document_dir,
        )
        logger = logger_instance.configure_logger()
        logger.info(
            f"Extraction Process started for {data.document_dir} in the backend with process_id: {extraction_api.process_id}"
        )
        mongo_logger.push_log(
            level="INFO",
            name=str(__name__),
            message=f"Extraction Process started for {data.document_dir} in the backend with process_id: {extraction_api.process_id}",
            process_id=extraction_api.process_id,
        )
        background_tasks.add_task(extraction_api.get_entities_from_dir)
        return OutputSchema(
            response="success",
            message=f"Started extraction process in the backend for files in dir: {data.document_dir}, with process id: {extraction_api.process_id}",
            process_id=extraction_api.process_id,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
