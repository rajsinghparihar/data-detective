from llama_index import ServiceContext
from llama_index.llms import LlamaCPP, OpenAI
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.postprocessor import SentenceTransformerRerank
from llama_index.llms.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from src.config import ConfigManager
from typing import Optional
from src.logger import CustomLogger


class LLMUtils:
    def __init__(self, init_reranker: Optional[bool] = True) -> None:
        logger = CustomLogger(__name__).logger
        logger.debug("Initializing LLMUtils")
        self.cm = ConfigManager()
        self.model_path = self.cm.LLM_MODEL_PATH

        logger.debug("Loading LLM, Model Name: %s", self.cm.LLM_MODEL_NAME)
        self._llm = self.load_llm(llama=True)

        logger.debug(
            "Loading Embedding Model, Model Name: %s", self.cm.EMBEDDING_MODEL_NAME
        )
        self._embed_model = HuggingFaceEmbedding(
            model_name=self.cm.EMBEDDING_MODEL_PATH, trust_remote_code=True
        )
        self.service_context = ServiceContext.from_defaults(
            llm=self._llm,
            embed_model=self._embed_model,
        )
        if init_reranker:
            logger.debug(
                "Loading Reranking Model, Model Name: %s", self.cm.RERANKING_MODEL_NAME
            )
            self._rerank = SentenceTransformerRerank(
                model=self.cm.RERANKING_MODEL_PATH, top_n=3
            )

    def load_llm(
        self,
        llama: bool = True,
        api_base: str = "http://0.0.0.0:8080/v1",
        temperature=0,
        max_tokens=1024,
        context_window=8192,
        n_gpu_layers=-1,
    ):
        if llama:
            return LlamaCPP(
                model_path=self.model_path,
                temperature=temperature,
                max_new_tokens=max_tokens,
                context_window=context_window,
                generate_kwargs={},
                model_kwargs={"n_gpu_layers": n_gpu_layers},
                messages_to_prompt=messages_to_prompt,
                completion_to_prompt=completion_to_prompt,
                verbose=True,
            )
        else:
            return OpenAI(
                model="gpt-4",
                temperature=temperature,
                max_tokens=max_tokens,
                messages_to_prompt=messages_to_prompt,
                completion_to_prompt=completion_to_prompt,
                api_base=api_base,
                api_key="sk-abcdef",
            )
