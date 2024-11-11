import asyncio

from llama_index.core.tools.tool_spec.base import BaseToolSpec

from ..llms.embedding_model import EmbeddingModel
from ..llms.llm_model import LLMModel
from ..store.store import VectorStore
from ..utils.cache import Cache
from .gather_evidence import gather_evidence
from .retrieve_evidence import retrieve_evidence
from .summarize_evidence import summarize_evidence
from .utils import output_descriptor


class PaperQAToolSpec(BaseToolSpec):
    spec_functions = [
        "gather_evidence",
        "retrieve_evidence",
    ]
    function_output_descriptors = {}
    store: VectorStore
    cache = Cache()
    embedding_model: EmbeddingModel
    summary_llm_model: LLMModel
    current_task_id: str = ""

    def __init__(
        self,
        store: VectorStore,
        cache: Cache,
        embedding_model: EmbeddingModel,
        summary_llm_model: LLMModel,
    ):
        super().__init__()
        self.store = store
        self.cache = cache
        self.embedding_model = embedding_model
        self.summary_llm_model = summary_llm_model

    @output_descriptor("Gathering evidence on \"{query}\"...")
    def gather_evidence(self, query: str) -> str:
        """
        Find and return pieces of evidence that are relevant
        to a given query.
        This can be called multiple times with varying search terms
        if insufficient information was found.
        This should only be called for queries relevant to insurance.
        Never ever call this tool for queries unrelated to insurance.

        Args:
            query (str): the query to be used

        """
        async def foo():
            response = await gather_evidence(
                self.cache,
                self.store,
                query,
                embedding_model=self.embedding_model,
                summary_llm_model=self.summary_llm_model,
                prefix=self.current_task_id,
            )
            return response
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(foo())

    @output_descriptor("Retrieving gathered evidence...")
    def retrieve_evidence(self, question: str) -> str:
        """
        Retrieves the evidence and summaries from earlier steps and
        combine them with the user's question to form an instruction
        to generate the final response.

        Args:
            question (str): the question to be answered

        """
        return retrieve_evidence(
            self.cache,
            self.store,
            question,
        )
