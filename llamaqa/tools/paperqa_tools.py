from typing import Optional
import asyncio

from llama_index.core.tools.tool_spec.base import BaseToolSpec

from .gather_evidence import gather_evidence
from .retrieve_evidence import retrieve_evidence
from .utils import tool_metadata
from ..llms.embedding_model import EmbeddingModel
from ..llms.llm_model import LLMModel
from ..store.store import VectorStore
from ..utils.cache import Cache
from ..utils.logger import CostLogger
from ..utils.policies import VALID_POLICIES


class PaperQAToolSpec(BaseToolSpec):
    spec_functions = [
        "gather_evidence_by_query",
        "gather_policy_overview",
        "retrieve_evidence",
    ]
    function_output_descriptors = {}
    store: VectorStore
    cache = Cache()
    embedding_model: EmbeddingModel
    summary_llm_model: LLMModel
    current_task_id: str = ""
    cost_logger: CostLogger

    def __init__(
        self,
        store: VectorStore,
        cache: Cache,
        embedding_model: EmbeddingModel,
        summary_llm_model: LLMModel,
        cost_logger: Optional[CostLogger] = None,
    ):
        super().__init__()
        self.store = store
        self.cache = cache
        self.embedding_model = embedding_model
        self.summary_llm_model = summary_llm_model
        self.cost_logger = cost_logger or CostLogger()

    @tool_metadata(
        desc=f"""
Find and return pieces of evidence that are relevant
to a given query and policy.
This can be called multiple times with varying search terms
if insufficient information was found.
This should only be called for queries relevant to insurance.
Never ever call this tool for queries unrelated to insurance.

valid_policies = {VALID_POLICIES}

Args:
    query (str): The query to search for
    policy (str) = None: The policy to filter by, must be one of values in valid_policies list. If None, defaults to searching all policies.
""",
        output_desc='Retrieving information with query "{query}" on policy "{policy}"...',
        default_kwargs={"policy": None},
    )
    def gather_evidence_by_query(
        self,
        query: str,
        policy: Optional[str] = None,
    ) -> str:
        async def gather_evidence_helper():
            response = await gather_evidence(
                self.cache,
                self.store,
                query=query,
                policy=policy,
                embedding_model=self.embedding_model,
                summary_llm_model=self.summary_llm_model,
                prefix=self.current_task_id,
            )
            return response

        loop = asyncio.get_event_loop()
        return loop.run_until_complete(gather_evidence_helper())

    @tool_metadata(
        desc=f"""
Find all information about a policy.
This is useful when you want to get key features about a policy,
summarize a policy, or broadly compare different policies.

valid_policies = {VALID_POLICIES}

Args:
    policy (str) = None: The policy to filter by, must be one of values in valid_policies list. If None, defaults to searching all policies.
""",
        output_desc='Retrieving all information about policy "{policy}"...',
    )
    def gather_policy_overview(
        self,
        policy: str,
    ) -> str:
        async def gather_evidence_helper():
            response = await gather_evidence(
                self.cache,
                self.store,
                query=None,  # Use query=None to return all information
                policy=policy,
                embedding_model=self.embedding_model,
                summary_llm_model=self.summary_llm_model,
                prefix=self.current_task_id,
            )
            return response

        loop = asyncio.get_event_loop()
        return loop.run_until_complete(gather_evidence_helper())

    @tool_metadata(
        output_desc="Retrieving gathered evidence...",
    )
    def retrieve_evidence(self, question: str) -> str:
        """
        Retrieves the evidence and summaries from earlier steps and
        combine them with the user's question to form an instruction
        to generate the final response.

        This should be called after all evidence has been gathered.

        Args:
            question (str): the question to be answered

        """
        return retrieve_evidence(
            self.cache,
            self.store,
            question,
        )
