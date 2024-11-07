from typing import Any

from llama_index.core.tools.tool_spec.base import BaseToolSpec

from llamaqa.store.store import VectorStore
from llamaqa.utils.inner_context import InnerContext


class DummyToolSpec(BaseToolSpec):
    spec_functions = [
        "gather_evidence",
        # "summarize_evidence",
        "generate_response",
        "retrieve_memory",
    ]
    store: VectorStore
    context = InnerContext()
    embedding_model: Any
    summary_llm_model: Any
    state: int = 0

    def __init__(self, store: VectorStore, context: InnerContext, embedding_model: Any, summary_llm_model: Any):
        self.store = store
        self.context = context
        self.embedding_model = embedding_model
        self.summary_llm_model = summary_llm_model

    async def gather_evidence(self, query: str) -> str:
        """
        Find and return pieces of evidence that are relevant
        to a given query.
        This can be called multiple times with varying search terms
        if insufficient information was found.

        Args:
            query (str): the query to be used

        """
        return "Found 10 pieces of evidence"

    async def summarize_evidence(self, question: str) -> str:
        """
        Create summaries relevant to a question using evidence
        found in an earlier step.

        Args:
            question (str): the question to be answered

        """
        return "Summarized 10 pieces of evidence"

    async def generate_response(self, question: str) -> str:
        """
        Generate a response to a question after finding evidence
        and generating summaries from earlier steps.
        The output from this tool should be returned exactly
        to the user without changes.

        Args:
            question (str): the question to be answered

        """
        if self.state == 0:
            self.state += 1
            return "The information provided only accounts for NTUC Income, no data is provided about AIA."
        return """AIA provides higher coverage than NTUC Income"""

    async def retrieve_memory(self) -> str:
        """
        Use this tool to retrieve information about evidences and summaries
        that were previously collected.
        """
        return "Found 10 pieces of evidence"
