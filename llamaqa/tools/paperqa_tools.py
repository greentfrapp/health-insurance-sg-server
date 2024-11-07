from typing import Any

from llama_index.core.tools.tool_spec.base import BaseToolSpec

from llama_index.core.callbacks import (
    CallbackManager,
    CBEventType,
    EventPayload,
    trace_method,
)
from llama_index.core.tools import BaseTool, ToolOutput, adapt_to_async_tool

from llamaqa.store.store import VectorStore
from llamaqa.utils.inner_context import InnerContext
from .gather_evidence import gather_evidence
from .generate_response import generate_response, retrieve_context
from .summarize_evidence import summarize_evidence


def tell_llm_about_failure_in_extract_reasoning_step(
    callback_manager: CallbackManager, _: ValueError
) -> ToolOutput:
    """
    If the developer has instructed to tell the Agent a complaint about its non-cooperation,
    we will emit a Tool Output that we prepared (at initialization time) to the LLM, so that
    the LLM can be more cooperative in its next generation.
    """
    print(_)
    message = """Error: Could not parse output. Please follow the thought-action-input format. Try again.
Maybe you should try calling the gather_evidence tool.
Remember that the format should be
```
Thought: I need to use a tool to help me answer the question.
Action: tool name if using a tool.
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
```
"""
    dummy_tool_output = ToolOutput(
        content=message,
        tool_name="unknown",
        raw_input={},
        raw_output=message,
    )
    with callback_manager.event(
        CBEventType.FUNCTION_CALL,
        payload={
            EventPayload.FUNCTION_CALL: "unknown",
        },
    ) as event:
        event.on_end(payload={EventPayload.FUNCTION_OUTPUT: str(dummy_tool_output)})

    return dummy_tool_output


class PaperQAToolSpec(BaseToolSpec):
    spec_functions = [
        "gather_evidence",
        # "summarize_evidence",
        # "generate_response",
        "retrieve_evidence",
        # "retrieve_memory",
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
        response = await gather_evidence(
            self.context,
            self.store,
            query,
            embedding_model=self.embedding_model,
        )
        await summarize_evidence(
            self.context,
            self.store,
            query,
            chunks=self.context.chunks[-5:],
            summary_llm_model=self.summary_llm_model,
        )
        return response

    async def summarize_evidence(self, question: str) -> str:
        """
        Create summaries relevant to a question using evidence
        found in an earlier step.

        Args:
            question (str): the question to be answered

        """
        return await summarize_evidence(
            self.context,
            self.store,
            question,
            summary_llm_model=self.summary_llm_model,
        )

    async def generate_response(self, question: str) -> str:
        """
        Generate a response to a question after finding evidence
        and generating summaries from earlier steps.
        The output from this tool might indicate that there is
        insufficient information. In that case, gather_evidence should
        be used again to collect more information.
        Otherwise, the output from this tool should be returned exactly
        to the user without changes.

        Args:
            question (str): the question to be answered

        """
        return await generate_response(
            self.context,
            self.store,
            question,
            llm_model=self.summary_llm_model,
        )

    async def retrieve_evidence(self, question: str) -> str:
        """
        Retrieves the evidence and summaries from earlier steps and
        combine them with the user's question to form an instruction
        to generate the final response.

        Args:
            question (str): the question to be answered

        """
        return retrieve_context(
            self.context,
            self.store,
            question,
        )

    async def retrieve_memory(self) -> str:
        """
        Use this tool to retrieve information about evidences and summaries
        that were previously collected.
        """
        return "Found 10 pieces of evidence"
