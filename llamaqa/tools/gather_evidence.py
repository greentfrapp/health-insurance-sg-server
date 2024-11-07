from functools import partial
from typing import Any, cast

from llamaqa.store.store import VectorStore
from llamaqa.utils.inner_context import InnerContext


class EmptyDocsError(RuntimeError):
    """Error to throw when we needed docs to be present."""


async def gather_evidence(
    context: InnerContext,
    store: VectorStore,
    query: str,
    k: int = 5,
    mmr_lambda: float = 0.9,
    embedding_model = None,
):
    """
    Gather evidence from previous papers given a specific question to increase evidence and relevant paper counts.

    A valuable time to invoke this tool is right after another tool increases paper count.
    Feel free to invoke this tool in parallel with other tools, but do not call this tool in parallel with itself.
    Only invoke this tool when the paper count is above zero, or this tool will be useless.

    Args:
        query: Specific query to gather evidence for.
        state: Current state.

    Returns:
        String describing gathered evidence and the current status.
    """
    
    store.mmr_lambda = mmr_lambda
    matches = (
        await store.max_marginal_relevance_search(
            query, k=k, fetch_k=2 * k, embedding_model=embedding_model
        )
    )[0]

    context.chunks += matches

    return f"Found {len(matches)} pieces of evidence. Call retrieve_evidence to view the evidence."
