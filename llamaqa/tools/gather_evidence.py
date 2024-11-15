from functools import partial
from typing import Optional, cast

from .utils import map_fxn_summary
from ..store.supabase_store import SupabaseStore
from ..utils.cache import Cache
from ..utils.context import Context
from ..utils.utils import (
    gather_with_concurrency,
    llm_parse_json,
)


SUMMARY_JSON_SYSTEM_PROMPT = """\
Provide a summary of the relevant information that could help answer the question based on the excerpt. Respond with the following JSON format:

{{
  "summary": "...",
  "relevance_score": "...",
  "points": [
    {{
        "quote": "...",
        "point": "..."
    }}
  ]
}}

where `summary` is relevant information from text - {summary_length} words, `relevance_score` is the relevance of `summary` to answer question (out of 10), and `points` is an array of `point` and `quote` pairs that supports the summary where each `quote` is an exact match quote (max 50 words) from the text that best supports the respective `point`. Make sure that the quote is an exact match without truncation or changes. Do not truncate the quote with any ellipsis.
"""  # noqa: E501


SUMMARY_JSON_PROMPT = (
    "Excerpt from {citation}\n\n----\n\n{text}\n\n----\n\nQuestion: {question}\n\n"
)


class EmptyDocsError(RuntimeError):
    """Error to throw when we needed docs to be present."""


async def gather_evidence(
    cache: Cache,
    store: SupabaseStore,
    query: Optional[str] = None,
    policy: Optional[str] = None,
    k: int = 5,
    mmr_lambda: float = 0.9,
    embedding_model = None,
    summary_llm_model = None,
    prefix: str = "",
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

    if query is None and policy is None:
        raise ValueError("At least one of query or policy must have a non-None value")

    # If query is None, retrieve all info about given policy
    if query is None:
        matches = await store.get_all_policy_info([policy])
        summaries = [Context(
            context=match.summary,
            text=match,
            score=5,
            points=match.points,
        ) for match in matches]
    # Use vector retrieval
    else:
        matches = (
            await store.max_marginal_relevance_search(
                query, k=k, fetch_k=2 * k, embedding_model=embedding_model,
                policies=[policy],
            )
        )[0]

        prompt_runner = partial(
            summary_llm_model.run_prompt,
            SUMMARY_JSON_PROMPT,
            system_prompt=SUMMARY_JSON_SYSTEM_PROMPT,
        )

        results = await gather_with_concurrency(
            n=4,
            coros=[
                map_fxn_summary(
                    text=m,
                    question=query,
                    prompt_runner=prompt_runner,
                    extra_prompt_data={
                        "summary_length": "about 100 words",
                        "citation": f"{m.name}: {m.doc.citation}",
                    },
                    parser=llm_parse_json,
                )
                for m in matches
            ],
        )
        summaries = [cast(Context, summary) for summary, _ in results]

    for summary in summaries:
        summary.text.name = prefix + summary.text.name
        summary.text.doc.docname = prefix + summary.text.doc.docname

    cache.summaries += summaries

    return f"Found {len(matches)} pieces of evidence. Call retrieve_evidence to view the evidence."
