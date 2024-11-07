from functools import partial
from typing import Any, Coroutine, List
import asyncio

from llamaqa.store.store import VectorStore
from llamaqa.utils.inner_context import InnerContext
from llamaqa.utils.text import TextPlus
from llamaqa.utils.utils import (
    llm_parse_json,
    map_fxn_summary,
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


async def gather_with_concurrency(n: int, coros: list[Coroutine]) -> list[Any]:
    # https://stackoverflow.com/a/61478547/2392535
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(sem_coro(c) for c in coros))



async def summarize_evidence(
    context: InnerContext,
    store: VectorStore,
    query: str,
    chunks: List[TextPlus] = [],
    summary_llm_model = None,
):
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
            for m in (chunks if len(chunks) else context.chunks)
        ],
    )
    context.summaries = [summary for summary, _ in results]
    return f"Generated {len(results)} summaries"
