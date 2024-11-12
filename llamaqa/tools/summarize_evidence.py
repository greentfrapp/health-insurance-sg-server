from functools import partial
from typing import List

from ..reader.doc import Text
from ..store.store import VectorStore
from ..utils.cache import Cache
from ..utils.utils import (
    gather_with_concurrency,
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

where `summary` is relevant information from text - {summary_length}, `relevance_score` is the relevance of `summary` to answer question (out of 10), and `points` is an array of `point` and `quote` pairs that supports the summary where each `quote` is an exact match quote (max 50 words) from the text that best supports the respective `point`. Make sure that the quote is an exact match without truncation or changes. Do not truncate the quote with any ellipsis.
"""  # noqa: E501


SUMMARY_JSON_PROMPT = (
    "Excerpt from {citation}\n\n----\n\n{text}\n\n----\n\nQuestion: {question}\n\n"
)


async def summarize_evidence(
    cache: Cache,
    store: VectorStore,
    query: str,
    chunks: List[Text] = [],
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
            for m in (chunks if len(chunks) else cache.chunks)
        ],
    )
    cache.summaries = [summary for summary, _ in results]
    return f"Generated {len(results)} summaries"
