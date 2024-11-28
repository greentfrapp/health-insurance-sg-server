from typing import List, Optional

from ..reader.doc import Point
from .context import Context

CONTEXT_OUTER_PROMPT = "{context_str}\n\nValid Keys: {valid_keys}"
CONTEXT_INNER_PROMPT = "{name}: {text}\nFrom {citation}"


CONTEXT_INNER_PROMPT_WITH_QUOTE = "{name}:\n{text}{quotes}\nFrom {citation}"


class Cache:
    summaries: List[Context]
    max_sources: int = 10

    def __init__(self):
        self.summaries = []

    def filtered_contexts(self, max_sources: Optional[int] = None):
        # remove any contexts with a score of 0
        filtered_contexts = sorted(
            self.summaries,
            key=lambda x: (-x.score, x.text.name),
        )
        filtered_contexts = [c for c in filtered_contexts if c.score > 0]
        return filtered_contexts  # [: (max_sources if max_sources is not None else self.max_sources) ]

    def get_string(self, max_sources: Optional[int] = None) -> str:
        def format_quotes(points: List[Point]) -> str:
            if not points:
                return ""
            return "\nRelevant quotes:\n" + "\n".join(
                f'quote{i+1}: "{p.quote}"' for i, p in enumerate(points)
            )

        inner_context_strs = [
            CONTEXT_INNER_PROMPT_WITH_QUOTE.format(
                name=c.text.name,
                text=c.context,
                quotes=format_quotes(c.points),
                citation=c.text.doc.citation,
                **(c.model_extra or {}),
            )
            for c in self.filtered_contexts(max_sources)
        ]
        context_str = CONTEXT_OUTER_PROMPT.format(
            context_str="\n\n".join(inner_context_strs),
            valid_keys=", ".join(
                [c.text.name for c in self.filtered_contexts(max_sources)]
            ),
        )
        return context_str
