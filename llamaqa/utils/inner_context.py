from typing import List

from .context import Context
from .text import TextPlus


CONTEXT_OUTER_PROMPT = "{context_str}\n\nValid Keys: {valid_keys}"
CONTEXT_INNER_PROMPT = "{name}: {text}\nFrom {citation}"


CONTEXT_INNER_PROMPT_WITH_QUOTE = "{name}: {text}\nRelevant Quotes:\n{quotes}\nFrom {citation}"


class InnerContext:
    chunks: List[TextPlus] = []
    summaries: List[Context] = []
    max_sources: int = 5

    @property
    def filtered_contexts(self):
        filtered_contexts = sorted(
            self.summaries,
            key=lambda x: (-x.score, x.text.name),
        )[: self.max_sources]
        # remove any contexts with a score of 0
        filtered_contexts = [c for c in filtered_contexts if c.score > 0]
        return filtered_contexts

    def get_string(self) -> str:
        inner_context_strs = [
            CONTEXT_INNER_PROMPT_WITH_QUOTE.format(
                name=c.text.name,
                text=c.context,
                quotes="\n".join(f"quote{i+1}: \"{p['quote']}\"" for i, p in enumerate(c.points)),
                citation=c.text.doc.citation,
                **(c.model_extra or {}),
            )
            for c in self.filtered_contexts
        ]
        context_str = CONTEXT_OUTER_PROMPT.format(
            context_str="\n\n".join(inner_context_strs),
            valid_keys=", ".join([c.text.name for c in self.filtered_contexts]),
        )
        return context_str
