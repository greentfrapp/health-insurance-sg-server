

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)
from typing import Any
from uuid import UUID, uuid4

from llamaqa.llms.llm_result import LLMResult
from .context import Context
from .doc import Doc
from .text import TextPlus


class Answer(BaseModel):
    """A class to hold the answer to a question."""

    model_config = ConfigDict(extra="ignore")

    id: UUID = Field(default_factory=uuid4)
    question: str
    answer: str = ""
    context: str = ""
    contexts: list[Context] = Field(default_factory=list)
    bib: dict[str, Context] = Field(default_factory=dict)
    references: str = ""
    formatted_answer: str = ""
    cost: float = 0.0
    # Map model name to a two-item list of LLM prompt token counts
    # and LLM completion token counts
    token_counts: dict[str, list[int]] = Field(default_factory=dict)
    config_md5: str | None = Field(
        default=None,
        frozen=True,
        description=(
            "MD5 hash of the settings used to generate the answer. Cannot change"
        ),
    )

    def __str__(self) -> str:
        """Return the answer as a string."""
        return self.formatted_answer

    @model_validator(mode="before")
    @classmethod
    def remove_computed(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data.pop("used_contexts", None)
        return data

    def get_citation(self, name: str) -> str:
        """Return the formatted citation for the given docname."""
        try:
            doc: Doc = next(
                filter(lambda x: x.text.name == name, self.contexts)
            ).text.doc
        except StopIteration as exc:
            raise ValueError(f"Could not find docname {name} in contexts.") from exc
        return doc.citation

    def get_unique_docs_from_contexts(self, score_threshold: int = 0) -> set[Doc]:
        """Parse contexts for docs with scores above the input threshold."""
        return {
            c.text.doc
            for c in filter(lambda x: x.score >= score_threshold, self.contexts)
        }

    def filter_content_for_user(self) -> None:
        """Filter out extra items (inplace) that do not need to be returned to the user."""
        self.contexts = [
            Context(
                context=c.context,
                score=c.score,
                text=TextPlus(
                    text="",
                    **c.text.model_dump(exclude={"text", "embedding", "doc"}),
                    doc=Doc(**c.text.doc.model_dump(exclude={"embedding"})),
                ),
            )
            for c in self.contexts
        ]

    @property
    def could_not_answer(self) -> bool:
        return "cannot answer" in self.answer.lower()
