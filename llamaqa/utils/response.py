from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

class Response(BaseModel):

    model_config = ConfigDict(extra="ignore")

    question: str
    answer: str = ""
    # context: str = ""
    # contexts: list[Context] = Field(default_factory=list)
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

    @property
    def could_not_answer(self) -> bool:
        return "cannot answer" in self.answer.lower()
