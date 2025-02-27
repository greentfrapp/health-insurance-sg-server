from typing import List

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from ..reader.doc import Point, Text


class Context(BaseModel):
    """A class to hold the context of a question."""

    model_config = ConfigDict(extra="allow")

    context: str = Field(description="Summary of the text with respect to a question.")
    text: Text
    score: int = 5
    points: List[Point] = []

    def __str__(self) -> str:
        """Return the context as a string."""
        return self.context
