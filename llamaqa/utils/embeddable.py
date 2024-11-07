from pydantic import BaseModel, Field


class Embeddable(BaseModel):
    embedding: list[float] | None = Field(default=None, repr=False)
