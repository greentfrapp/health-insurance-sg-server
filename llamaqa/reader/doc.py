from datetime import datetime
from pydantic import BaseModel, Field
from typing import Any, List, Optional
import re

import numpy as np

from ..utils.embeddable import Embeddable


class Point(BaseModel):
    point: str
    quote: str


class Text(Embeddable):
    text: str
    name: str
    doc: "Doc"
    pages: List[int] = []
    summary: Optional[str] = None
    points: List[Point] = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if len(self.pages) == 0:
            self.pages = self._get_pages_from_text_name(self.name)

    def __hash__(self) -> int:
        return hash(self.text)

    @staticmethod
    def _get_pages_from_text_name(text_name: str):
        pattern = re.compile(".*? pages (\\d+)-(\\d+)")
        matches = pattern.match(text_name)
        if matches:
            start = int(matches.groups()[0])
            end = int(matches.groups()[1])
            return [int(n + start) for n in np.arange(end - start + 1)]
        else:
            return []


class Doc(Embeddable):
    docname: str
    citation: str
    title: str = ""
    authors: Optional[List[str]] = None
    abstract: Optional[str] = None
    filepath: Optional[str] = None
    published_at: Optional[datetime] = None
    doi: Optional[str] = None
    dockey: Any
    overwrite_fields_from_metadata: bool = Field(
        default=True,
        description=(
            "flag to overwrite fields from metadata when upgrading to a DocDetails"
        ),
    )
    texts: List[Text] = []

    def __hash__(self) -> int:
        return hash((self.docname, self.dockey))
