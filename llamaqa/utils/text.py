from pydantic import Field
from typing import List
import re

import numpy as np

from .doc import Doc
from .embeddable import Embeddable

class Text(Embeddable):
    text: str
    name: str
    doc: Doc

    def __hash__(self) -> int:
        return hash(self.text)

class TextPlus(Text):
    pages: List[int] = []
    text: str
    name: str
    doc: Doc

    def __hash__(self) -> int:
        return hash(self.text)

    @classmethod
    def from_text(cls, text: Text):
        # Retrieve page numbers
        pattern = re.compile(".*? pages (\\d+)-(\\d+)")
        matches = pattern.match(text.name)
        if matches:
            start = int(matches.groups()[0])
            end = int(matches.groups()[1])
            return cls(
                text=text.text,
                name=text.name,
                doc=text.doc,
                pages=[n+start for n in np.arange(end-start+1)],
                embedding=text.embedding,
            )
        else:
            return cls(
                text=text.text,
                name=text.name,
                doc=text.doc,
                embedding=text.embedding,
            )