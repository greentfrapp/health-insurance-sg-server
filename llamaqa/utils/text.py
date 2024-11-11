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
    pages: List[int] = []
    
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
            return [int(n+start) for n in np.arange(end-start+1)]
        else:
            return []
