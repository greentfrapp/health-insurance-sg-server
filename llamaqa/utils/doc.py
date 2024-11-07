from pydantic import Field
from typing import Any

from .embeddable import Embeddable


class Doc(Embeddable):
    docname: str
    citation: str
    filepath: str = ""
    dockey: Any
    overwrite_fields_from_metadata: bool = Field(
        default=True,
        description=(
            "flag to overwrite fields from metadata when upgrading to a DocDetails"
        ),
    )

    def __hash__(self) -> int:
        return hash((self.docname, self.dockey))
