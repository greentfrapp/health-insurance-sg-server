from enum import StrEnum
from typing import assert_never

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from llamaqa import __version__


CITATION_PROMPT = (
    "Provide the citation for the following text in MLA Format. "
    "Do not write an introductory sentence. "
    "If reporting date accessed, the current year is 2024\n\n"
    "{text}\n\n"
    "Citation:"
)


CITATION_JSON_PROMPT = (
    "Infer the title, authors, citation, publication date, doi and abstract as a JSON from this text. "
    "If any field can not be found, return it as null. "
    "Bonus points for inferring the DOI. "
    "Return your result in the following format "
    "Use title, authors, citation, published_at, doi and abstract as keys. "
    '"citation" should be the citation in MLA format. '
    '"authors" should be a list of authors with correct capitalization. '
    '"published_at" should be a formatted timestamp in the following format yyyy-mm-dd. '
    "{text}\n\n"
    "Citation JSON:"
)


class ParsingOptions(StrEnum):
    PAPERQA_DEFAULT = "paperqa_default"

    def available_for_inference(self) -> list["ParsingOptions"]:
        return [self.PAPERQA_DEFAULT]  # type: ignore[list-item]


def _get_parse_type(opt: ParsingOptions, config: "ParsingSettings") -> str:
    if opt == ParsingOptions.PAPERQA_DEFAULT:
        return config.parser_version_string
    assert_never(opt)


class ChunkingOptions(StrEnum):
    SIMPLE_OVERLAP = "simple_overlap"

    @property
    def valid_parsings(self) -> list[ParsingOptions]:
        # Note that SIMPLE_OVERLAP must be valid for all by default
        # TODO: implement for future parsing options
        valid_parsing_dict: dict[str, list[ParsingOptions]] = {}
        return valid_parsing_dict.get(self.value, [])


class ParsingSettings(BaseModel):
    """Settings relevant for parsing and chunking documents."""

    model_config = ConfigDict(extra="forbid")

    chunk_size: int = Field(
        default=5000,
        description="Number of characters per chunk. If 0, no chunking will be done.",
    )
    page_size_limit: int | None = Field(
        default=1_280_000,
        description=(
            "Optional limit on the number of characters to parse in one 'page', default"
            " is 1.28 million chars, 10X larger than a 128k tokens context limit"
            " (ignoring chars vs tokens difference)."
        ),
    )
    use_doc_details: bool = Field(
        default=False, description="Whether to try to get metadata details for a Doc"
    )
    overlap: int = Field(
        default=250, description="Number of characters to overlap chunks"
    )
    citation_prompt: str = Field(
        default=CITATION_PROMPT,
        description="Prompt that tries to create citation from peeking one page",
    )
    citation_json_prompt: str = Field(
        default=CITATION_JSON_PROMPT,
        description="Prompt that tries to create citation JSON from peeking one page",
    )
    disable_doc_valid_check: bool = Field(
        default=False,
        description=(
            "Whether to disable checking if a document looks like text (was parsed"
            " correctly)"
        ),
    )
    defer_embedding: bool = Field(
        default=False,
        description=(
            "Whether to embed documents immediately as they are added, or defer until"
            " summarization."
        ),
    )
    chunking_algorithm: ChunkingOptions = ChunkingOptions.SIMPLE_OVERLAP

    def chunk_type(self, chunking_selection: ChunkingOptions | None = None) -> str:
        """Future chunking implementations (i.e. by section) will get an elif clause here."""
        if chunking_selection is None:
            chunking_selection = self.chunking_algorithm
        if chunking_selection == ChunkingOptions.SIMPLE_OVERLAP:
            return (
                f"{self.parser_version_string}|{chunking_selection.value}"
                f"|tokens={self.chunk_size}|overlap={self.overlap}"
            )
        assert_never(chunking_selection)

    @property
    def parser_version_string(self) -> str:
        return f"paperqa-{__version__}"

    def is_chunking_valid_for_parsing(self, parsing: str):
        # must map the parsings because they won't include versions by default
        return self.chunking_algorithm == ChunkingOptions.SIMPLE_OVERLAP or parsing in {  # type: ignore[unreachable]
            _get_parse_type(p, self) for p in self.chunking_algorithm.valid_parsings
        }
