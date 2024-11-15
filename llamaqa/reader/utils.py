from __future__ import annotations

import math
import os
import re
import string
from pathlib import Path
from typing import ClassVar, Literal, overload
from uuid import uuid5, UUID

import pypdf
import tiktoken
from html2text import __version__ as html2text_version
from html2text import html2text
from pydantic import BaseModel

from llamaqa import __version__ as llamaqa_version
from ..llms.llm_model import LLMModel
from ..reader.doc import Doc, Text


NAMESPACE_CITATION = UUID("5345abad-94db-4db0-a1b1-6107ba7a4cb7")


class ImpossibleParsingError(Exception):
    """Error to throw when a parsing is impossible."""

    LOG_METHOD_NAME: ClassVar[str] = "warning"


class ChunkMetadata(BaseModel):
    """Metadata for chunking algorithm."""

    chunk_chars: int
    overlap: int
    chunk_type: str


class ParsedMetadata(BaseModel):
    """Metadata for parsed text."""

    parsing_libraries: list[str]
    total_parsed_text_length: int
    llamaqa_version: str = llamaqa_version
    parse_type: str | None = None
    chunk_metadata: ChunkMetadata | None = None


class ParsedText(BaseModel):
    """Parsed text (pre-chunking)."""

    content: dict | str | list[str]
    metadata: ParsedMetadata

    def encode_content(self):
        # we tokenize using tiktoken so cuts are in reasonable places
        # See https://github.com/openai/tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        if isinstance(self.content, str):
            return enc.encode_ordinary(self.content)
        elif isinstance(self.content, list):  # noqa: RET505
            return [enc.encode_ordinary(c) for c in self.content]
        else:
            raise NotImplementedError(
                "Encoding only implemented for str and list[str] content."
            )

    def reduce_content(self) -> str:
        """Reduce any content to a string."""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            return "\n\n".join(self.content)
        return "\n\n".join(self.content.values())


# def parse_pdf_to_pages(
#     path: str | os.PathLike, page_size_limit: int | None = None
# ) -> ParsedText:

#     with pymupdf.open(path) as file:
#         pages: dict[str, str] = {}
#         total_length = 0

#         for i in range(file.page_count):
#             try:
#                 page = file.load_page(i)
#             except pymupdf.mupdf.FzErrorFormat as exc:
#                 raise ImpossibleParsingError(
#                     f"Page loading via {pymupdf.__name__} failed on page {i} of"
#                     f" {file.page_count} for the PDF at path {path}, likely this PDF"
#                     " file is corrupt."
#                 ) from exc
#             text = page.get_text("text", sort=True)
#             if page_size_limit and len(text) > page_size_limit:
#                 raise ImpossibleParsingError(
#                     f"The text in page {i} of {file.page_count} was {len(text)} chars"
#                     f" long, which exceeds the {page_size_limit} char limit for the PDF"
#                     f" at path {path}."
#                 )
#             pages[str(i + 1)] = text
#             total_length += len(text)

#     metadata = ParsedMetadata(
#         parsing_libraries=[f"pymupdf ({pymupdf.__version__})"],
#         llamaqa_version=llamaqa_version,
#         total_parsed_text_length=total_length,
#         parse_type="pdf",
#     )
#     return ParsedText(content=pages, metadata=metadata)


def parse_pdf_to_pages(
    path: str | os.PathLike, page_size_limit: int | None = None
) -> ParsedText:
    with open(path, "rb") as pdf_file:
        pdf_reader = pypdf.PdfReader(pdf_file)
        pages: dict[str, str] = {}
        total_length = 0
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            text = re.sub(r"[\s\n]+", " ", text).replace("\x00", "").strip()
            if page_size_limit and len(text) > page_size_limit:
                raise ImpossibleParsingError(
                    f"The text in page {i} was {len(text)} chars"
                    f" long, which exceeds the {page_size_limit} char limit for the PDF"
                    f" at path {path}."
                )
            pages[str(i + 1)] = text
            total_length += len(text)
    metadata = ParsedMetadata(
        parsing_libraries=[f"pypdf ({pypdf.__version__})"],
        llamaqa_version=llamaqa_version,
        total_parsed_text_length=total_length,
        parse_type="pdf",
    )
    return ParsedText(content=pages, metadata=metadata)


def chunk_pdf(
    parsed_text: ParsedText, doc: Doc, chunk_chars: int, overlap: int
) -> list[Text]:
    pages: list[str] = []
    texts: list[Text] = []
    split: str = ""

    if not isinstance(parsed_text.content, dict):
        raise NotImplementedError(
            f"ParsedText.content must be a `dict`, not {type(parsed_text.content)}."
        )

    if not parsed_text.content:
        raise ImpossibleParsingError(
            f"No text was parsed from the document named {doc.docname!r} with ID"
            f" {doc.dockey}, either empty or corrupted."
        )

    for page_num, page_text in parsed_text.content.items():
        split += page_text
        pages.append(page_num)
        # split could be so long it needs to be split
        # into multiple chunks. Or it could be so short
        # that it needs to be combined with the next chunk.
        while len(split) > chunk_chars:
            # pretty formatting of pages (e.g. 1-3, 4, 5-7)
            pg = "-".join([pages[0], pages[-1]])
            texts.append(
                Text(
                    text=split[:chunk_chars], name=f"{doc.docname} pages {pg}", doc=doc
                )
            )
            split = split[chunk_chars - overlap :]
            pages = [page_num]

    if len(split) > overlap or not texts:
        pg = "-".join([pages[0], pages[-1]])
        texts.append(
            Text(text=split[:chunk_chars], name=f"{doc.docname} pages {pg}", doc=doc)
        )
    return texts


def parse_text(
    path: str | os.PathLike,
    html: bool = False,
    split_lines: bool = False,
    use_tiktoken: bool = True,
    page_size_limit: int | None = None,
) -> ParsedText:
    """Simple text splitter, can optionally use tiktoken, parse html, or split into newlines.

    Args:
        path: path to file.
        html: flag to use html2text library for parsing.
        split_lines: flag to split lines into a list.
        use_tiktoken: flag to use tiktoken library to encode text.
        page_size_limit: optional limit on the number of characters per page. Only
            relevant when split_lines is True.
    """
    path = Path(path)
    try:
        with path.open() as f:
            text = list(f) if split_lines else f.read()
    except UnicodeDecodeError:
        with path.open(encoding="utf-8", errors="ignore") as f:
            text = f.read()

    parsing_libraries: list[str] = ["tiktoken (cl100k_base)"] if use_tiktoken else []
    if html:
        if not isinstance(text, str):
            raise NotImplementedError(
                "HTML parsing is not yet set up to work with split_lines."
            )
        parse_type: str = "html"
        text = html2text(text)
        parsing_libraries.append(f"html2text ({html2text_version})")
    else:
        parse_type = "txt"
    if isinstance(text, str):
        total_length: int = len(text)
    else:
        total_length = sum(len(t) for t in text)
        for i, t in enumerate(text):
            if page_size_limit and len(text) > page_size_limit:
                raise ImpossibleParsingError(
                    f"The {parse_type} on page {i} of {len(text)} was {len(t)} chars"
                    f" long, which exceeds the {page_size_limit} char limit at path"
                    f" {path}."
                )
    return ParsedText(
        content=text,
        metadata=ParsedMetadata(
            parsing_libraries=parsing_libraries,
            llamaqa_version=llamaqa_version,
            total_parsed_text_length=total_length,
            parse_type=parse_type,
        ),
    )


def chunk_text(
    parsed_text: ParsedText,
    doc: Doc,
    chunk_chars: int,
    overlap: int,
    use_tiktoken: bool = True,
) -> list[Text]:
    """Parse a document into chunks, based on tiktoken encoding.

    NOTE: We get some byte continuation errors.
    Currently ignored, but should explore more to make sure we don't miss anything.
    """
    texts: list[Text] = []
    enc = tiktoken.get_encoding("cl100k_base")

    if not isinstance(parsed_text.content, str):
        raise NotImplementedError(
            f"ParsedText.content must be a `str`, not {type(parsed_text.content)}."
        )

    content = parsed_text.content if not use_tiktoken else parsed_text.encode_content()
    if not content:  # Avoid div0 in token calculations
        raise ImpossibleParsingError(
            f"No text was parsed from the document named {doc.docname!r} with ID"
            f" {doc.dockey}, either empty or corrupted."
        )

    # convert from characters to chunks
    char_count = parsed_text.metadata.total_parsed_text_length  # e.g., 25,000
    token_count = len(content)  # e.g., 4,500
    chars_per_token = char_count / token_count  # e.g., 5.5
    chunk_tokens = chunk_chars / chars_per_token  # e.g., 3000 / 5.5 = 545
    overlap_tokens = overlap / chars_per_token  # e.g., 100 / 5.5 = 18
    chunk_count = math.ceil(token_count / chunk_tokens)  # e.g., 4500 / 545 = 9

    for i in range(chunk_count):
        split = content[
            max(int(i * chunk_tokens - overlap_tokens), 0) : int(
                (i + 1) * chunk_tokens + overlap_tokens
            )
        ]
        texts.append(
            Text(
                text=enc.decode(split) if use_tiktoken else split,
                name=f"{doc.docname} chunk {i + 1}",
                doc=doc,
            )
        )
    return texts


def chunk_code_text(
    parsed_text: ParsedText, doc: Doc, chunk_chars: int, overlap: int
) -> list[Text]:
    """Parse a document into chunks, based on line numbers (for code)."""
    split = ""
    texts: list[Text] = []
    last_line = 0

    if not isinstance(parsed_text.content, list):
        raise NotImplementedError(
            f"ParsedText.content must be a `list`, not {type(parsed_text.content)}."
        )

    for i, line in enumerate(parsed_text.content):
        split += line
        while len(split) > chunk_chars:
            texts.append(
                Text(
                    text=split[:chunk_chars],
                    name=f"{doc.docname} lines {last_line}-{i}",
                    doc=doc,
                )
            )
            split = split[chunk_chars - overlap :]
            last_line = i
    if len(split) > overlap or not texts:
        texts.append(
            Text(
                text=split[:chunk_chars],
                name=f"{doc.docname} lines {last_line}-{i}",
                doc=doc,
            )
        )
    return texts


@overload
def read_doc(
    path: str | os.PathLike,
    doc: Doc,
    parsed_text_only: Literal[False],
    include_metadata: Literal[False],
    chunk_chars: int = ...,
    overlap: int = ...,
    page_size_limit: int | None = ...,
) -> list[Text]: ...


@overload
def read_doc(
    path: str | os.PathLike,
    doc: Doc,
    parsed_text_only: Literal[False] = ...,
    include_metadata: Literal[False] = ...,
    chunk_chars: int = ...,
    overlap: int = ...,
    page_size_limit: int | None = ...,
) -> list[Text]: ...


@overload
def read_doc(
    path: str | os.PathLike,
    doc: Doc,
    parsed_text_only: Literal[True],
    include_metadata: bool = ...,
    chunk_chars: int = ...,
    overlap: int = ...,
    page_size_limit: int | None = ...,
) -> ParsedText: ...


@overload
def read_doc(
    path: str | os.PathLike,
    doc: Doc,
    parsed_text_only: Literal[False],
    include_metadata: Literal[True],
    chunk_chars: int = ...,
    overlap: int = ...,
    page_size_limit: int | None = ...,
) -> tuple[list[Text], ParsedMetadata]: ...


def read_doc(
    path: str | os.PathLike,
    doc: Doc,
    parsed_text_only: bool = False,
    include_metadata: bool = False,
    chunk_chars: int = 3000,
    overlap: int = 100,
    page_size_limit: int | None = None,
) -> list[Text] | ParsedText | tuple[list[Text], ParsedMetadata]:
    """Parse a document and split into chunks.

    Optionally can include just the parsing as well as metadata about the parsing/chunking

    Args:
        path: local document path
        doc: object with document metadata
        parsed_text_only: return parsed text without chunking
        include_metadata: return a tuple
        chunk_chars: size of chunks
        overlap: size of overlap between chunks
        page_size_limit: optional limit on the number of characters per page
    """
    str_path = str(path)
    parsed_text = None

    # start with parsing -- users may want to store this separately
    if str_path.endswith(".pdf"):
        parsed_text = parse_pdf_to_pages(path, page_size_limit=page_size_limit)
    elif str_path.endswith(".txt"):
        parsed_text = parse_text(path, page_size_limit=page_size_limit)
    elif str_path.endswith(".html"):
        parsed_text = parse_text(path, html=True, page_size_limit=page_size_limit)
    else:
        parsed_text = parse_text(
            path, split_lines=True, use_tiktoken=False, page_size_limit=page_size_limit
        )

    if parsed_text_only:
        return parsed_text

    # next chunk the parsed text

    # check if chunk is 0 (no chunking)
    if chunk_chars == 0:
        chunked_text = [
            Text(text=parsed_text.reduce_content(), name=doc.docname, doc=doc)
        ]
        chunk_metadata = ChunkMetadata(chunk_chars=0, overlap=0, chunk_type="no_chunk")
    elif str_path.endswith(".pdf"):
        chunked_text = chunk_pdf(
            parsed_text, doc, chunk_chars=chunk_chars, overlap=overlap
        )
        chunk_metadata = ChunkMetadata(
            chunk_chars=chunk_chars, overlap=overlap, chunk_type="overlap_pdf_by_page"
        )
    elif str_path.endswith((".txt", ".html")):
        chunked_text = chunk_text(
            parsed_text, doc, chunk_chars=chunk_chars, overlap=overlap
        )
        chunk_metadata = ChunkMetadata(
            chunk_chars=chunk_chars, overlap=overlap, chunk_type="overlap"
        )
    else:
        chunked_text = chunk_code_text(
            parsed_text, doc, chunk_chars=chunk_chars, overlap=overlap
        )
        chunk_metadata = ChunkMetadata(
            chunk_chars=chunk_chars, overlap=overlap, chunk_type="overlap_code_by_line"
        )

    if include_metadata:
        parsed_text.metadata.chunk_metadata = chunk_metadata
        return chunked_text, parsed_text.metadata

    return chunked_text


def generate_dockey(citation: str) -> str:
    return str(uuid5(NAMESPACE_CITATION, citation))


def maybe_is_text(s: str, thresh: float = 2.5) -> bool:
    if not s:
        return False
    # Calculate the entropy of the string
    entropy = 0.0
    for c in string.printable:
        p = s.count(c) / len(s)
        if p > 0:
            entropy += -p * math.log2(p)

    # Check if the entropy is within a reasonable range for text
    return entropy > thresh


SUMMARY_JSON_PROMPT = """{text}

Summarize the text above and respond with the following JSON format:

{{
  "summary": "...",
  "points": [
    {{
        "quote": "...",
        "point": "..."
    }}
  ]
}}

where `summary` is relevant information from text - about 100 words,
and `points` is an array of maximum 10 `point` and `quote` pairs that supports the summary
where each `quote` is an exact match quote (max 50 words) from the text that
best supports the respective `point`.
Make sure that the quote is an exact match with the same capitalization
and without truncation or changes.
Do not truncate the quote with any ellipsis.

If the text is a placeholder or if there is nothing to summarize, simply return null as the summary:
{{
    "summary": null,
    "points": []
}}
"""


async def summarize_chunk(text: str, llm_model: LLMModel):
    result = await llm_model.run_prompt(
        prompt=SUMMARY_JSON_PROMPT,
        data={"text": text},
        skip_system=True,  # skip system because it's too hesitant to answer
    )
    summary_json = result.to_json()
    return summary_json
