from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, cast
import logging
import os
import re

from pydantic import (
    BaseModel,
)

from .doc import Doc, Text, Point
from .parsing_settings import ParsingSettings
from .utils import (
    generate_dockey,
    maybe_is_text,
    read_doc,
    summarize_chunk,
)
from ..llms.embedding_model import EmbeddingModel
from ..llms.llm_model import LLMModel
from ..utils.utils import gather_with_concurrency


logger = logging.getLogger(__name__)


class Reader(BaseModel):
    parse_config: ParsingSettings
    llm_model: LLMModel
    embedding_model: EmbeddingModel

    async def get_metadata(
        self,
        path: Path,
        title: Optional[str] = None,
        citation: Optional[str] = None,
        abstract: Optional[str] = None,
        authors: Optional[List[str]] | None = None,
        published_at: Optional[str] = None,
        doi: Optional[str] = None,
        docname: str | None = None,
        dockey: Any | None = None,
        filepath: Optional[str] = None,
    ):
        # Parse citation
        if any(
            [
                arg is None
                for arg in [
                    citation,
                    title,
                    authors,
                    published_at,
                    doi,
                    abstract,
                ]
            ]
        ):
            # Peek first chunk
            texts = read_doc(
                path,
                Doc(docname="", citation="", dockey=None),  # Fake doc
                chunk_chars=self.parse_config.chunk_size,
                overlap=self.parse_config.overlap,
                page_size_limit=self.parse_config.page_size_limit,
            )
            if not texts:
                raise ValueError(f"Could not read document {path}. Is it empty?")
            result = await self.llm_model.run_prompt(
                prompt=self.parse_config.citation_json_prompt,
                data={"text": texts[0].text},
                skip_system=True,  # skip system because it's too hesitant to answer
            )
            try:
                citation_json = result.to_json()
                citation = citation or citation_json.get("citation")
                title = title or citation_json.get("title")
                authors = authors or citation_json.get("authors")
                published_at = published_at or citation_json.get("published_at")
                doi = doi or citation_json.get("doi")
                abstract = abstract or citation_json.get("abstract")

            except ValueError:
                logger.warn("Unable to load JSON from parsed citation")

                result = await self.llm_model.run_prompt(
                    prompt=self.parse_config.citation_prompt,
                    data={"text": texts[0].text},
                    skip_system=True,  # skip system because it's too hesitant to answer
                )
                citation = result.text
                if (
                    len(citation) < 3  # noqa: PLR2004
                    or "Unknown" in citation
                    or "insufficient" in citation
                ):
                    citation = (
                        f"Unknown, {os.path.basename(path)}, {datetime.now().year}"
                    )

        if citation is None:
            raise ValueError(
                f"Unable to infer citation for {path}, please provide a citation"
            )

        # Generate dockey from citation info to support dedup
        if dockey is None:
            dockey = generate_dockey(citation)

        # Generate docname
        if docname is None:
            # get first name and year from citation
            match = re.search(r"([A-Z][a-z]+)", citation)
            if match is not None:
                author = match.group(1)
            else:
                # panicking - no word??
                raise ValueError(
                    f"Could not parse docname from citation {citation}. "
                    "Consider just passing key explicitly - e.g. docs.py "
                    "(path, citation, key='mykey')"
                )
            year = ""
            match = re.search(r"(\d{4})", citation)
            if match is not None:
                year = match.group(1)
            docname = f"{author}{year}"

        doc = Doc(
            title=title,
            authors=authors,
            docname=docname,
            citation=citation,
            dockey=dockey,
            doi=doi,
            published_at=published_at,
            abstract=abstract,
            filepath=filepath,
        )

        # TODO
        # see if we can upgrade to DocDetails
        # if not, we can progress with a normal Doc
        # if "overwrite_fields_from_metadata" is used:
        # will map "docname" to "key", and "dockey" to "doc_id"
        # if (title or doi) and self.parse_config.use_doc_details:
        #     if kwargs.get("metadata_client"):
        #         metadata_client = kwargs["metadata_client"]
        #     else:
        #         metadata_client = DocMetadataClient(
        #             session=kwargs.pop("session", None),
        #             clients=kwargs.pop("clients", DEFAULT_CLIENTS),
        #         )

        #     query_kwargs: dict[str, Any] = {}

        #     if doi:
        #         query_kwargs["doi"] = doi
        #     if authors:
        #         query_kwargs["authors"] = authors
        #     if title:
        #         query_kwargs["title"] = title

        #     doc = await metadata_client.upgrade_doc_to_doc_details(
        #         doc, **(query_kwargs | kwargs)
        #     )

        abstract_emb = None
        if abstract:
            abstract_emb = (
                await self.embedding_model.embed_documents(texts=[abstract])
            )[0]
        doc.embedding = abstract_emb
        return doc

    async def read_doc(
        self,
        path: Path,
        title: Optional[str] = None,
        citation: Optional[str] = None,
        abstract: Optional[str] = None,
        authors: Optional[List[str]] | None = None,
        published_at: Optional[str] = None,
        doi: Optional[str] = None,
        docname: str | None = None,
        dockey: Any | None = None,
        filepath: Optional[str] = None,
        summarize_chunks=False,
        **kwargs,
    ) -> Doc:
        doc = self.get_metadata(
            path,
            title,
            citation,
            abstract,
            authors,
            published_at,
            doi,
            docname,
            dockey,
            filepath,
        )

        # Read document and chunk text
        texts = read_doc(
            path,
            doc,
            chunk_chars=self.parse_config.chunk_size,
            overlap=self.parse_config.overlap,
            page_size_limit=self.parse_config.page_size_limit,
        )
        texts = cast(List[Text], texts)
        # loose check to see if document was loaded
        if (
            not texts
            or len(texts[0].text) < 10  # noqa: PLR2004
            or (
                not self.parse_config.disable_doc_valid_check
                and not maybe_is_text(texts[0].text)
            )
        ):
            raise ValueError(
                f"This does not look like a text document: {path}. Pass disable_check"
                " to ignore this error."
            )

        for t, t_embedding in zip(
            texts,
            await self.embedding_model.embed_documents(texts=[t.text for t in texts]),
            strict=True,
        ):
            t.embedding = t_embedding

        if summarize_chunks:
            results = await gather_with_concurrency(
                n=4,
                coros=[
                    summarize_chunk(
                        text=text.text,
                        llm_model=self.llm_model,
                    )
                    for text in texts
                ],
                progress=True,
            )
            for text, summary in zip(texts, results, strict=True):
                text.summary = summary["summary"]
                text.points = [Point(**p) for p in summary["points"]]

        doc.texts = texts

        return doc
