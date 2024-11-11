from collections.abc import Iterable, Sequence
from pydantic import Field
from typing import Any
import json
import re

from supabase._async.client import create_client as create_async_client
import numpy as np

from .store import VectorStore, cosine_similarity
from ..utils.doc import Doc
from ..utils.embeddable import Embeddable
from ..utils.text import Text


class SupabaseStore(VectorStore):
    texts: list[Embeddable] = []
    _embeddings_matrix: np.ndarray | None = None
    supabase_url: str = Field()
    supabase_key: str = Field()

    def clear(self) -> None:
        super().clear()
        self.texts = []
        self._embeddings_matrix = None

    def add_texts_and_embeddings(self, texts: Iterable[Embeddable]) -> None:
        super().add_texts_and_embeddings(texts)
        self.texts.extend(texts)
        self._embeddings_matrix = np.array([t.embedding for t in self.texts])

    async def similarity_search(
        self, query: str, k: int, embedding_model: Any
    ) -> tuple[Sequence[Text], list[float]]:
        supabase = await create_async_client(self.supabase_url, self.supabase_key)
        
        if not self.texts: 
            response = (
                await supabase.table("chunks")
                .select("document(id,citation,filepath),pages,text,text_emb")
                # .eq("document", "02d7b71c-fece-582f-ac84-d976360b6292")
                .execute()
            )
            self.texts = []
            for chunk in response.data:
                docname = None
                document = chunk.get("document")
                dockey = document.get("id")
                citation = document.get("citation")
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
                pages = chunk.get("pages")
                pages_str = " pages " + f"{pages[0]}-{pages[-1]}"
                self.texts.append(
                    Text(
                        text=chunk.get("text"),
                        name=docname + pages_str,
                        doc=Doc(
                            dockey=dockey,
                            citation=citation,
                            docname=docname,
                            filepath=document.get("filepath"),
                        ),
                        pages=chunk.get("pages"),
                        embedding=json.loads(chunk.get("text_emb")),
                    )
                )
        self._embeddings_matrix = np.array([t.embedding for t in self.texts])

        k = min(k, len(self.texts))
        if k == 0:
            return [], []

        np_query = np.array((await embedding_model.embed_documents([query]))[0])

        similarity_scores = cosine_similarity(
            np_query.reshape(1, -1), self._embeddings_matrix
        )[0]
        similarity_scores = np.nan_to_num(similarity_scores, nan=-np.inf)
        # minus so descending
        # we could use arg-partition here
        # but a lot of algorithms expect a sorted list
        sorted_indices = np.argsort(-similarity_scores)
        return (
            [self.texts[i] for i in sorted_indices[:k]],
            [similarity_scores[i] for i in sorted_indices[:k]],
        )
