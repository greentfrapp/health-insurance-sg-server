from collections.abc import Iterable, Sequence
from pydantic import Field
from typing import Any, List, Optional
import json
import re

from postgrest.exceptions import APIError
from supabase._async.client import create_client as create_async_client
import numpy as np

from .store import VectorStore, cosine_similarity
from .utils import upload_chunk
from ..llms.embedding_model import EmbeddingModel
from ..reader.doc import Doc, Text, Point
from ..utils.embeddable import Embeddable
from ..utils.policies import POLICY_IDS
from ..utils.utils import gather_with_concurrency


def response_to_text(data: List) -> List[Text]:
    texts = []
    for chunk in data:
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

        points = [Point(**p) for p in chunk.get("points", [])]

        texts.append(
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
                summary=chunk.get("summary"),
                points=points,
            )
        )
    return texts


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

    async def get_all_policy_info(self, policies: List[str]):
        document_ids = set(sum([POLICY_IDS[p] for p in policies], []))
        # Support offline retrieval
        if self.texts:
            return [t for t in self.texts if t["dockey"] in document_ids]
        # Support Supabase retrieval
        supabase = await create_async_client(self.supabase_url, self.supabase_key)
        response = (
            await supabase.table("chunks")
            .select("document(id,citation,filepath),pages,text,text_emb,summary,points")
            .or_(",".join([f"document.eq.{d}" for d in document_ids]))
            .not_.is_("summary", "null")
            .execute()
        )
        return response_to_text(response.data)

    async def pgvector_search(
        self,
        query: str,
        embedding_model: EmbeddingModel,
        match_threshold: float = 0.5,
        k: int = 10,
        document_ids: Optional[List[str]] = None,
    ):
        supabase = await create_async_client(self.supabase_url, self.supabase_key)
        np_query = np.array((await embedding_model.embed_documents([query]))[0])
        response = await supabase.rpc(
            "match_chunks",
            {
                "query_embedding": list(np_query),
                "match_threshold": match_threshold,
                "match_count": k,
                "document_ids": document_ids or [],
            },
        ).execute()
        # Format response.data
        data = []
        for row in response.data:
            data.append(
                {
                    "document": {
                        "id": row["doc_id"],
                        "citation": row["doc_citation"],
                        "filepath": row["doc_filepath"],
                    },
                    "pages": row["pages"],
                    "text": row["text"],
                    "text_emb": row["text_emb"],
                    "similarity": row["similarity"],
                }
            )
        return data

    async def similarity_search(
        self,
        query: str,
        k: int,
        embedding_model: EmbeddingModel,
        policies: Optional[List[str]] = None,
    ) -> tuple[Sequence[Text], list[float]]:
        policies = policies or []
        document_ids = list(set(sum([POLICY_IDS[p] for p in policies], [])))
        # Support offline retrieval
        if self.texts:
            self._embeddings_matrix = np.array([t.embedding for t in self.texts])
            k = min(k, len(self.texts))
            if k == 0:
                return [], []

            np_query = np.array((await embedding_model.embed_documents([query]))[0])

            similarity_scores = cosine_similarity(
                np_query.reshape(1, -1), self._embeddings_matrix
            )[0]
            similarity_scores = np.nan_to_num(similarity_scores, nan=-np.inf)
        # Support Supabase retrieval
        else:
            response_data = await self.pgvector_search(
                query,
                embedding_model,
                match_threshold=0,
                document_ids=document_ids,
            )
            texts = response_to_text(response_data)
            similarity_scores = np.array([r["similarity"] for r in response_data])
        # minus so descending
        # we could use arg-partition here
        # but a lot of algorithms expect a sorted list
        sorted_indices = np.argsort(-similarity_scores)
        return (
            [texts[i] for i in sorted_indices[:k]],
            [similarity_scores[i] for i in sorted_indices[:k]],
        )

    # Modification of MMRS with added policy filter
    async def max_marginal_relevance_search(
        self,
        query: str,
        k: int,
        fetch_k: int,
        embedding_model: Any,
        policies: Optional[List[str]] = None,
    ) -> tuple[Sequence[Text], list[float]]:
        """Vectorized implementation of Maximal Marginal Relevance (MMR) search.

        Args:
            query: Query vector.
            k: Number of results to return.
            fetch_k: Number of results to fetch from the vector store.
            embedding_model: model used to embed the query

        Returns:
            List of tuples (doc, score) of length k.
        """
        if fetch_k < k:
            raise ValueError("fetch_k must be greater or equal to k")

        texts, scores = await self.similarity_search(
            query, fetch_k, embedding_model, policies
        )
        if len(texts) <= k or self.mmr_lambda >= 1.0:
            return texts, scores

        embeddings = np.array([t.embedding for t in texts])
        np_scores = np.array(scores)
        similarity_matrix = cosine_similarity(embeddings, embeddings)

        selected_indices = [0]
        remaining_indices = list(range(1, len(texts)))

        while len(selected_indices) < k:
            selected_similarities = similarity_matrix[:, selected_indices]
            max_sim_to_selected = selected_similarities.max(axis=1)

            mmr_scores = (
                self.mmr_lambda * np_scores
                - (1 - self.mmr_lambda) * max_sim_to_selected
            )
            mmr_scores[selected_indices] = -np.inf  # Exclude already selected documents

            max_mmr_index = mmr_scores.argmax()
            selected_indices.append(max_mmr_index)
            remaining_indices.remove(max_mmr_index)

        return [texts[i] for i in selected_indices], [
            scores[i] for i in selected_indices
        ]

    async def get_existing_dockeys(self) -> List[str]:
        supabase = await create_async_client(self.supabase_url, self.supabase_key)
        
        start = 0
        batchsize = 1000
        data = []
        while True:
            response = (
                await supabase.table("documents")
                .select("id")
                .range(start, start + batchsize - 1)
                .execute()
            )
            data += response.data
            if len(response.data) < batchsize:
                break
            
        return [r["id"] for r in data]

    async def upload(
        self,
        doc: Doc,
        ignore_duplicate_doc=False,
    ):
        supabase = await create_async_client(self.supabase_url, self.supabase_key)

        # Upload document to `documents` table
        try:
            response = (
                await supabase.table("documents")
                .insert(
                    {
                        "id": doc.dockey,
                        "title": doc.title,
                        "abstract": doc.abstract,
                        "abstract_emb": doc.embedding,
                        "citation": doc.citation,
                        "authors": doc.authors,
                        "published_at": str(doc.published_at)
                        if doc.published_at
                        else None,
                        "filepath": doc.filepath,
                    }
                )
                .execute()
            )
            if not len(response.data):
                raise ValueError("Document not inserted")

        except APIError as e:
            if e.message.startswith("duplicate key"):
                if ignore_duplicate_doc:
                    return
                else:
                    raise ValueError(
                        "Another document with the same citation has already been uploaded previously"
                    ) from e
            else:
                raise e

        await gather_with_concurrency(
            n=4,
            coros=[upload_chunk(t, supabase) for t in doc.texts],
        )
