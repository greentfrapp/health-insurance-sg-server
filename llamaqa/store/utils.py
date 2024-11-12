from supabase._async.client import AsyncClient

from ..reader.doc import Text


async def upload_chunk(chunk: Text, supabase: AsyncClient):
    response = (
        await supabase.table("chunks")
        .insert(
            {
                "document": chunk.doc.dockey,
                "pages": chunk.pages,
                "text": chunk.text,
                "text_emb": chunk.embedding,
                "summary": chunk.summary,
                "points": [p.dict() for p in chunk.points],
            }
        )
        .execute()
    )
    if not len(response.data):
        raise ValueError("Chunk not inserted")
