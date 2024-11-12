create or replace function match_chunks (
  query_embedding vector(768),
  match_threshold float,
  match_count int
)
returns table (
  id uuid,
  doc uuid,
  doc_title text,
  pages int[],
  text text,
  text_emb vector(768),
  similarity float
)
language sql stable
as $$
  select
    chunks.id,
    chunks.document as doc,
    documents.title as doc_title,
    chunks.pages,
    chunks.text,
    chunks.text_emb,
    1 - (chunks.text_emb <=> query_embedding) as similarity
  from chunks
  left join documents on chunks.document = documents.id
  where chunks.text_emb <=> query_embedding < 1 - match_threshold
  order by chunks.text_emb <=> query_embedding
  limit match_count;
$$;
