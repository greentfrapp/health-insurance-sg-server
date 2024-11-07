# PaperQA Notes

## What happens when `Docs().add` is run?

1. Get metadata
    a.Create a fake Doc instance then run paperqa.readers.read_doc, which parses the document and splits it into chunks
    b. Use LLM to generate citation from chunks via config.citation_prompt
        Generate docname
2. Run paperqa.readers.read_doc again using Doc instance with filled metadata
    Default parameters:
        chunk_chars = 3000
        overlap = 100
    Each chunk is stored as a paperqa.types.Text instance with docname and page number
3. Run paperqa.docs.aadd_texts
    a. Embed chunks
    b. Attach each embedding to each chunk's Text instance

Note: if running chunking ourselves, can use paperqa.docs.aadd_texts function
But this (probably) doesn't work at a million-doc scale, since we are effectively holding the docs in RAM.
Instead, the max marginal relevance search might need to be done at the database level 

Note: Default parser doesn't seem to parse double column formats correctly

## DB structure for storing papers

### Considerations

- A `chunks` table for storing chunks with their embeddings
- Possibly a `documents` table for storing papers with their embeddings
    - Then we search through `documents` table first to filter relevant papers before searching `chunks`
    - Will save search space by N times where N is the mean #chunks per paper (~20)

Ideally:
- Ignore references sections
- A fast way to run vector queries

### Tables

```sql
create table documents (
  id uuid primary key,
  title text not null,
  abstract text not null,
  abstract_emb vector(768),
  citation text,
  authors text,
  published_at timestamptz,
  created_at timestamptz default now()
);

create table chunks (
  id uuid primary key,
  document uuid references documents (id),
  pages int[],
  text text,
  text_emb vector(768),
  created_at timestamptz default now()
);
```

#### Citation Fidelity

For a simple prototype with a QA interface plus in-document references, we need a way to locate the reference.

Vanilla PaperQA cites references on a chunk-level, which can span several pages. Ideally, we want the citation to be on a sentence- or phrase-level.

One way is to have PaperQA generate the relevant quote and then map it to the closest fragment in the chunk and the PDF.

Alternatively, when generating the summary, have PaperQA generate in point form with each point mapping to a quote.

#### Conversational Interface

PaperQA does not support conversation features like chat history.

The initial prototype can have a QA interface. We can subsequently introduce a more chat-like interface. 

### Known Issues

- Response cannot cross-reference citations e.g. if response 2 repeats a citation from response 1, the citation will not be found in the Answer instance
    - Can be solved by making citations unique per response and then creating a universal citation list
- Agent may not interpret tables correctly, especially with merged cells
- When eventually supporting multiple users, need a way to persist conversation history while keeping server stateless
- Formats in bullet points and tables are possible but require post-processing on the front-end
- Provide all citations as a popup
- Agent often complains about not having sufficient information without first calling gather_evidence
- Support citations across multiple documents

- [x] Loading time is too long without any updates - should expose thought process
- [x] Fix citation order
- [x] Need to update conversation history to remove hidden prompts e.g. "Remember to call gather_evidence if you do not have enough information, especially if you are citing anything."
