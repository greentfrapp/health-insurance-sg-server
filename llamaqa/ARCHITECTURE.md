# LlamaQA Architecture

## Overview

The aim of this repo is to implement a variant of PaperQA with the LlamaIndex framework.

While the PaperQA library works well out of the box, this re-implementation is meant to resolve the following issues:

- Better disentangling between components Agent, Tools, Store, Model, Context [P1]
    - Current PaperQA implementation of Agent, Store, and Context is confusing - for instance, "aget_evidence" is a method within the Docs class that calls MMRS and runs the RCS prompt. It feels like this should be contained within the "GatherEvidence" tool. Instead, the "GatherEvidence" tool runs the Docs.aget_evidence method. With the current implementation, significant changes have to be made when we move part of the vector search to the database server.
    - Current NamedTool base class is very minimal. Instead tools should support a set of default arguments that provides access to agent memory and chat context, possibly allowing tools to augment agent memory.
- Support for chat-like interface [P1]
    - Allow follow-up queries
    - Allow agent to clarify user's request
- Support for persistent inner context/memory [P2]
    - This is different from chat history, more like a scratchpad or our inner thoughts
    - In a regular conversation, consecutive questions are likely to be related. An inner context allows the agent to reference previous thoughts and contexts that might be relevant to new related queries. Preferring these over gathering new evidence from scratch can help save on query cost and also improve relevance.
- Support for tool ranking/preference [P3]
    - We might augment a tool with engineered preferences. For example, the agent should prefer referencing its inner context over referencing new papers.
- Support intermediate pausing and interjection [P3]
    - Allow user to view agent's current thought process and to interject in the middle of the agent's workflow
- Support for additional/better tools [P1]
    - Additional filters for "GatherEvidence" like page number, tags, and whether to retrieve all chunks (i.e. Summary Retriever in LlamaIndex)
    - Augment RCS step to support exact quotes
    - Allow agent to recursively decompose query to sub-queries, whose answers can then be used as context for the original request e.g. "Compare algo X and Y" can be broken down to "Summarize X" and "Summarize Y"

## General Architecture

Agent
Store
Memory
Tools
  | AsyncRunner(?)
  | StoreContext
  | GatherEvidence
  | GenerateResponse
  | SendMessage
Contexts
  | Observation
      | Quote
      | Point

## Roadmap

1. Reimplementation of PaperQA without frills
    - Store retrieves chunks from Supabase
    - Reproduce `Docs.query` function
2. Add support for decomposing queries into sub-queries
3. Support chat interface

## Sample Flows

User: "Compare X and Y"

Agent:
    [No previous context found, creating new context]
    [GatherEvidence(query="X and Y", level="document")]
        DocX1
    [GatherEvidence(document="DocX1", level="chunks", query="X and Y")]
        DocX1_C4
    [GenerateResponse()]
        "insufficient information..."
    [GatherEvidence(query="X", level="document")]
        DocX1, DocX2
    [GatherEvidence(document="DocX1", level="chunks", query="__ALL__")]
        DocX1_C1, ..., DocX1_C20
    [GatherEvidence(query="Y", level="document")]
        DocY1, DocY2, DocY3
    [GatherEvidence(document="DocY1", level="chunks", query="__ALL__")]
        DocY1_C1, ..., DocY1_C15
    [GenerateResponse()]
        "some reply"
    [Reply("some reply")]

User: "Which documents did you use?"

Agent:
    [Previous contexts ["00001 - Compare X and Y - 10 chunks found in 2 papers"]]
    []


