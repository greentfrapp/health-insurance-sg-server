from ..store.store import VectorStore
from ..utils.cache import Cache

SYSTEM_PROMPT = (
    "Answer in a direct and concise tone. "
    "Your audience is an expert, so be highly specific. "
    "If there are ambiguous terms or acronyms, first define them."
)

QA_PROMPT = (
    "Answer the question below with the context.\n\n"
    "Context (with relevance scores):\n\n{context}\n\n----\n\n"
    "Question: {question}\n\n"
    "Write an answer based on the context. "
    "If the context provides insufficient information reply "
    '"I cannot answer." '
    "For each part of your answer, indicate which sources and quotes most support "
    "it via citation keys at the end of sentences, "
    "like {example_citation} or {example_citation_quote}. Only cite from the context "
    "above and only use the valid keys or quotes. As much as possible, cite quotes. "
    "Do not repeat any quote verbatim in your answer. "
    "Write in a style accessible to the layperson but keep your "
    "wording and content accurate without any misrepresentation. "
    "The context comes from a variety of sources and is only a summary, "
    "so there may inaccuracies or ambiguities. Do not add any extraneous information. "
    "\n\n"
    "Answer ({answer_length}, please split into paragraphs of about 50 to 60 words each):"
)

EXAMPLE_CITATION: str = "(Example2012Example pages 3-4)"
EXAMPLE_CITATION_QUOTE: str = "(Example2012Example pages 3-4 quote1, quote2, Example2012Example pages 10-13 quote1)"


def retrieve_evidence(
    cache: Cache,
    store: VectorStore,
    query: str,
):
    context_str = cache.get_string()
    return QA_PROMPT.format(
        **{
            "context": context_str,
            "answer_length": "about 200 words, but can be longer",
            "question": query,
            "example_citation": EXAMPLE_CITATION,
            "example_citation_quote": EXAMPLE_CITATION_QUOTE,
        }
    )
