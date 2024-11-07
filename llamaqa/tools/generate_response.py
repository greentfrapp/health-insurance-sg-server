from llamaqa.llms.llm_model import LLMModel
from llamaqa.store.store import VectorStore
from llamaqa.utils.inner_context import InnerContext
from llamaqa.utils.response import Response
from llamaqa.utils.utils import name_in_text


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
    '"I cannot answer."'
    "For each part of your answer, indicate which sources and quotes most support "
    "it via citation keys at the end of sentences, "
    "like {example_citation} or {example_citation_quote}. Only cite from the context "
    "above and only use the valid keys or quotes. "
    "Do not repeat any quote in your answer. "
    "Write in a style accessible to the layperson but keep your "
    "wording and content accurate without any misrepresentation. "
    "The context comes from a variety of sources and is only a summary, "
    "so there may inaccuracies or ambiguities. Do not add any extraneous information. "
    "\n\n"
    "Answer ({answer_length}, please split into paragraphs of about 50 to 60 words each):"
)

EXAMPLE_CITATION: str = "(Example2012Example pages 3-4)"
EXAMPLE_CITATION_QUOTE: str = "(Example2012Example pages 3-4 quote1, quote2, Example2012Example pages 10-13 quote1)"


async def generate_response(
    context: InnerContext,
    store: VectorStore,
    query: str,
    llm_model: LLMModel,
):
    context_str = context.get_string()
    answer_result = await llm_model.run_prompt(
        prompt=QA_PROMPT,
        data={
            "context": context_str,
            "answer_length": "about 200 words, but can be longer",
            "question": query,
            "example_citation": EXAMPLE_CITATION,
            "example_citation_quote": EXAMPLE_CITATION_QUOTE,
        },
        name="answer",
        system_prompt=SYSTEM_PROMPT,
    )
    answer_text = answer_result.text

    # Format references
    bib = {}
    if EXAMPLE_CITATION in answer_text:
        answer_text = answer_text.replace(EXAMPLE_CITATION, "")
    for c in context.filtered_contexts:
        # do check for whole key (so we don't catch Callahan2019a with Callahan2019)
        if name_in_text(c.text.name, answer_text):
            bib[c.text.name] = c
    bib_str = "\n\n".join(
        [f"{i+1}. ({k}): {c.text.doc.citation}" for i, (k, c) in enumerate(bib.items())]
    )

    # formatted_answer = f"Question: {query}\n\n{answer_text}\n"
    formatted_answer = f"{answer_text}\n"
    if bib:
        formatted_answer += f"\nReferences\n\n{bib_str}\n"

    response = Response(
        question=query,
        answer=answer_result.text,
        references=bib_str,
        formatted_answer=formatted_answer
    )
    return response


def retrieve_context(
    context: InnerContext,
    store: VectorStore,
    query: str,
):
    context_str = context.get_string()
    return QA_PROMPT.format(**{
        "context": context_str,
        "answer_length": "about 200 words, but can be longer",
        "question": query,
        "example_citation": EXAMPLE_CITATION,
        "example_citation_quote": EXAMPLE_CITATION_QUOTE,
    })
