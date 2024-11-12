from collections import OrderedDict
from typing import cast
import re

from llama_index.core.callbacks import (
    CallbackManager,
    CBEventType,
    EventPayload,
)
from llama_index.core.tools import ToolOutput

from ...tools.paperqa_tools import PaperQAToolSpec
from ...tools.retrieve_evidence import (
    EXAMPLE_CITATION,
)
from ...utils.answer import Answer
from ...utils.cache import Cache
from ...utils.context import Context


def infer_stream_chunk_is_final(
    chunk: str, missed_chunks_storage: list
) -> bool:
    """Infers if a chunk from a live stream is the start of the final
    reasoning step. (i.e., and should eventually become
    ResponseReasoningStep — not part of this function's logic tho.).

    Args:
        chunk (ChatResponse): the current chunk stream to check
        missed_chunks_storage (list): list to store missed chunks

    Returns:
        bool: Boolean on whether the chunk is the start of the final response
    """
    latest_content = chunk
    if latest_content:
        # doesn't follow thought-action format
        # keep first chunks
        if len(latest_content) < len("Thought"):
            missed_chunks_storage.append(chunk)
        elif not latest_content.startswith("Thought") and "\nThought:" not in latest_content:
            return True
        elif "Answer:" in latest_content:
            missed_chunks_storage.clear()
            return True
    return False


def name_pos_in_text(name: str, text: str) -> int:
    sname = name.strip()
    pattern = rf"\b({re.escape(sname)})\b(?!\w)"
    match = re.search(pattern, text)
    if match:
        return match.start()
    else:
        return -1


def format_response(query: str, response: str, toolspec: PaperQAToolSpec):
    answer_text = response
    # Format references
    bib_positions = []
    if EXAMPLE_CITATION in answer_text:
        answer_text = answer_text.replace(EXAMPLE_CITATION, "")
    for c in toolspec.cache.filtered_contexts:
        # do check for whole key (so we don't catch Callahan2019a with Callahan2019)
        position = name_pos_in_text(c.text.name, answer_text)
        if position >= 0:
            bib_positions.append({
                "bib": c,
                "pos": position,
            })
    bib_positions.sort(key=lambda x: x["pos"])
    bib = OrderedDict()
    for bib_pos in bib_positions:
        citation = cast(Context, bib_pos["bib"])
        bib[citation.text.name] = citation
    bib_str = "\n\n".join(
        [f"{i+1}. ({k}): {c.text.doc.citation}" for i, (k, c) in enumerate(bib.items())]
    )
    formatted_answer = f"Question: {query}\n\n{answer_text}\n"
    if bib:
        formatted_answer += f"\nReferences\n\n{bib_str}\n"
    response = Answer(
        question=query,
        answer=answer_text,
        references=bib_str,
        formatted_answer=formatted_answer,
        bib=bib,
    )

    # Convert citations into <cite> tags
    docnames = set(b.text.doc.docname for b in response.bib.values())
    docnames_str = "|".join(docnames)
    print(docnames_str)
    text_names = set(response.bib.keys())
    citation_group_pattern = re.compile(f"\\(({docnames_str}) pages \\d+-\\d+,?( quote\\d+(, quote\\d+)*)?((,|;) ({docnames_str}) pages \\d+-\\d+,?( quote\\d+((,|;) quote\\d+)*)?)*\\)")
    citation_single_pattern = re.compile(f"((?P<citation>({docnames_str}) pages \\d+-\\d+),?(?P<quotes> quote\\d+((,|;) quote\\d+)*)?)((,|;) )?")

    references_list = []

    def create_quote_tag(match: re.Match, text_name: str):
        references_list.append(f"{text_name} {match.groupdict()['q']}")
        return f"<doc>{text_name} {match.groupdict()['q']}</doc>"

    def replace_individual_citations(match: re.Match):
        quotes_text = match.groupdict()["quotes"]
        text_name = match.groupdict()['citation'].strip()
        if text_name not in text_names:
            return ""
        if quotes_text:
            return re.sub("(?P<q>quote\\d+)(, )?", lambda m: create_quote_tag(m, text_name), quotes_text)
        else:
            references_list.append(text_name)
            return f"<doc>{text_name}</doc>"

    def replace_with_tag(match: re.Match):
        text = match.group().strip("(").strip(")")
        new_text = re.sub(citation_single_pattern, replace_individual_citations, text)
        return f"<cite>{new_text}</cite>"

    response.answer = re.sub(citation_group_pattern, replace_with_tag, response.answer.strip())

    period_citation_pattern = re.compile(f"\\.\\s*?(?P<citation><cite>.*?</cite>)")
    def move_period_mark(match: re.Match):
        return f"{match.groupdict()['citation']}."
    
    response.answer = re.sub(period_citation_pattern, move_period_mark, response.answer)
    response.answer = re.sub(re.compile("\\.+"), ".", response.answer)

    # Format response
    references = []
    for r in references_list:
        docname = r.split(" quote")[0]
        context = cast(Context, response.bib[docname])
        quote = None
        if " quote" in r:
            quote = context.points[int(r.split(" quote")[1]) - 1]["quote"]
        references.append({
            "id": r,
            "filepath": context.text.doc.filepath,
            "citation": context.text.doc.citation,
            "pages": context.text.pages,
            "quote": quote,
        })
    return {
        "question": response.question,
        "text": response.answer,
        "references": references
    }


def tell_llm_about_failure_in_extract_reasoning_step(
    callback_manager: CallbackManager, _: ValueError
) -> ToolOutput:
    """
    If the developer has instructed to tell the Agent a complaint about its non-cooperation,
    we will emit a Tool Output that we prepared (at initialization time) to the LLM, so that
    the LLM can be more cooperative in its next generation.
    """
    message = """Error: Could not parse output. Please follow the thought-action-input format. Try again.
Maybe you should try calling the gather_evidence tool.
Remember that the format should be
```
Thought: I need to use a tool to help me answer the question.
Action: tool name if using a tool.
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
```
"""
    dummy_tool_output = ToolOutput(
        content=message,
        tool_name="unknown",
        raw_input={},
        raw_output=message,
    )
    with callback_manager.event(
        CBEventType.FUNCTION_CALL,
        payload={
            EventPayload.FUNCTION_CALL: "unknown",
        },
    ) as event:
        event.on_end(payload={EventPayload.FUNCTION_OUTPUT: str(dummy_tool_output)})

    return dummy_tool_output