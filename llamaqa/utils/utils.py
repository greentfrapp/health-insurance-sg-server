from __future__ import annotations

import inspect
import json
import re
from collections.abc import Callable
from typing import Any

from llamaqa.llms.llm_result import LLMResult
from .context import Context
from .text import Text, TextPlus


def is_coroutine_callable(obj):
    if inspect.isfunction(obj):
        return inspect.iscoroutinefunction(obj)
    elif callable(obj):  # noqa: RET505
        return inspect.iscoroutinefunction(obj.__call__)
    return False


def extract_score(text: str) -> int:
    # check for N/A
    last_line = text.split("\n")[-1]
    if "N/A" in last_line or "n/a" in last_line or "NA" in last_line:
        return 0
    # check for not applicable, not relevant in summary
    if "not applicable" in text.lower() or "not relevant" in text.lower():
        return 0

    score = re.search(r"[sS]core[:is\s]+([0-9]+)", text)
    if not score:
        score = re.search(r"\(([0-9])\w*\/", text)
    if not score:
        score = re.search(r"([0-9]+)\w*\/", text)
    if score:
        s = int(score.group(1))
        if s > 10:  # noqa: PLR2004
            s = int(s / 10)  # sometimes becomes out of 100
        return s
    last_few = text[-15:]
    scores = re.findall(r"([0-9]+)", last_few)
    if scores:
        s = int(scores[-1])
        if s > 10:  # noqa: PLR2004
            s = int(s / 10)  # sometimes becomes out of 100
        return s
    if len(text) < 100:  # noqa: PLR2004
        return 1
    return 5


def strip_citations(text: str) -> str:
    # Combined regex for identifying citations (see unit tests for examples)
    citation_regex = r"\b[\w\-]+\set\sal\.\s\([0-9]{4}\)|\((?:[^\)]*?[a-zA-Z][^\)]*?[0-9]{4}[^\)]*?)\)"
    # Remove the citations from the text
    return re.sub(citation_regex, "", text, flags=re.MULTILINE)


def llm_parse_json(text: str) -> dict:
    """Read LLM output and extract JSON data from it."""
    # fetch from markdown ```json if present
    ptext = text.strip().split("```json")[-1].split("```")[0]
    # split anything before the first { after the last }
    ptext = ("{" + ptext.split("{", 1)[-1]).rsplit("}", 1)[0] + "}"

    def escape_newlines(match: re.Match) -> str:
        return match.group(0).replace("\n", "\\n")

    # Match anything between double quotes
    # including escaped quotes and other escaped characters.
    # https://regex101.com/r/VFcDmB/1
    pattern = r'"(?:[^"\\]|\\.)*"'
    ptext = re.sub(pattern, escape_newlines, ptext)
    try:
        return json.loads(ptext)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse JSON from text {text!r}. Your model may not be capable of"
            " supporting JSON output or our parsing technique could use some work. Try"
            " a different model or specify `Settings(prompts={'use_json': False})`"
        ) from e


async def map_fxn_summary(
    text: TextPlus,
    question: str,
    prompt_runner: Any | None,
    extra_prompt_data: dict[str, str] | None = None,
    parser: Callable[[str], dict[str, Any]] | None = None,
    callbacks: list[Callable[[str], None]] | None = None,
) -> tuple[Context, LLMResult]:
    """Parses the given text and returns a context object with the parser and prompt runner.

    The parser should at least return a dict with `summary`. A `relevant_score` will be used and any
    extra fields except `question` will be added to the context object. `question` is stripped
    because it can be incorrectly parsed from LLM outputs when parsing them as JSON.

    Args:
        text: The text to parse.
        question: The question to use for the chain.
        prompt_runner: The prompt runner to call - should have question, citation,
            summary_length, and text fields.
        extra_prompt_data: Optional extra kwargs to pass to the prompt runner's data.
        parser: The parser to use for parsing - return empty dict on Failure to fallback to text parsing.
        callbacks: LLM callbacks to execute in the prompt runner.

    Returns:
        The context object and LLMResult to get info about the LLM execution.
    """
    # needed empties for failures/skips
    llm_result = LLMResult(model="", date="")
    extras: dict[str, Any] = {}
    citation = text.name + ": " + text.doc.citation
    success = False

    if prompt_runner:
        llm_result = await prompt_runner(
            {"question": question, "citation": citation, "text": text.text}
            | (extra_prompt_data or {}),
            callbacks,
            "evidence:" + text.name,
        )
        context = llm_result.text
        result_data = parser(context) if parser else {}
        success = bool(result_data)
        if success:
            try:
                context = result_data.pop("summary")
                score = (
                    result_data.pop("relevance_score")
                    if "relevance_score" in result_data
                    else extract_score(context)
                )
                # just in case question was present
                result_data.pop("question", None)
                extras = result_data
            except KeyError:
                success = False
    else:
        context = text.text
        # If we don't assign scores, just default to 5.
        # why 5? Because we filter out 0s in another place
        # and 5/10 is the other default I could come up with
        score = 5
        success = True
    # remove citations that collide with our grounded citations (for the answer LLM)
    context = strip_citations(context)
    if not success:
        score = extract_score(context)

    return (
        Context(
            context=context,
            text=TextPlus.from_text(Text(
                text=text.text,
                name=text.name,
                doc=text.doc.__class__(**text.doc.model_dump(exclude={"embedding"})),
            )),
            score=score,  # pylint: disable=possibly-used-before-assignment
            **extras,
        ),
        llm_result,
    )


def name_in_text(name: str, text: str) -> bool:
    sname = name.strip()
    pattern = rf"\b({re.escape(sname)})\b(?!\w)"
    return bool(re.search(pattern, text))


def name_pos_in_text(name: str, text: str) -> int:
    sname = name.strip()
    pattern = rf"\b({re.escape(sname)})\b(?!\w)"
    match = re.search(pattern, text)
    if match:
        return match.start()
    else:
        return -1
