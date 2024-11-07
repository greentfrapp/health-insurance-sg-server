from collections import OrderedDict
import re

from llamaqa.tools.generate_response import EXAMPLE_CITATION
from llamaqa.utils.utils import name_pos_in_text
from llamaqa.utils.answer import Answer
from llamaqa.utils.inner_context import InnerContext


def format_response(query: str, response: str, context: InnerContext):
    answer_text = response
    # Format references
    bib_positions = []
    if EXAMPLE_CITATION in answer_text:
        answer_text = answer_text.replace(EXAMPLE_CITATION, "")
    for c in context.filtered_contexts:
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
        citation = bib_pos["bib"]
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
    docnames = "|".join(set(b.text.doc.docname for b in response.bib.values()))
    citation_group_pattern = re.compile(f"\\(({docnames}) pages \\d+-\\d+,?( quote\\d+(, quote\\d+)*)?((,|;) ({docnames}) pages \\d+-\\d+,?( quote\\d+((,|;) quote\\d+)*)?)*\\)")
    citation_single_pattern = re.compile(f"((?P<citation>({docnames}) pages \\d+-\\d+),?(?P<quotes> quote\\d+((,|;) quote\\d+)*)?)((,|;) )?")

    def replace_individual_citations(match: re.Match):
        quotes_text = match.groupdict()["quotes"]
        if quotes_text:
            quotes_formatted = re.sub("(?P<q>quote\\d+)(, )?", lambda q: f"<quote>{q.groupdict()['q']}</quote>", quotes_text)
        else:
            quotes_formatted = ""
        return f"<doc>{match.groupdict()['citation'].strip()}{quotes_formatted}</doc>"

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
    return {
        "question": response.question,
        "text": response.answer,
        "references": [
            {
                "id": b.text.name,
                "filepath": b.text.doc.filepath,
                "value": b.text.doc.citation,
                "pages": b.text.pages,
                "quotes": b.points,
            } for b in response.bib.values()
        ]
    }
