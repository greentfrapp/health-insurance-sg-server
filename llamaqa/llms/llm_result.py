import contextvars
import logging
import re
from datetime import datetime
from uuid import UUID, uuid4

import dirtyjson
import litellm
from pydantic import (
    BaseModel,
    Field,
    computed_field,
)

logger = logging.getLogger(__name__)


cvar_answer_id = contextvars.ContextVar[UUID | None]("answer_id", default=None)


def find_json(rgx: str, text: str):
    match = re.search(rgx, text)
    if match is None:
        return None
    else:
        return match.groupdict().get("json")


def llm_parse_json(text: str) -> dict:
    """Read LLM output and extract JSON data from it."""

    # First check for ```json
    code_snippet_pattern = r"```json(?P<json>(.|\s|\n)*?)```"
    code_snippet_result = find_json(code_snippet_pattern, text)
    # Then try to find the longer match between [.*?] and {.*?}
    array_pattern = r"(?P<json>\[(.|\s|\n)*\])"
    array_result = find_json(array_pattern, text)
    dict_pattern = r"(?P<json>{(.|\s|\n)*})"
    dict_result = find_json(dict_pattern, text)

    if array_result and dict_result and len(dict_result) > len(array_result):
        results = [
            code_snippet_result,
            dict_result,
            array_result,
        ]
    else:
        results = [
            code_snippet_result,
            array_result,
            dict_result,
        ]

    # Try each result in order
    for result in results:
        if result is not None:
            try:
                return dirtyjson.loads(result)
            except dirtyjson.error.Error:
                continue

    error_message = (
        f"Failed to parse JSON from text {text!r}. Your model may not be capable of"
        " supporting JSON output or our parsing technique could use some work. Try"
        " a different model"
    )
    raise ValueError(error_message)


class LLMResult(BaseModel):
    """A class to hold the result of a LLM completion.

    To associate a group of LLMResults, you can use the `set_llm_answer_ids` context manager:

    ```python
    my_answer_id = uuid4()
    with set_llm_answer_ids(my_answer_id):
        # code that generates LLMResults
        pass
    ```

    and all the LLMResults generated within the context will have the same `answer_id`.
    This can be combined with LLMModels `llm_result_callback` to store all LLMResults.
    """

    id: UUID = Field(default_factory=uuid4)
    answer_id: UUID | None = Field(
        default_factory=cvar_answer_id.get,
        description="A persistent ID to associate a group of LLMResults",
    )
    name: str | None = None
    prompt: str | list[dict] | None = Field(
        default=None,
        description="Optional prompt (str) or list of serialized prompts (list[dict]).",
    )
    text: str = ""
    prompt_count: int = 0
    completion_count: int = 0
    model: str
    date: str = Field(default_factory=datetime.now().isoformat)
    seconds_to_first_token: float = Field(
        default=0.0, description="Delta time (sec) to first response token's arrival."
    )
    seconds_to_last_token: float = Field(
        default=0.0, description="Delta time (sec) to last response token's arrival."
    )

    def __str__(self) -> str:
        return self.text

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cost(self) -> float:
        """Return the cost of the result in dollars."""
        if self.prompt_count and self.completion_count:
            try:
                pc = litellm.model_cost[self.model]["input_cost_per_token"]
                oc = litellm.model_cost[self.model]["output_cost_per_token"]
                return pc * self.prompt_count + oc * self.completion_count
            except KeyError:
                logger.warning(f"Could not find cost for model {self.model}.")
        return 0.0

    def to_json(self):
        return llm_parse_json(self.text)
