from datetime import datetime
from uuid import UUID, uuid4
import contextvars
import json
import logging
import re

from pydantic import (
    BaseModel,
    Field,
    computed_field,
)
import litellm


logger = logging.getLogger(__name__)


cvar_answer_id = contextvars.ContextVar[UUID | None]("answer_id", default=None)


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
