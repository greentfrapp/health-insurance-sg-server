"""
Suggest follow-up responses for the user
"""

from typing import cast

from dirtyjson.attributed_containers import AttributedList
from llama_index.core.agent.react.step import add_user_step_to_reasoning

from ...llms.llm_result import llm_parse_json
from ...utils.policies import VALID_POLICIES
from .step import PaperQAAgentWorker

ALTERNATIVE_POLICY_QUESTIONS = [
    f"- How does this compare to {p}" for p in VALID_POLICIES
]


SUGGEST_FOLLOW_UP_PROMPT = f"""
Suggest up to 2 follow-up responses that the user can respond with.

Only choose from the following options:
- A general relevant question no more than 10 words
- "Format your response as a table"
- "Simplify your response"
- "How does this compare to <another policy>"

Where <another policy> is one of:
{VALID_POLICIES}

Do not suggest responses similar to the user's last 5 responses.

Format your answer as a JSON array like this:

[
    "<suggestion 1>",
    ...
]

If no follow-up responses are appropriate, return an empty list like this: []
"""


async def suggest_follow_up(agent):
    worker = cast(PaperQAAgentWorker, agent.agent_worker)
    task = agent.create_task(SUGGEST_FOLLOW_UP_PROMPT)
    step_queue = agent.state.get_step_queue(task.task_id)
    step = step_queue.popleft()

    if step.input is not None:
        add_user_step_to_reasoning(
            step,
            task.extra_state["new_memory"],
            task.extra_state["current_reasoning"],
            verbose=worker._verbose,
        )

    tools = worker.get_tools(task.input)

    input_chat = worker._react_chat_formatter.format(
        tools,
        chat_history=task.memory.get(input=task.input)
        + task.extra_state["new_memory"].get_all(),
        current_reasoning=task.extra_state["current_reasoning"],
    )

    chat_stream = worker._llm.stream_chat(input_chat)

    response_buffer = ""
    for chunk in chat_stream:
        value = chunk.message.content[len(response_buffer) :]
        response_buffer += value

    suggestions = llm_parse_json(response_buffer)
    if type(suggestions) is not AttributedList:
        return []
    else:
        return suggestions[:2]
