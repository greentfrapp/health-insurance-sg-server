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
Suggest 0 to 2 follow-up responses that can be presented to the user.
These responses are potential replies that the user can pose to you.

Choose relevant responses from the following options:
- A general relevant question no more than 10 words
- "How does this compare to <another policy>"
- "Format your response as a table" # Use this if your prior response involves comparisons or lists
- "Simplify your response" # Use this if your prior response might be too verbose
- "What are the different plans under this policy?"
- "What are the premiums for this policy?",
- "What are the limitations and exclusions?",
- "Are pre-existing conditions covered?",
- "Can I choose my preferred doctors and hospitals?",
- "How will making a claim affect my future premiums or policy?",
- "What is the claims process like?"

Where <another policy> is one of:
{VALID_POLICIES}

Do not suggest responses similar to the user's last 5 responses.

Only suggest relevant responses.
If no follow-up responses are appropriate, simply return an empty list.

Format your answer like this:

Thought: <thought process>

```json
[
    "<suggestion 1>",
    ...
]
```
"""


async def suggest_follow_up(agent):
    try:
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
    except:
        return []
