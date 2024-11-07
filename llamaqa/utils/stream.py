from typing import cast
import json
import uuid

from llama_index.core.agent.react.types import ActionReasoningStep
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llamaqa.utils.inner_context import InnerContext
from llamaqa.agents.paperqa import PaperQAAgent
from llamaqa.utils.api import format_response


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


def stream_thoughts(agent: PaperQAAgent, context: InnerContext, query: str):
    worker = agent.agent_worker
    task = agent.create_task(f"{query}\nRemember to call gather_evidence if the user is asking about insurance, especially if you are citing anything.")
    
    tool_descriptors = {
        "gather_evidence": "Gathering evidence on \"{query}\"",
        "retrieve_evidence": "Retrieving gathered evidence",
    }

    iters = 0
    max_iters = 10
    step_queue = agent.state.get_step_queue(task.task_id)
    step = step_queue.popleft()
    while True:
        iters += 1

        response = worker._run_step_stream(
            step=step,
            task=task,
        )

        for chunk in response.response_gen:
            yield(chunk)
            is_done = infer_stream_chunk_is_final(
                response.unformatted_response, []
            )
        full_response = response.response
        
        if not is_done:
            tools = worker.get_tools(task.input)
            tools_dict = {
                tool.metadata.get_name(): tool for tool in tools
            }
            
            # Extract tool to yield
            try:
                _, current_reasoning, is_done = worker._extract_reasoning_step(
                    full_response, is_streaming=True
                )
                reasoning_step = cast(ActionReasoningStep, current_reasoning[-1])
                if reasoning_step.action in tools_dict:
                    yield(f"Action Desc: {tool_descriptors[reasoning_step.action].format(**reasoning_step.action_input)}...")
            except ValueError:
                pass

            # given react prompt outputs, call tools or return response
            reasoning_steps, is_done = worker._process_actions(
                task, tools=tools, output=full_response, is_streaming=True
            )
            if reasoning_steps[-1].observation.startswith("Found"):
                yield("Action Output:" + reasoning_steps[-1].observation.split(".")[0] + "...")
            task.extra_state["current_reasoning"].extend(reasoning_steps)

            step = step.get_next_step(
                step_id=str(uuid.uuid4()),
                input=None,
            )
        if is_done:
            break
        if iters >= max_iters:
            break
    
    memory_dict = agent.memory.chat_store.to_dict()["store"]["chat_history"]
    memory_dict = [ChatMessage(**i) for i in memory_dict]
    for message in memory_dict[::-1]:
        if message.role == MessageRole.USER:
            break
    message.content = query
    agent.memory.set(memory_dict)
    
    yield("Final Response: " + json.dumps(format_response(
        query,
        full_response.split("Answer:")[-1].strip(),
        context,
    )))
