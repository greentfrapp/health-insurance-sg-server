"""PaperQA agent worker"""

import asyncio
import json
import uuid
from functools import partial
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    cast,
    Callable,
)

from llama_index.core.agent.react.formatter import ReActChatFormatter
from llama_index.core.agent.react.output_parser import ReActOutputParser
from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    BaseReasoningStep,
    ObservationReasoningStep,
    ResponseReasoningStep,
)
from llama_index.core.agent.types import (
    BaseAgentWorker,
    Task,
    TaskStep,
    TaskStepOutput,
)
from llama_index.core.base.llms.types import MessageRole
from llama_index.core.callbacks import (
    CallbackManager,
    CBEventType,
    EventPayload,
    trace_method,
)
from llama_index.core.chat_engine.types import (
    AGENT_CHAT_RESPONSE_TYPE,
    AgentChatResponse,
    StreamingAgentChatResponse,
)
from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.agent import AgentToolCallEvent
from llama_index.core.llms.llm import LLM
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.memory.types import BaseMemory
from llama_index.core.objects.base import ObjectRetriever
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.settings import Settings
from llama_index.core.tools import BaseTool, ToolOutput, adapt_to_async_tool
from llama_index.core.tools.types import AsyncBaseTool
from llama_index.core.types import Thread
from llama_index.core.utils import print_text


dispatcher = get_dispatcher(__name__)

from llama_index.core.agent.react.step import (
    ReActAgentWorker,
    add_user_step_to_reasoning,
)

class PaperQAAgentWorker(ReActAgentWorker):
    def _get_task_step_response(
        self, agent_response: AGENT_CHAT_RESPONSE_TYPE, step: TaskStep, is_done: bool
    ) -> TaskStepOutput:
        """Get task step response."""
        if is_done:
            new_steps = []
        else:
            new_steps = [
                step.get_next_step(
                    step_id=str(uuid.uuid4()),
                    # NOTE: input is unused
                    input=None,
                )
            ]

        return TaskStepOutput(
            output=agent_response,
            task_step=step,
            is_last=is_done,
            next_steps=new_steps,
        )

    def _extract_reasoning_step(
        self, output: ChatResponse|str, is_streaming: bool = False
    ) -> Tuple[str, List[BaseReasoningStep], bool]:
        """
        Extracts the reasoning step from the given output.

        This method parses the message content from the output,
        extracts the reasoning step, and determines whether the processing is
        complete. It also performs validation checks on the output and
        handles possible errors.
        """
        if isinstance(output, ChatResponse):
            if output.message.content is None:
                raise ValueError("Got empty message.")
            message_content = output.message.content
        else:
            message_content = output

        current_reasoning = []
        try:
            reasoning_step = self._output_parser.parse(message_content, is_streaming)
        except BaseException as exc:
            raise ValueError(f"Could not parse output: {message_content}") from exc
        if self._verbose:
            print_text(f"{reasoning_step.get_content()}\n", color="pink")
        current_reasoning.append(reasoning_step)

        if reasoning_step.is_done:
            return message_content, current_reasoning, True

        reasoning_step = cast(ActionReasoningStep, reasoning_step)
        if not isinstance(reasoning_step, ActionReasoningStep):
            raise ValueError(f"Expected ActionReasoningStep, got {reasoning_step}")

        return message_content, current_reasoning, False

    def _process_actions(
        self,
        task: Task,
        tools: Sequence[AsyncBaseTool],
        output: ChatResponse,
        is_streaming: bool = False,
    ) -> Tuple[List[BaseReasoningStep], bool]:
        tools_dict: Dict[str, AsyncBaseTool] = {
            tool.metadata.get_name(): tool for tool in tools
        }
        tool = None

        try:
            _, current_reasoning, is_done = self._extract_reasoning_step(
                output, is_streaming
            )
        except ValueError as exp:
            current_reasoning = []
            tool_output = self._handle_reasoning_failure_fn(self.callback_manager, exp)
        else:
            if is_done:
                return current_reasoning, True

            # call tool with input
            reasoning_step = cast(ActionReasoningStep, current_reasoning[-1])
            if reasoning_step.action in tools_dict:
                tool = tools_dict[reasoning_step.action]
                with self.callback_manager.event(
                    CBEventType.FUNCTION_CALL,
                    payload={
                        EventPayload.FUNCTION_CALL: reasoning_step.action_input,
                        EventPayload.TOOL: tool.metadata,
                    },
                ) as event:
                    try:
                        dispatcher.event(
                            AgentToolCallEvent(
                                arguments=json.dumps({**reasoning_step.action_input}),
                                tool=tool.metadata,
                            )
                        )
                        tool_output = tool.call(**reasoning_step.action_input)
                    except Exception as e:
                        tool_output = ToolOutput(
                            content=f"Error: {e!s}",
                            tool_name=tool.metadata.name,
                            raw_input={"kwargs": reasoning_step.action_input},
                            raw_output=e,
                            is_error=True,
                        )
                    event.on_end(
                        payload={EventPayload.FUNCTION_OUTPUT: str(tool_output)}
                    )
            else:
                tool_output = self._handle_nonexistent_tool_name(reasoning_step)

        task.extra_state["sources"].append(tool_output)

        observation_step = ObservationReasoningStep(
            observation=str(tool_output),
            return_direct=(
                tool.metadata.return_direct and not tool_output.is_error
                if tool
                else False
            ),
        )
        current_reasoning.append(observation_step)
        if self._verbose:
            print_text(f"{observation_step.get_content()}\n", color="blue")
        return (
            current_reasoning,
            tool.metadata.return_direct and not tool_output.is_error if tool else False,
        )

    # TODO: Update this to also break out for non-answers
    def _run_step_stream(
        self,
        step: TaskStep,
        task: Task,
    ) -> TaskStepOutput:
        """Run step."""
        if step.input is not None:
            add_user_step_to_reasoning(
                step,
                task.extra_state["new_memory"],
                task.extra_state["current_reasoning"],
                verbose=self._verbose,
            )
        # TODO: see if we want to do step-based inputs
        tools = self.get_tools(task.input)

        input_chat = self._react_chat_formatter.format(
            tools,
            chat_history=task.memory.get(input=task.input)
            + task.extra_state["new_memory"].get_all(),
            current_reasoning=task.extra_state["current_reasoning"],
        )

        chat_stream = self._llm.stream_chat(input_chat)

        # iterate over stream, break out if is final answer after the "Answer: "
        full_response = ChatResponse(
            message=ChatMessage(content=None, role="assistant")
        )
        missed_chunks_storage: List[ChatResponse] = []
        is_done = False

        """START NEW CODE"""
        # for latest_chunk in chat_stream:
        #     print(latest_chunk)
        #     full_response = latest_chunk
        #     is_done = self._infer_stream_chunk_is_final(
        #         latest_chunk, missed_chunks_storage
        #     )
        #     if is_done:
        #         break
        # quit()
        is_done = True
        latest_chunk = chat_stream.__next__()
        non_streaming_agent_response = None

        start_idx = 0
        # if start_idx != -1 and latest_chunk.message.content:
        #     latest_chunk.message.content = latest_chunk.message.content[
        #         start_idx + len("Answer:") :
        #     ].strip()
        # set delta to the content, minus the "Answer: "
        latest_chunk.delta = latest_chunk.message.content

        # add back the chunks that were missed
        response_stream = self._add_back_chunk_to_stream(
            chunks=[*missed_chunks_storage, latest_chunk], chat_stream=chat_stream
        )

        # print("Stream: " + chat_stream.__next__().message.content)

        # Get the response in a separate thread so we can yield the response
        agent_response_stream = StreamingAgentChatResponse(
            chat_stream=response_stream,
            sources=task.extra_state["sources"],
        )
        thread = Thread(
            target=agent_response_stream.write_response_to_history,
            args=(task.extra_state["new_memory"],),
            kwargs={"on_stream_end_fn": partial(self.finalize_task, task)},
        )
        thread.start()

        response = agent_response_stream or non_streaming_agent_response
        assert response is not None
        return response
        output = self._get_task_step_response(response, step, is_done)
        # output.is_last = False
        return output
        """END NEW CODE"""


        for latest_chunk in chat_stream:
            full_response = latest_chunk
            is_done = self._infer_stream_chunk_is_final(
                latest_chunk, missed_chunks_storage
            )
            if is_done:
                break

        non_streaming_agent_response = None
        agent_response_stream = None
        if not is_done:
            # given react prompt outputs, call tools or return response
            reasoning_steps, is_done = self._process_actions(
                task, tools=tools, output=full_response, is_streaming=True
            )
            task.extra_state["current_reasoning"].extend(reasoning_steps)
            # use _get_response to return intermediate response
            non_streaming_agent_response = self._get_response(
                task.extra_state["current_reasoning"], task.extra_state["sources"]
            )
            if is_done:
                non_streaming_agent_response.is_dummy_stream = True
                task.extra_state["new_memory"].put(
                    ChatMessage(
                        content=non_streaming_agent_response.response,
                        role=MessageRole.ASSISTANT,
                    )
                )
        else:
            # remove "Answer: " from the response, and anything before it
            start_idx = (latest_chunk.message.content or "").find("Answer:")
            if start_idx != -1 and latest_chunk.message.content:
                latest_chunk.message.content = latest_chunk.message.content[
                    start_idx + len("Answer:") :
                ].strip()

            # set delta to the content, minus the "Answer: "
            latest_chunk.delta = latest_chunk.message.content

            # add back the chunks that were missed
            response_stream = self._add_back_chunk_to_stream(
                chunks=[*missed_chunks_storage, latest_chunk], chat_stream=chat_stream
            )

            # Get the response in a separate thread so we can yield the response
            agent_response_stream = StreamingAgentChatResponse(
                chat_stream=response_stream,
                sources=task.extra_state["sources"],
            )
            thread = Thread(
                target=agent_response_stream.write_response_to_history,
                args=(task.extra_state["new_memory"],),
                kwargs={"on_stream_end_fn": partial(self.finalize_task, task)},
            )
            thread.start()

        response = agent_response_stream or non_streaming_agent_response
        assert response is not None

        return self._get_task_step_response(response, step, is_done)