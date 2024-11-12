"""PaperQA agent worker"""

import json
import uuid
from typing import (
    Dict,
    List,
    Sequence,
    Tuple,
    cast,
)

from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    BaseReasoningStep,
    ObservationReasoningStep,
)
from llama_index.core.agent.types import (
    Task,
    TaskStep,
    TaskStepOutput,
)
from llama_index.core.callbacks import (
    CBEventType,
    EventPayload,
)
from llama_index.core.chat_engine.types import (
    AGENT_CHAT_RESPONSE_TYPE,
)
from llama_index.core.base.llms.types import ChatResponse
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.agent import AgentToolCallEvent
from llama_index.core.tools import ToolOutput
from llama_index.core.tools.types import AsyncBaseTool
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
        return chat_stream
