from typing import (
    Any,
    Callable,
    List,
    Optional,
    Sequence,
    Union,
)

from llama_index.core.agent.react.formatter import ReActChatFormatter
from llama_index.core.agent.react.output_parser import ReActOutputParser
from llama_index.core.agent.react.step import ReActAgentWorker
from llama_index.core.agent.runner.base import AgentRunner
from llama_index.core.agent.types import (
    BaseAgent,
    BaseAgentWorker,
    Task,
    TaskStep,
    TaskStepOutput,
)
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.callbacks import (
    CallbackManager,
    CBEventType,
    EventPayload,
    trace_method,
)
from llama_index.core.chat_engine.types import (
    AGENT_CHAT_RESPONSE_TYPE,
    AgentChatResponse,
    ChatResponseMode,
    StreamingAgentChatResponse,
)
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.agent import (
    AgentRunStepEndEvent,
    AgentRunStepStartEvent,
    AgentChatWithStepStartEvent,
    AgentChatWithStepEndEvent,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.memory.types import BaseMemory
from llama_index.core.objects.base import ObjectRetriever
from llama_index.core.tools import BaseTool, ToolOutput

from llama_index.core.agent import ReActAgent

from .step import PaperQAAgentWorker


dispatcher = get_dispatcher(__name__)

class PaperQAAgent(ReActAgent):
    def __init__(
        self,
        tools: Sequence[BaseTool],
        llm: LLM,
        memory: BaseMemory,
        max_iterations: int = 10,
        react_chat_formatter: Optional[ReActChatFormatter] = None,
        output_parser: Optional[ReActOutputParser] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
        context: Optional[str] = None,
        handle_reasoning_failure_fn: Optional[
            Callable[[CallbackManager, Exception], ToolOutput]
        ] = None,
    ) -> None:
        """Init params."""
        callback_manager = callback_manager or llm.callback_manager
        if context and react_chat_formatter:
            raise ValueError("Cannot provide both context and react_chat_formatter")
        if context:
            react_chat_formatter = ReActChatFormatter.from_context(context)

        step_engine = PaperQAAgentWorker.from_tools(
            tools=tools,
            tool_retriever=tool_retriever,
            llm=llm,
            max_iterations=max_iterations,
            react_chat_formatter=react_chat_formatter,
            output_parser=output_parser,
            callback_manager=callback_manager,
            verbose=verbose,
            handle_reasoning_failure_fn=handle_reasoning_failure_fn,
        )
        AgentRunner.__init__(
            self,
            step_engine,
            memory=memory,
            llm=llm,
            callback_manager=callback_manager,
            verbose=verbose,
        )

    @dispatcher.span
    def _run_step(
        self,
        task_id: str,
        step: Optional[TaskStep] = None,
        input: Optional[str] = None,
        mode: ChatResponseMode = ChatResponseMode.WAIT,
        **kwargs: Any,
    ) -> TaskStepOutput:
        """Execute step."""
        task = self.state.get_task(task_id)
        step_queue = self.state.get_step_queue(task_id)
        step = step or step_queue.popleft()
        if input is not None:
            step.input = input

        dispatcher.event(
            AgentRunStepStartEvent(task_id=task_id, step=step, input=input)
        )

        if self.verbose:
            print(f"> Running step {step.step_id}. Step input: {step.input}")

        # TODO: figure out if you can dynamically swap in different step executors
        # not clear when you would do that by theoretically possible

        if mode == ChatResponseMode.WAIT:
            cur_step_output = self.agent_worker.run_step(step, task, **kwargs)
        elif mode == ChatResponseMode.STREAM:
            cur_step_output = self.agent_worker.stream_step(step, task, **kwargs)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        # append cur_step_output next steps to queue
        next_steps = cur_step_output.next_steps
        step_queue.extend(next_steps)

        # add cur_step_output to completed steps
        completed_steps = self.state.get_completed_steps(task_id)
        completed_steps.append(cur_step_output)

        dispatcher.event(AgentRunStepEndEvent(step_output=cur_step_output))
        return cur_step_output

    @dispatcher.span
    def _chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
        mode: ChatResponseMode = ChatResponseMode.WAIT,
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        """Chat with step executor."""
        if chat_history is not None:
            self.memory.set(chat_history)
        task = self.create_task(message)

        result_output = None
        dispatcher.event(AgentChatWithStepStartEvent(user_msg=message))
        while True:
            # TODO: This stage needs to start yielding thoughts and action outputs
            # pass step queue in as argument, assume step executor is stateless
            cur_step_output = self._run_step(
                task.task_id, mode=mode, tool_choice=tool_choice
            )

            if cur_step_output.is_last:
                result_output = cur_step_output
                break

            # ensure tool_choice does not cause endless loops
            tool_choice = "auto"

        result = self.finalize_response(
            task.task_id,
            result_output,
        )
        dispatcher.event(AgentChatWithStepEndEvent(response=result))
        return result

    @dispatcher.span
    @trace_method("chat")
    def stream_thoughts_and_chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Optional[Union[str, dict]] = None,
    ):
        if tool_choice is None:
            tool_choice = self.default_tool_choice
        with self.callback_manager.event(
            CBEventType.AGENT_STEP,
            payload={EventPayload.MESSAGES: [message]},
        ) as e:
            chat_response = self._chat(
                message, chat_history, tool_choice, mode=ChatResponseMode.STREAM
            )
            assert isinstance(chat_response, StreamingAgentChatResponse) or (
                isinstance(chat_response, AgentChatResponse)
                and chat_response.is_dummy_stream
            )
            e.on_end(payload={EventPayload.RESPONSE: chat_response})

        return chat_response  # type: ignore

