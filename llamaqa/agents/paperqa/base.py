import asyncio
import json
import logging
import os
import uuid
from typing import Callable, List, Optional, Sequence, cast

from litellm import completion_cost
from litellm.exceptions import APIConnectionError, ServiceUnavailableError
from litellm.types.utils import ModelResponse
from llama_index.core import PromptTemplate
from llama_index.core.agent import AgentRunner
from llama_index.core.agent.react import ReActAgent, ReActChatFormatter
from llama_index.core.agent.react.step import add_user_step_to_reasoning
from llama_index.core.agent.react.types import ActionReasoningStep
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms.llm import LLM
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.memory.types import BaseMemory
from llama_index.core.objects.base import ObjectRetriever
from llama_index.core.tools import BaseTool, ToolOutput
from llama_index.llms.litellm import LiteLLM

from ...llms import LiteLLMEmbeddingModel, LiteLLMModel
from ...store.supabase_store import SupabaseStore
from ...tools.paperqa_tools import PaperQAToolSpec
from ...utils.cache import Cache
from ...utils.logger import CostLogger
from ...utils.policies import VALID_POLICIES
from .fallback import FALLBACK_FINAL_RESPONSE, FALLBACK_RESPONSE_CONTENT
from .parser import PaperQAOutputParser
from .prompts import PAPERQA_SYSTEM_PROMPT
from .step import PaperQAAgentWorker
from .suggest import suggest_follow_up
from .utils import (
    format_response,
    infer_stream_chunk_is_final,
    parse_action_response,
    tell_llm_about_failure_in_extract_reasoning_step,
)

logger = logging.getLogger("paperqa-agent")


class PaperQAAgent(ReActAgent):
    toolspec: PaperQAToolSpec
    cost_logger: CostLogger

    def __init__(
        self,
        tools: Sequence[BaseTool],
        llm: LLM,
        memory: BaseMemory,
        max_iterations: int = 10,
        react_chat_formatter: Optional[ReActChatFormatter] = None,
        output_parser: Optional[PaperQAOutputParser] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
        context: Optional[str] = None,
        handle_reasoning_failure_fn: Optional[
            Callable[[CallbackManager, Exception], ToolOutput]
        ] = None,
        cost_logger: Optional[CostLogger] = None,
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
        self.update_prompts(
            {"agent_worker:system_prompt": PromptTemplate(PAPERQA_SYSTEM_PROMPT)}
        )

        self.cost_logger = cost_logger

    @classmethod
    def from_config(cls, **kwargs):
        supabase_url = kwargs.get("supabase_url", os.environ["SUPABASE_URL"])
        supabase_service_key = kwargs.get(
            "supabase_service_key", os.environ["SUPABASE_SERVICE_KEY"]
        )
        embedding_model_name = kwargs.get(
            "embedding_model", "gemini/text-embedding-004"
        )
        summary_llm_model_name = kwargs.get(
            "summary_llm_model", "gemini/gemini-1.5-flash-002"
        )
        llm_model_name = kwargs.get("llm_model", "gemini/gemini-1.5-flash-002")
        toolspec = kwargs.get("toolspec")
        cost_logger = CostLogger()

        if toolspec is None:
            store = SupabaseStore(
                supabase_url=supabase_url,
                supabase_key=supabase_service_key,
            )
            cache = Cache()
            embedding_model = LiteLLMEmbeddingModel(
                name=embedding_model_name, cost_logger=cost_logger
            )
            summary_llm_model = LiteLLMModel(
                name=summary_llm_model_name, cost_logger=cost_logger
            )
            toolspec = PaperQAToolSpec(
                store=store,
                cache=cache,
                embedding_model=embedding_model,
                summary_llm_model=summary_llm_model,
                cost_logger=cost_logger,
            )

        llm = LiteLLM(llm_model_name)
        memory = ChatMemoryBuffer.from_defaults(
            chat_history=[],
            llm=llm,
        )

        self = cls(
            tools=toolspec.to_tool_list(),
            tool_retriever=None,
            llm=llm,
            memory=memory,
            max_iterations=10,
            react_chat_formatter=None,
            output_parser=PaperQAOutputParser(),
            callback_manager=None,
            verbose=kwargs.get("verbose", True),
            context=None,
            handle_reasoning_failure_fn=tell_llm_about_failure_in_extract_reasoning_step,
            cost_logger=cost_logger,
        )
        self.toolspec = toolspec
        return self

    async def stream_thoughts(
        self,
        query: str,
        current_document: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        step_by_step=False,
    ):
        self.memory.put(
            ChatMessage(
                role=MessageRole.SYSTEM,
                content="Remember to call gather_evidence_by_query or gather_policy_overview if the user is asking about Singapore health insurance, especially if you are citing anything. You can access documents that the user is seeing via these tools. Otherwise, just answer as per usual. Please avoid questions unrelated to Singapore health insurance but explain. You can ask the user to elaborate or clarify.",
            )
        )
        if current_document:
            if current_document in VALID_POLICIES:
                self.memory.put(
                    ChatMessage(
                        role=MessageRole.SYSTEM,
                        content=f'The user is currently looking at "{current_document}". This might be relevant to their request.',
                    )
                )
        self.memory.put(ChatMessage(role=MessageRole.USER, content=query))

        worker = cast(PaperQAAgentWorker, self.agent_worker)
        task = self.create_task(query)

        self.toolspec.current_task_id = task.task_id.split("-")[0]

        iters = 0
        max_iters = 10
        step_queue = self.state.get_step_queue(task.task_id)
        step = step_queue.popleft()
        while True:
            iters += 1

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

            response_success = False
            num_retries = 3
            retry_after = 5
            current_retry = 0
            while not response_success:
                try:
                    chat_stream = worker._llm.stream_chat(input_chat)

                    response_buffer = ""
                    for chunk in chat_stream:
                        value = chunk.message.content[len(response_buffer) :]
                        yield value
                        response_buffer += value
                        is_done = infer_stream_chunk_is_final(response_buffer, [])

                    response_buffer = parse_action_response(response_buffer)

                    response_success = True
                except (APIConnectionError, ServiceUnavailableError) as e:
                    current_retry += 1
                    if current_retry > num_retries:
                        break
                    logger.warn(
                        str(e)
                        + f"\nRetrying ({current_retry}/{num_retries}) after {retry_after}s..."
                    )
                    await asyncio.sleep(retry_after)
            if not response_success:
                self.memory.put(
                    ChatMessage(
                        role=MessageRole.ASSISTANT, content=FALLBACK_RESPONSE_CONTENT
                    )
                )
                yield FALLBACK_FINAL_RESPONSE
                return

            if not is_done:
                tools = worker.get_tools(task.input)
                tools_dict = {tool.metadata.get_name(): tool for tool in tools}

                # Extract tool to yield tool description
                try:
                    # Temporarily disable verbose to prevent repeated logging
                    _verbose = worker._verbose
                    worker._verbose = False
                    _, current_reasoning, is_done = worker._extract_reasoning_step(
                        response_buffer, is_streaming=True
                    )
                    worker._verbose = _verbose
                    reasoning_step = cast(ActionReasoningStep, current_reasoning[-1])
                    if reasoning_step.action in tools_dict:
                        # Populate with default kwargs and log description
                        if hasattr(
                            tools_dict[reasoning_step.action].fn, "__default_kwargs__"
                        ):
                            reasoning_step.action_input = {
                                **tools_dict[
                                    reasoning_step.action
                                ].fn.__default_kwargs__,
                                **reasoning_step.action_input,
                            }
                        thought = f"""Action Desc: {
                            tools_dict[reasoning_step.action].fn.__output_desc__.format(
                                **reasoning_step.action_input
                            )
                        }"""
                        # self.memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=thought))
                        yield thought
                except ValueError:
                    pass

                # given react prompt outputs, call tools or return response
                reasoning_steps, is_done = worker._process_actions(
                    task, tools=tools, output=response_buffer, is_streaming=True
                )
                if reasoning_steps[-1].observation.startswith("Found"):
                    thought = (
                        "Action Output:" + reasoning_steps[-1].observation.split(".")[0]
                    )
                    # self.memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=thought))
                    yield thought
                task.extra_state["current_reasoning"].extend(reasoning_steps)

                step = step.get_next_step(
                    step_id=str(uuid.uuid4()),
                    input=None,
                )

            # Calculate cost with final chunk
            cost = completion_cost(
                completion_response=ModelResponse(
                    model=chunk.raw.model,
                    usage=chunk.raw._hidden_params.get("usage"),
                ),
                custom_llm_provider=chunk.raw._hidden_params.get("custom_llm_provider"),
            )
            self.cost_logger.log_cost(cost)

            if step_by_step:
                interrupt = input("Enter to continue or 'q' to quit: ")
                if interrupt == "q":
                    quit()
            if is_done:
                break
            else:
                if "\nThought: " in response_buffer:
                    response_buffer = response_buffer.split("\nThought: ")[0]
                if "```Thought: " in response_buffer:
                    response_buffer = response_buffer.split("```Thought: ")[0] + "```"
                self.memory.put(
                    ChatMessage(role=MessageRole.ASSISTANT, content=response_buffer)
                )
            if iters >= max_iters:
                break

        final_response = response_buffer.split("Answer:")[-1].strip()
        self.memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=final_response))

        # Suggest shortcut responses
        suggested_responses = await suggest_follow_up(self)

        recent_history = []
        for i, message in enumerate(
            self.memory.chat_store.to_dict()["store"]["chat_history"][::-1]
        ):
            if message["role"] == "assistant":
                if i != 0:
                    message["hidden"] = True
                else:
                    message["formattedContent"] = format_response(
                        query,
                        final_response,
                        self.toolspec,
                        prev_document_ids=document_ids or [],
                    )
                    message["formattedContent"][
                        "suggestedResponses"
                    ] = suggested_responses
                recent_history.insert(0, message)
            else:
                break

        yield "Final Response: " + json.dumps(recent_history)

    def pprint_memory(self):
        class sty:
            WHITE = "\033[37m"
            CYAN = "\033[38;5;51m"
            MAGENTA = "\033[38;5;207m"
            BOLD = "\033[1m"
            RESET = "\033[0m"

        for memory in self.memory.chat_store.store["chat_history"]:
            if memory.role == MessageRole.USER:
                color = sty.CYAN
            elif memory.role == MessageRole.ASSISTANT:
                color = sty.MAGENTA
            else:
                color = sty.WHITE
            formatted_str = f"{sty.BOLD}{color}{str(memory.role.value.upper())}{sty.RESET}{color}: {memory.content}{sty.RESET}"
            print(formatted_str)
