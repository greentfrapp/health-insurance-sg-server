import asyncio
import contextlib
import functools
import logging
from abc import ABC
from inspect import isasyncgenfunction, signature
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Optional,
    TypeVar,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from ..utils.utils import is_coroutine_callable
from .llm_result import LLMResult

# Estimate from OpenAI's FAQ
# https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
CHARACTERS_PER_TOKEN_ASSUMPTION: float = 4.0
# Added tokens from user/role message
# Need to add while doing rate limits
# Taken from empirical counts in tests
EXTRA_TOKENS_FROM_USER_ROLE: int = 7

DEFAULT_SYSTEM_PROMPT = ""
logger = logging.getLogger(__name__)


def prepare_args(func: Callable, chunk: str, name: str | None) -> tuple[tuple, dict]:
    with contextlib.suppress(TypeError):
        if "name" in signature(func).parameters:
            return (chunk,), {"name": name}
    return (chunk,), {}


async def do_callbacks(
    async_callbacks: Iterable[Callable[..., Awaitable]],
    sync_callbacks: Iterable[Callable[..., Any]],
    chunk: str,
    name: str | None,
) -> None:
    for f in async_callbacks:
        args, kwargs = prepare_args(f, chunk, name)
        await f(*args, **kwargs)
    for f in sync_callbacks:
        args, kwargs = prepare_args(f, chunk, name)
        f(*args, **kwargs)


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str | None
    prompt_tokens: int
    completion_tokens: int
    cost: Optional[float] = None

    def __str__(self):
        return self.text


class LLMModel(ABC, BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    llm_type: str | None = None
    name: str
    llm_result_callback: (
        Callable[[LLMResult], None] | Callable[[LLMResult], Awaitable[None]] | None
    ) = Field(
        default=None,
        description=(
            "An async callback that will be executed on each"
            " LLMResult (different than callbacks that execute on each chunk)"
        ),
        exclude=True,
    )
    config: dict = Field(default_factory=dict)

    async def acomplete(self, prompt: str) -> Chunk:
        """Return the completion as string and the number of tokens in the prompt and completion."""
        raise NotImplementedError

    async def acomplete_iter(self, prompt: str) -> AsyncIterable[Chunk]:
        """Return an async generator that yields chunks of the completion.

        Only the last tuple will be non-zero.
        """
        raise NotImplementedError
        if False:  # type: ignore[unreachable]  # pylint: disable=using-constant-test
            yield  # Trick mypy: https://github.com/python/mypy/issues/5070#issuecomment-1050834495

    async def achat(self, messages: Iterable[dict[str, str]]) -> Chunk:
        """Return the completion as string and the number of tokens in the prompt and completion."""
        raise NotImplementedError

    async def achat_iter(
        self,
        messages: Iterable[dict[str, str]],
    ) -> AsyncIterable[Chunk]:
        """Return an async generator that yields chunks of the completion.

        Only the last tuple will be non-zero.
        """
        raise NotImplementedError
        if False:  # type: ignore[unreachable]  # pylint: disable=using-constant-test
            yield  # Trick mypy: https://github.com/python/mypy/issues/5070#issuecomment-1050834495

    def infer_llm_type(self) -> str:
        return "completion"

    def count_tokens(self, text: str) -> int:
        return len(text) // 4  # gross approximation

    async def run_prompt(
        self,
        prompt: str,
        data: dict,
        callbacks: list[Callable] | None = None,
        name: str | None = None,
        skip_system: bool = False,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> LLMResult:
        if self.llm_type is None:
            self.llm_type = self.infer_llm_type()
        if self.llm_type == "chat":
            return await self._run_chat(
                prompt, data, callbacks, name, skip_system, system_prompt
            )
        if self.llm_type == "completion":
            return await self._run_completion(
                prompt, data, callbacks, name, skip_system, system_prompt
            )
        raise ValueError(f"Unknown llm_type {self.llm_type!r}.")

    async def _run_chat(
        self,
        prompt: str,
        data: dict,
        callbacks: list[Callable] | None = None,
        name: str | None = None,
        skip_system: bool = False,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> LLMResult:
        """Run a chat prompt.

        Args:
            prompt: Prompt to use.
            data: Keys for the input variables that will be formatted into prompt.
            callbacks: Optional functions to call with each chunk of the completion.
            name: Optional name for the result.
            skip_system: Set True to skip the system prompt.
            system_prompt: System prompt to use.

        Returns:
            Result of the chat.
        """
        system_message_prompt = {"role": "system", "content": system_prompt}
        human_message_prompt = {"role": "user", "content": prompt}
        messages = [
            {"role": m["role"], "content": m["content"].format(**data)}
            for m in (
                [human_message_prompt]
                if skip_system
                else [system_message_prompt, human_message_prompt]
            )
        ]
        result = LLMResult(
            model=self.name,
            name=name,
            prompt=messages,
            prompt_count=(
                sum(self.count_tokens(m["content"]) for m in messages)
                + sum(self.count_tokens(m["role"]) for m in messages)
            ),
        )

        start_clock = asyncio.get_running_loop().time()
        if callbacks is None:
            chunk = await self.achat(messages)
            output = chunk.text
        else:
            sync_callbacks = [f for f in callbacks if not is_coroutine_callable(f)]
            async_callbacks = [f for f in callbacks if is_coroutine_callable(f)]
            completion = await self.achat_iter(messages)  # type: ignore[misc]
            text_result = []
            async for chunk in completion:
                if chunk.text:
                    if result.seconds_to_first_token == 0:
                        result.seconds_to_first_token = (
                            asyncio.get_running_loop().time() - start_clock
                        )
                    text_result.append(chunk.text)
                    await do_callbacks(
                        async_callbacks, sync_callbacks, chunk.text, name
                    )
            output = "".join(text_result)
        usage = chunk.prompt_tokens, chunk.completion_tokens
        if sum(usage) > 0:
            result.prompt_count, result.completion_count = usage
        elif output:
            result.completion_count = self.count_tokens(output)
        result.text = output or ""
        result.seconds_to_last_token = asyncio.get_running_loop().time() - start_clock
        if self.llm_result_callback:
            if is_coroutine_callable(self.llm_result_callback):
                await self.llm_result_callback(result)  # type: ignore[misc]
            else:
                self.llm_result_callback(result)
        return result

    async def _run_completion(
        self,
        prompt: str,
        data: dict,
        callbacks: Iterable[Callable] | None = None,
        name: str | None = None,
        skip_system: bool = False,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> LLMResult:
        """Run a completion prompt.

        Args:
            prompt: Prompt to use.
            data: Keys for the input variables that will be formatted into prompt.
            callbacks: Optional functions to call with each chunk of the completion.
            name: Optional name for the result.
            skip_system: Set True to skip the system prompt.
            system_prompt: System prompt to use.

        Returns:
            Result of the completion.
        """
        formatted_prompt: str = (
            prompt if skip_system else system_prompt + "\n\n" + prompt
        ).format(**data)
        result = LLMResult(
            model=self.name,
            name=name,
            prompt=formatted_prompt,
            prompt_count=self.count_tokens(formatted_prompt),
        )

        start_clock = asyncio.get_running_loop().time()
        if callbacks is None:
            chunk = await self.acomplete(formatted_prompt)
            output = chunk.text
        else:
            sync_callbacks = [f for f in callbacks if not is_coroutine_callable(f)]
            async_callbacks = [f for f in callbacks if is_coroutine_callable(f)]

            completion = self.acomplete_iter(formatted_prompt)
            text_result = []
            async for chunk in completion:
                if chunk.text:
                    if result.seconds_to_first_token == 0:
                        result.seconds_to_first_token = (
                            asyncio.get_running_loop().time() - start_clock
                        )
                    text_result.append(chunk.text)
                    await do_callbacks(
                        async_callbacks, sync_callbacks, chunk.text, name
                    )
            output = "".join(text_result)
        usage = chunk.prompt_tokens, chunk.completion_tokens
        if sum(usage) > 0:
            result.prompt_count, result.completion_count = usage
        elif output:
            result.completion_count = self.count_tokens(output)
        result.text = output or ""
        result.seconds_to_last_token = asyncio.get_running_loop().time() - start_clock
        if self.llm_result_callback:
            if is_coroutine_callable(self.llm_result_callback):
                await self.llm_result_callback(result)  # type: ignore[misc]
            else:
                self.llm_result_callback(result)
        return result


LLMModelOrChild = TypeVar("LLMModelOrChild", bound=LLMModel)


def rate_limited(
    func: Callable[[LLMModelOrChild, Any], Awaitable[Chunk] | AsyncIterable[Chunk]],
) -> Callable[
    [LLMModelOrChild, Any, Any],
    Awaitable[Chunk | AsyncIterator[Chunk] | AsyncIterator[LLMModelOrChild]],
]:
    """Decorator to rate limit relevant methods of an LLMModel."""

    @functools.wraps(func)
    async def wrapper(
        self: LLMModelOrChild, *args: Any, **kwargs: Any
    ) -> Chunk | AsyncIterator[Chunk] | AsyncIterator[LLMModelOrChild]:
        if not hasattr(self, "check_rate_limit"):
            raise NotImplementedError(
                f"Model {self.name} must have a `check_rate_limit` method."
            )

        # Estimate token count based on input
        if func.__name__ in {"acomplete", "acomplete_iter"}:
            prompt = args[0] if args else kwargs.get("prompt", "")
            token_count = (
                len(prompt) / CHARACTERS_PER_TOKEN_ASSUMPTION
                + EXTRA_TOKENS_FROM_USER_ROLE
            )
        elif func.__name__ in {"achat", "achat_iter"}:
            messages = args[0] if args else kwargs.get("messages", [])
            token_count = len(str(messages)) / CHARACTERS_PER_TOKEN_ASSUMPTION
        else:
            token_count = 0  # Default if method is unknown

        await self.check_rate_limit(token_count)

        # If wrapping a generator, count the tokens for each
        # portion before yielding
        if isasyncgenfunction(func):

            async def rate_limited_generator() -> AsyncGenerator[LLMModelOrChild, None]:
                async for item in func(self, *args, **kwargs):
                    token_count = 0
                    if isinstance(item, Chunk):
                        token_count = int(
                            len(item.text or "") / CHARACTERS_PER_TOKEN_ASSUMPTION
                        )
                    await self.check_rate_limit(token_count)
                    yield item

            return rate_limited_generator()

        result = await func(self, *args, **kwargs)  # type: ignore[misc]

        if func.__name__ in {"acomplete", "achat"} and isinstance(result, Chunk):
            await self.check_rate_limit(result.completion_tokens)
        return result

    return wrapper
