from sys import version_info
from typing import (
    Any,
    AsyncIterable,
    Iterable,
    cast,
)

import litellm
from pydantic import (
    ConfigDict,
    Field,
    TypeAdapter,
    model_validator,
)

from ..utils.logger import CostLogger
from .llm_model import (
    Chunk,
    LLMModel,
    rate_limited,
)

IS_PYTHON_BELOW_312 = version_info < (3, 12)
if not IS_PYTHON_BELOW_312:
    _DeploymentTypedDictValidator = TypeAdapter(
        list[litellm.DeploymentTypedDict],
        config=ConfigDict(arbitrary_types_allowed=True),
    )


DEFAULT_VERTEX_SAFETY_SETTINGS: list[dict[str, str]] = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_ONLY_HIGH",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_ONLY_HIGH",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_ONLY_HIGH",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_ONLY_HIGH",
    },
]


def get_litellm_retrying_config(timeout: float = 60.0) -> dict[str, Any]:
    """Get retrying configuration for litellm.acompletion and litellm.aembedding."""
    return {"num_retries": 3, "timeout": timeout}


class PassThroughRouter(litellm.Router):
    """Router that is just a wrapper on LiteLLM's normal free functions."""

    def __init__(self, **kwargs):
        self._default_kwargs = kwargs

    async def atext_completion(self, *args, **kwargs):
        return await litellm.atext_completion(*args, **(self._default_kwargs | kwargs))

    async def acompletion(self, *args, **kwargs):
        return await litellm.acompletion(*args, **(self._default_kwargs | kwargs))


class LiteLLMModel(LLMModel):
    """A wrapper around the litellm library."""

    config: dict = Field(
        default_factory=dict,
        description=(
            "Configuration of this model containing several important keys. The"
            " optional `model_list` key stores a list of all model configurations"
            " (SEE: https://docs.litellm.ai/docs/routing). The optional"
            " `router_kwargs` key is keyword arguments to pass to the Router class."
            " Inclusion of a key `pass_through_router` with a truthy value will lead"
            " to using not using LiteLLM's Router, instead just LiteLLM's free"
            f" functions (see {PassThroughRouter.__name__}). Rate limiting applies"
            " regardless of `pass_through_router` being present. The optional"
            " `rate_limit` key is a dictionary keyed by model group name with values"
            " of type limits.RateLimitItem (in tokens / minute) or valid"
            " limits.RateLimitItem string for parsing."
        ),
    )
    name: str = "gpt-4o-mini"
    _router: litellm.Router | None = None
    cost_logger: CostLogger = CostLogger()

    @model_validator(mode="before")
    @classmethod
    def maybe_set_config_attribute(cls, data: dict[str, Any]) -> dict[str, Any]:
        """If a user only gives a name, make a sensible config dict for them."""
        if "config" not in data:
            data["config"] = {}
        if "name" in data and "model_list" not in data["config"]:
            data["config"] = {
                "model_list": [
                    {
                        "model_name": data["name"],
                        "litellm_params": {"model": data["name"]}
                        | (
                            {}
                            if "gemini" not in data["name"]
                            else {"safety_settings": DEFAULT_VERTEX_SAFETY_SETTINGS}
                        ),
                    }
                ],
            } | data["config"]

        if "router_kwargs" not in data["config"]:
            data["config"]["router_kwargs"] = {}
        data["config"]["router_kwargs"] = (
            get_litellm_retrying_config() | data["config"]["router_kwargs"]
        )
        if not data["config"].get("pass_through_router"):
            data["config"]["router_kwargs"] = {"retry_after": 5} | data["config"][
                "router_kwargs"
            ]

        # we only support one "model name" for now, here we validate
        model_list = data["config"]["model_list"]
        if IS_PYTHON_BELOW_312:
            if not isinstance(model_list, list):
                # Work around https://github.com/BerriAI/litellm/issues/5664
                raise TypeError(f"model_list must be a list, not a {type(model_list)}.")
        else:
            # pylint: disable-next=possibly-used-before-assignment
            _DeploymentTypedDictValidator.validate_python(model_list)
        if len({m["model_name"] for m in model_list}) > 1:
            raise ValueError("Only one model name per model list is supported for now.")
        return data

    def __getstate__(self):
        # Prevent _router from being pickled, SEE: https://stackoverflow.com/a/2345953
        state = super().__getstate__()
        state["__dict__"] = state["__dict__"].copy()
        state["__dict__"].pop("_router", None)
        return state

    @property
    def router(self) -> litellm.Router:
        if self._router is None:
            router_kwargs: dict = self.config.get("router_kwargs", {})
            if self.config.get("pass_through_router"):
                self._router = PassThroughRouter(**router_kwargs)
            else:
                self._router = litellm.Router(
                    model_list=self.config["model_list"], **router_kwargs
                )
        return self._router

    async def check_rate_limit(self, token_count: float, **kwargs) -> None:
        # if "rate_limit" in self.config:
        #     await GLOBAL_LIMITER.try_acquire(
        #         ("client", self.name),
        #         self.config["rate_limit"].get(self.name, None),
        #         weight=max(int(token_count), 1),
        #         **kwargs,
        #     )
        return None

    @rate_limited
    async def acomplete(self, prompt: str) -> Chunk:  # type: ignore[override]
        response = await self.router.atext_completion(model=self.name, prompt=prompt)
        self.cost_logger.log_cost(response._hidden_params.get("response_cost"))
        return Chunk(
            text=response.choices[0].text,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            cost=response._hidden_params.get("response_cost"),
        )

    @rate_limited
    async def acomplete_iter(  # type: ignore[override]
        self, prompt: str
    ) -> AsyncIterable[Chunk]:
        completion = await self.router.atext_completion(
            model=self.name,
            prompt=prompt,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in completion:
            yield Chunk(
                text=chunk.choices[0].text, prompt_tokens=0, completion_tokens=0
            )
        if hasattr(chunk, "usage") and hasattr(chunk.usage, "prompt_tokens"):
            self.cost_logger.log_cost(chunk._hidden_params.get("response_cost"))
            yield Chunk(
                text=chunk.choices[0].text,
                prompt_tokens=chunk.usage.prompt_tokens,
                completion_tokens=chunk.usage.completion_tokens,
                cost=chunk._hidden_params.get("response_cost"),
            )

    @rate_limited
    async def achat(  # type: ignore[override]
        self, messages: Iterable[dict[str, str]]
    ) -> Chunk:
        response = await self.router.acompletion(self.name, list(messages))
        self.cost_logger.log_cost(response._hidden_params.get("response_cost"))
        return Chunk(
            text=cast(litellm.Choices, response.choices[0]).message.content,
            prompt_tokens=response.usage.prompt_tokens,  # type: ignore[attr-defined]
            completion_tokens=response.usage.completion_tokens,  # type: ignore[attr-defined]
            cost=response._hidden_params.get("response_cost"),
        )

    @rate_limited
    async def achat_iter(  # type: ignore[override]
        self, messages: Iterable[dict[str, str]]
    ) -> AsyncIterable[Chunk]:
        completion = await self.router.acompletion(
            self.name,
            list(messages),
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in completion:
            yield Chunk(
                text=chunk.choices[0].delta.content,
                prompt_tokens=0,
                completion_tokens=0,
            )
        if hasattr(chunk, "usage") and hasattr(chunk.usage, "prompt_tokens"):
            self.cost_logger.log_cost(chunk._hidden_params.get("response_cost"))
            yield Chunk(
                text=None,
                prompt_tokens=chunk.usage.prompt_tokens,
                completion_tokens=chunk.usage.completion_tokens,
                cost=chunk._hidden_params.get("response_cost"),
            )

    def infer_llm_type(self) -> str:
        if all(
            "text-completion" in m.get("litellm_params", {}).get("model", "")
            for m in self.config["model_list"]
        ):
            return "completion"
        return "chat"

    def count_tokens(self, text: str) -> int:
        return litellm.token_counter(model=self.name, text=text)
