from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    Field,
    field_validator,
)
import litellm
import tiktoken

from .litellm_model import get_litellm_retrying_config
from ..utils.logger import CostLogger


# Estimate from OpenAI's FAQ
# https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
CHARACTERS_PER_TOKEN_ASSUMPTION: float = 4.0
# Added tokens from user/role message
# Need to add while doing rate limits
# Taken from empirical counts in tests
EXTRA_TOKENS_FROM_USER_ROLE: int = 7

MODEL_COST_MAP = litellm.get_model_cost_map("")


class EmbeddingModes(StrEnum):
    DOCUMENT = "document"
    QUERY = "query"


class EmbeddingModel(ABC, BaseModel):
    name: str
    config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional `rate_limit` key, value must be a RateLimitItem or RateLimitItem"
            " string for parsing"
        ),
    )

    async def check_rate_limit(self, token_count: float, **kwargs) -> None:
        # if "rate_limit" in self.config:
        #     await GLOBAL_LIMITER.try_acquire(
        #         ("client", self.name),
        #         self.config["rate_limit"],
        #         weight=max(int(token_count), 1),
        #         **kwargs,
        #     )
        return

    def set_mode(self, mode: EmbeddingModes) -> None:
        """Several embedding models have a 'mode' or prompt which affects output."""

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        pass


class LiteLLMEmbeddingModel(EmbeddingModel):
    name: str = Field(default="text-embedding-3-small")
    config: dict[str, Any] = Field(
        default_factory=dict,  # See below field_validator for injection of kwargs
        description=(
            "The optional `rate_limit` key's value must be a RateLimitItem or"
            " RateLimitItem string for parsing. The optional `kwargs` key is keyword"
            " arguments to pass to the litellm.aembedding function. Note that LiteLLM's"
            " Router is not used here."
        ),
    )
    cost_logger: CostLogger = CostLogger()

    @field_validator("config")
    @classmethod
    def set_up_default_config(cls, value: dict[str, Any]) -> dict[str, Any]:
        if "kwargs" not in value:
            value["kwargs"] = get_litellm_retrying_config(
                timeout=120,  # 2-min timeout seemed reasonable
            )
        return value

    def _truncate_if_large(self, texts: list[str]) -> list[str]:
        """Truncate texts if they are too large by using litellm cost map."""
        if self.name not in MODEL_COST_MAP:
            return texts
        max_tokens = MODEL_COST_MAP[self.name]["max_input_tokens"]
        # heuristic about ratio of tokens to characters
        conservative_char_token_ratio = 3
        maybe_too_large = max_tokens * conservative_char_token_ratio
        if any(len(t) > maybe_too_large for t in texts):
            try:
                enct = tiktoken.encoding_for_model("cl100k_base")
                enc_batch = enct.encode_ordinary_batch(texts)
                return [enct.decode(t[:max_tokens]) for t in enc_batch]
            except KeyError:
                return [t[: max_tokens * conservative_char_token_ratio] for t in texts]

        return texts

    async def embed_documents(
        self, texts: list[str], batch_size: int = 16
    ) -> list[list[float]]:
        texts = self._truncate_if_large(texts)
        N = len(texts)
        embeddings = []
        for i in range(0, N, batch_size):
            await self.check_rate_limit(
                sum(
                    len(t) / CHARACTERS_PER_TOKEN_ASSUMPTION
                    for t in texts[i : i + batch_size]
                )
            )

            response = await litellm.aembedding(
                self.name,
                input=texts[i : i + batch_size],
                **self.config.get("kwargs", {}),
            )
            embeddings.extend([e["embedding"] for e in response.data])
            self.cost_logger.log_cost(response._hidden_params.get("response_cost", 0))

        return embeddings
