import logging

from pydantic import BaseModel, ConfigDict


class CostLogger(BaseModel):
    logger: logging.Logger = logging.getLogger("paperqa-cost")
    _total_cost: float = 0
    _STY_COLOR = "\033[38;5;69m"
    _STY_RESET = "\033[0m"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def log_cost(self, cost: float):
        cost = cost or 0
        self.logger.info(f"{self._STY_COLOR}Cost: {cost}USD{self._STY_RESET}")
        self._total_cost += cost

    def reset(self):
        self._total_cost = 0

    @property
    def total_cost(self):
        return self._total_cost
