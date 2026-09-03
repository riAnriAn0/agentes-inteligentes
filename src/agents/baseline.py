from __future__ import annotations

import random

from src.agents.base import Agent
from src.types import Action, Perception


class RandomAgent(Agent):
    """Baseline fornecido apenas para testar a infraestrutura."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def act(self, perception: Perception) -> Action:
        actions = [Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST, Action.WAIT]
        if perception.on_victim:
            actions.append(Action.RESCUE)
        if perception.on_charger and perception.battery < perception.max_battery:
            actions.append(Action.RECHARGE)
        return self.rng.choice(actions)
