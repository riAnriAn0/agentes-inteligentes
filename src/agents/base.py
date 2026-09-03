from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.types import Action, Perception, Transition


class Agent(ABC):
    name = "agent"

    def set_training(self, training: bool) -> None:
        """Alterna entre coleta de experiência e avaliação da política."""
        self.training = training

    @abstractmethod
    def act(self, perception: Perception) -> Action:
        raise NotImplementedError

    def observe_transition(self, transition: Transition) -> None:
        pass

    def reset(self) -> None:
        pass

    def diagnostics(self) -> dict[str, Any]:
        return {}
