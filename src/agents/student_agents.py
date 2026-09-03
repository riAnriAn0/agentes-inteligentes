from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from src.agents.base import Agent
from src.types import Action, Cell, Perception, Transition


MOVEMENT_ACTIONS = [Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST]


class SimpleReflexAgent(Agent):
    """OBRIGATÓRIO: usar somente a percepção atual."""

    name = "simple"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def act(self, perception: Perception) -> Action:
        # TODO: regras condição -> ação.
        # Não use memória, mapa interno, visitados ou Q-table.
        raise NotImplementedError("Implemente SimpleReflexAgent.act().")

    def reset(self) -> None:
        pass


class ModelBasedAgent(Agent):
    """OBRIGATÓRIO: manter estado interno, sem exigir busca."""

    name = "model"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.reset()

    def reset(self) -> None:
        # TODO: adapte ou amplie a memória conforme sua estratégia.
        self.known_map: dict[tuple[int, int], Cell] = {}
        self.visit_count: dict[tuple[int, int], int] = defaultdict(int)
        self.last_action: Action | None = None

    def _update_model(self, perception: Perception) -> None:
        # TODO: atualize known_map, visit_count etc.
        pass

    def act(self, perception: Perception) -> Action:
        # TODO:
        # 1. atualize o modelo interno;
        # 2. trate resgate/recarga;
        # 3. prefira células pouco visitadas;
        # 4. evite paredes e perigos conhecidos.
        raise NotImplementedError("Implemente ModelBasedAgent.act().")

    def diagnostics(self) -> dict[str, Any]:
        return {
            "known_cells": len(self.known_map),
            "visited_cells": len(self.visit_count),
        }


class LearningAgent(Agent):
    """OBRIGATÓRIO: sugestão de implementação com Q-learning tabular."""

    name = "learning"

    def __init__(
        self,
        seed: int = 0,
        alpha: float = 0.2,
        gamma: float = 0.95,
        epsilon: float = 0.15,
        evaluation_epsilon: float = 0.0,
    ) -> None:
        self.rng = random.Random(seed)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.evaluation_epsilon = evaluation_epsilon
        self.training = True
        self.q: dict[tuple[Any, Action], float] = defaultdict(float)

    @property
    def exploration_rate(self) -> float:
        """Epsilon usado na fase atual do experimento."""
        return self.epsilon if self.training else self.evaluation_epsilon

    def reset(self) -> None:
        # Não apague self.q: o conhecimento deve persistir entre episódios.
        pass

    def _state(self, perception: Perception) -> Any:
        # TODO: crie uma representação compacta do estado.
        raise NotImplementedError("Implemente LearningAgent._state().")

    def _available_actions(self, perception: Perception) -> list[Action]:
        # TODO: filtre ações claramente inválidas usando a percepção atual.
        raise NotImplementedError("Implemente LearningAgent._available_actions().")

    def act(self, perception: Perception) -> Action:
        # TODO: política epsilon-greedy usando self.exploration_rate.
        raise NotImplementedError("Implemente LearningAgent.act().")

    def observe_transition(self, transition: Transition) -> None:
        if not self.training:
            return
        # TODO: atualização Q-learning.
        raise NotImplementedError("Implemente LearningAgent.observe_transition().")

    def diagnostics(self) -> dict[str, Any]:
        return {
            "q_entries": len(self.q),
            "q_states": len({state for state, _action in self.q}),
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "evaluation_epsilon": self.evaluation_epsilon,
            "training": int(self.training),
        }


class GoalBasedAgent(Agent):
    """OPCIONAL: recomendado após busca/planejamento."""

    name = "goal"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def act(self, perception: Perception) -> Action:
        raise NotImplementedError("Extensão opcional: agente baseado em objetivos.")


class UtilityBasedAgent(Agent):
    """OPCIONAL: recomendado após busca e funções de utilidade."""

    name = "utility"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def act(self, perception: Perception) -> Action:
        raise NotImplementedError("Extensão opcional: agente baseado em utilidade.")
