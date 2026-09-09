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
        # Dicas:
        # - use perception.cell_at((row, col)) para consultar células visíveis;
        # - use perception.position para obter a posição atual;
        # - use perception.battery e perception.max_battery para gerenciar a bateria;
        # - use perception.on_victim, perception.on_charger e perception.on_exit para ações especiais;
        # - use perception.rescued e perception.total_victims para monitorar o progresso do resgate.

        # identifica se está sobre uma vítima e resgata
        if perception.on_victim:
            return Action.RESCUE

        # identifica se está sobre um carregador e recarrega
        if perception.on_charger and perception.battery < perception.max_battery:
            return Action.RECHARGE

        # identifica os vizinhos e verifica se são seguros (não são paredes ou perigos)
        neighbors = {
            Action.NORTH: (perception.position[0] - 1, perception.position[1]),
            Action.SOUTH: (perception.position[0] + 1, perception.position[1]),
            Action.EAST: (perception.position[0], perception.position[1] + 1),
            Action.WEST: (perception.position[0], perception.position[1] - 1),
        }

        def is_safe(action: Action) -> bool:
            cell = perception.cell_at(neighbors[action])
            return cell is not None and cell not in {Cell.WALL, Cell.HAZARD}

        # prioriza ações que levam a vítimas visíveis
        visible_victim = [action for action in MOVEMENT_ACTIONS if is_safe(action) and perception.cell_at(neighbors[action]) == Cell.VICTIM]
        if visible_victim:
            return visible_victim[0]

        # prioriza ações que levam a carregadores visíveis se a bateria estiver baixa
        if perception.battery < perception.max_battery // 3:
            visible_charger = [
                action
                for action in MOVEMENT_ACTIONS
                if is_safe(action) and perception.cell_at(neighbors[action]) == Cell.CHARGER
            ]
            if visible_charger:
                return visible_charger[0]

        # prioriza ações que levam a saída visível se todas as vítimas foram resgatadas
        for action in MOVEMENT_ACTIONS:
            if is_safe(action):
                return action

        return Action.WAIT

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
