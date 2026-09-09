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
        self.known_map.update(perception.visible_cells)
        self.visit_count[perception.position] += 1

    def act(self, perception: Perception) -> Action:
        self._update_model(perception)

        if perception.on_victim:
            return Action.RESCUE

        if perception.on_charger and perception.battery < perception.max_battery:
            return Action.RECHARGE
        
        neighbors = {
            Action.NORTH: (perception.position[0] - 1, perception.position[1]),
            Action.SOUTH: (perception.position[0] + 1, perception.position[1]),
            Action.EAST: (perception.position[0], perception.position[1] + 1),
            Action.WEST: (perception.position[0], perception.position[1] - 1),
        }

        def cell_for(action: Action) -> Cell | None:
            position = neighbors[action]
            return perception.cell_at(position) or self.known_map.get(position)

        def is_safe(action: Action) -> bool:
            cell = cell_for(action)
            return cell is not None and cell not in {Cell.WALL, Cell.HAZARD}

        available = [action for action in MOVEMENT_ACTIONS if is_safe(action)]
        if not available:
            action = Action.WAIT
        else:
            action = min(
                available,
                key=lambda candidate: (
                    self.visit_count.get(neighbors[candidate], 0),
                    MOVEMENT_ACTIONS.index(candidate),
                ),
            )

        self.last_action = action
        return action

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
        row, column = perception.position
        neighbors = (
            perception.cell_at((row - 1, column)),
            perception.cell_at((row + 1, column)),
            perception.cell_at((row, column - 1)),
            perception.cell_at((row, column + 1)),
        )
        battery_bucket = min(
            4,
            (perception.battery * 5) // max(1, perception.max_battery),
        )
        return (
            perception.position,
            battery_bucket,
            tuple(cell.value if cell is not None else None for cell in neighbors),
            perception.on_victim,
            perception.on_charger,
            perception.on_exit,
            perception.rescued,
            perception.total_victims,
        )

    def _available_actions(self, perception: Perception) -> list[Action]:
        row, column = perception.position
        neighbors = {
            Action.NORTH: (row - 1, column),
            Action.SOUTH: (row + 1, column),
            Action.EAST: (row, column + 1),
            Action.WEST: (row, column - 1),
        }
        actions = [
            action
            for action in MOVEMENT_ACTIONS
            if perception.cell_at(neighbors[action]) not in {Cell.WALL, Cell.HAZARD}
        ]
        if perception.on_victim:
            actions.append(Action.RESCUE)
        if perception.on_charger and perception.battery < perception.max_battery:
            actions.append(Action.RECHARGE)
        actions.append(Action.WAIT)
        return actions

    def act(self, perception: Perception) -> Action:
        state = self._state(perception)
        actions = self._available_actions(perception)
        if self.rng.random() < self.exploration_rate:
            return self.rng.choice(actions)

        values = [self.q.get((state, action), 0.0) for action in actions]
        best_value = max(values)
        return next(
            action for action, value in zip(actions, values) if value == best_value
        )

    def observe_transition(self, transition: Transition) -> None:
        if not self.training:
            return
        state = self._state(transition.perception)
        key = (state, transition.action)
        current_value = self.q[key]

        future_value = 0.0
        if not transition.done:
            next_state = self._state(transition.next_perception)
            next_actions = self._available_actions(transition.next_perception)
            future_value = max(
                (self.q.get((next_state, action), 0.0) for action in next_actions),
                default=0.0,
            )

        target = transition.reward + self.gamma * future_value
        self.q[key] = current_value + self.alpha * (target - current_value)

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
