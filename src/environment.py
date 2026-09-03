from __future__ import annotations

import random
from dataclasses import dataclass

from src.scenarios import Scenario
from src.types import Action, Cell, Perception


MOVE_DELTA = {
    Action.NORTH: (-1, 0),
    Action.SOUTH: (1, 0),
    Action.WEST: (0, -1),
    Action.EAST: (0, 1),
}


@dataclass
class StepResult:
    reward: float
    done: bool
    info: dict


class RescueEnvironment:
    """Ambiente fornecido pelo professor. O agente recebe somente Perception."""

    def __init__(self, scenario: Scenario, seed: int = 0) -> None:
        self.scenario = scenario
        self.seed = seed
        self.rng = random.Random(seed)

        self.height = len(scenario.grid)
        self.width = len(scenario.grid[0])
        if any(len(row) != self.width for row in scenario.grid):
            raise ValueError("Todas as linhas do mapa devem ter o mesmo tamanho.")

        self.base_cells: dict[tuple[int, int], Cell] = {}
        for r, row in enumerate(scenario.grid):
            for c, symbol in enumerate(row):
                self.base_cells[(r, c)] = Cell(symbol)

        self.initial_victims = {
            pos for pos, cell in self.base_cells.items() if cell == Cell.VICTIM
        }
        self.reset()

    def reset(self) -> Perception:
        self.position = self.scenario.start
        self.battery = self.scenario.max_battery
        self.step_count = 0
        self.rescued = 0
        self.victims = set(self.initial_victims)
        self.done = False
        self.energy_used = 0
        self.hazard_entries = 0
        self.failed = False
        self.success = False
        return self.perceive()

    @property
    def total_victims(self) -> int:
        return len(self.initial_victims)

    def in_bounds(self, pos: tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.height and 0 <= c < self.width

    def cell(self, pos: tuple[int, int]) -> Cell:
        if pos in self.victims:
            return Cell.VICTIM
        original = self.base_cells[pos]
        return Cell.FREE if original == Cell.VICTIM else original

    def _visible_positions(self) -> set[tuple[int, int]]:
        radius = self.scenario.observation_radius
        if radius is None:
            return set(self.base_cells)

        r0, c0 = self.position
        visible: set[tuple[int, int]] = set()
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if abs(dr) + abs(dc) <= radius:
                    pos = (r0 + dr, c0 + dc)
                    if self.in_bounds(pos):
                        visible.add(pos)
        return visible

    def perceive(self) -> Perception:
        visible = {pos: self.cell(pos) for pos in self._visible_positions()}
        current = self.cell(self.position)
        return Perception(
            step=self.step_count,
            position=self.position,
            battery=self.battery,
            max_battery=self.scenario.max_battery,
            visible_cells=visible,
            on_victim=current == Cell.VICTIM,
            on_charger=current == Cell.CHARGER,
            on_exit=current == Cell.EXIT,
            rescued=self.rescued,
            total_victims=self.total_victims,
        )

    def _consume_energy(self, amount: int) -> None:
        before = self.battery
        self.battery = max(0, self.battery - max(0, amount))
        self.energy_used += before - self.battery

    def _move(self, action: Action) -> tuple[bool, bool]:
        if self.rng.random() < self.scenario.move_fail_prob:
            self._consume_energy(1)
            return False, False

        dr, dc = MOVE_DELTA[action]
        target = (self.position[0] + dr, self.position[1] + dc)
        if not self.in_bounds(target) or self.cell(target) == Cell.WALL:
            self._consume_energy(1)
            return False, False

        self.position = target
        self._consume_energy(1)
        entered_hazard = self.cell(self.position) == Cell.HAZARD
        if entered_hazard:
            self.hazard_entries += 1
            self._consume_energy(self.scenario.hazard_extra_cost)
        return True, entered_hazard

    def step(self, action: Action) -> StepResult:
        if self.done:
            raise RuntimeError("O episódio já terminou.")

        if not isinstance(action, Action):
            action = Action(action)

        reward = -1.0
        info: dict = {}

        if action in MOVE_DELTA:
            moved, hazard = self._move(action)
            if not moved:
                reward -= 1.0
                info["movement_failed_or_blocked"] = True
            if hazard:
                reward -= 10.0
                info["entered_hazard"] = True

        elif action == Action.RESCUE:
            if self.position in self.victims:
                self.victims.remove(self.position)
                self.rescued += 1
                reward += 100.0
                info["rescued"] = True
            else:
                reward -= 2.0
                info["invalid_rescue"] = True

        elif action == Action.RECHARGE:
            if self.cell(self.position) == Cell.CHARGER:
                before = self.battery
                amount = self.scenario.recharge_amount
                self.battery = (
                    self.scenario.max_battery
                    if amount is None
                    else min(self.scenario.max_battery, self.battery + amount)
                )
                recovered = self.battery - before
                if recovered > 0:
                    info["recharged"] = recovered
                else:
                    reward -= 1.0
                    info["unnecessary_recharge"] = True
            else:
                reward -= 2.0
                info["invalid_recharge"] = True

        elif action == Action.WAIT:
            reward -= 0.5

        self.step_count += 1

        if not self.victims and self.cell(self.position) == Cell.EXIT:
            self.done = True
            self.success = True
            reward += 50.0
            info["success"] = True

        if self.battery <= 0 and not self.done:
            self.done = True
            self.failed = True
            reward -= 100.0
            info["battery_failure"] = True

        if self.step_count >= self.scenario.max_steps and not self.done:
            self.done = True
            info["timeout"] = True

        return StepResult(reward=reward, done=self.done, info=info)

    def metrics(self, total_reward: float) -> dict:
        return {
            "scenario": self.scenario.name,
            "seed": self.seed,
            "reward": total_reward,
            "steps": self.step_count,
            "rescued": self.rescued,
            "total_victims": self.total_victims,
            "energy_used": self.energy_used,
            "hazard_entries": self.hazard_entries,
            "success": int(self.success),
            "battery_end": self.battery,
        }

    def render(self) -> str:
        rows: list[str] = []
        for r in range(self.height):
            chars = []
            for c in range(self.width):
                pos = (r, c)
                chars.append("R" if pos == self.position else self.cell(pos).value)
            rows.append(" ".join(chars))

        status = (
            f"passo={self.step_count} | bateria={self.battery} | "
            f"resgatadas={self.rescued}/{self.total_victims}"
        )
        return "\n".join(rows) + "\n" + status
