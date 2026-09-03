from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class Action(str, Enum):
    NORTH = "NORTH"
    SOUTH = "SOUTH"
    EAST = "EAST"
    WEST = "WEST"
    RESCUE = "RESCUE"
    RECHARGE = "RECHARGE"
    WAIT = "WAIT"


class Cell(str, Enum):
    FREE = "."
    WALL = "#"
    VICTIM = "V"
    HAZARD = "H"
    CHARGER = "C"
    EXIT = "E"


@dataclass(frozen=True)
class Perception:
    step: int
    position: tuple[int, int]
    battery: int
    max_battery: int
    visible_cells: Mapping[tuple[int, int], Cell]
    on_victim: bool
    on_charger: bool
    on_exit: bool
    rescued: int
    total_victims: int

    def cell_at(self, position: tuple[int, int]) -> Cell | None:
        return self.visible_cells.get(position)


@dataclass(frozen=True)
class Transition:
    perception: Perception
    action: Action
    reward: float
    next_perception: Perception
    done: bool
