from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    name: str
    grid: tuple[str, ...]
    start: tuple[int, int]
    max_battery: int
    max_steps: int
    observation_radius: int | None
    move_fail_prob: float = 0.0
    hazard_extra_cost: int = 2
    recharge_amount: int | None = None
    description: str = ""


SCENARIOS: dict[str, Scenario] = {
    "simple": Scenario(
        name="simple",
        grid=(
            "..#..V.",
            ".##....",
            "....#..",
            ".V.....",
            "...##..",
            "....C.E",
        ),
        start=(5, 0),
        max_battery=70,
        max_steps=140,
        observation_radius=None,
        description="Determinístico e completamente observável.",
    ),
    "partial": Scenario(
        name="partial",
        grid=(
            "..#..V.",
            ".##....",
            "....#..",
            ".V.....",
            "...##..",
            "....C.E",
        ),
        start=(5, 0),
        max_battery=70,
        max_steps=140,
        observation_radius=1,
        description="Parcialmente observável; memória tende a ser útil.",
    ),
    "risky": Scenario(
        name="risky",
        grid=(
            ".H..#V.",
            ".H#....",
            "...H#..",
            ".V.H...",
            ".##H...",
            "..C...E",
        ),
        start=(5, 0),
        max_battery=45,
        max_steps=160,
        observation_radius=1,
        hazard_extra_cost=3,
        description="Parcialmente observável, com perigos e bateria restrita.",
    ),
    "stochastic": Scenario(
        name="stochastic",
        grid=(
            ".H..#V.",
            ".H#....",
            "...H#..",
            ".V.H...",
            ".##H...",
            "..C...E",
        ),
        start=(5, 0),
        max_battery=45,
        max_steps=160,
        observation_radius=1,
        move_fail_prob=0.15,
        hazard_extra_cost=3,
        description="Parcialmente observável e estocástico.",
    ),
}


def get_scenario(name: str) -> Scenario:
    if name not in SCENARIOS:
        valid = ", ".join(sorted(SCENARIOS))
        raise ValueError(f"Cenário desconhecido: {name}. Opções: {valid}")
    return SCENARIOS[name]
