from __future__ import annotations

from src.agents import (
    GoalBasedAgent,
    LearningAgent,
    ModelBasedAgent,
    RandomAgent,
    SimpleReflexAgent,
    UtilityBasedAgent,
)


def make_agent(
    name: str,
    seed: int = 0,
    *,
    alpha: float = 0.2,
    gamma: float = 0.95,
    epsilon: float = 0.15,
    evaluation_epsilon: float = 0.0,
):
    name = name.lower()
    if name == "random":
        return RandomAgent(seed=seed)
    if name == "simple":
        return SimpleReflexAgent(seed=seed)
    if name == "model":
        return ModelBasedAgent(seed=seed)
    if name == "learning":
        return LearningAgent(
            seed=seed,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            evaluation_epsilon=evaluation_epsilon,
        )
    if name == "goal":
        return GoalBasedAgent(seed=seed)
    if name == "utility":
        return UtilityBasedAgent(seed=seed)
    raise ValueError(
        f"Agente desconhecido: {name}. "
        "Opções: random, simple, model, learning, goal, utility."
    )
