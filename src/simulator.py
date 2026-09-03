from __future__ import annotations

from dataclasses import dataclass

from src.agents.base import Agent
from src.environment import RescueEnvironment
from src.types import Transition


@dataclass
class EpisodeResult:
    metrics: dict
    rewards: list[float]


def run_episode(
    env: RescueEnvironment,
    agent: Agent,
    render: bool = False,
    *,
    training: bool = True,
) -> EpisodeResult:
    agent.set_training(training)
    agent.reset()
    perception = env.reset()
    total_reward = 0.0
    rewards: list[float] = []

    if render:
        print(env.render())
        print("-" * 60)

    while not env.done:
        action = agent.act(perception)
        result = env.step(action)
        next_perception = env.perceive()

        transition = Transition(
            perception=perception,
            action=action,
            reward=result.reward,
            next_perception=next_perception,
            done=result.done,
        )
        agent.observe_transition(transition)

        total_reward += result.reward
        rewards.append(result.reward)
        perception = next_perception

        if render:
            print(f"Ação: {action.value} | recompensa: {result.reward:.1f}")
            print(env.render())
            print("-" * 60)

    metrics = env.metrics(total_reward)
    metrics["agent"] = agent.name
    metrics.update(agent.diagnostics())
    return EpisodeResult(metrics=metrics, rewards=rewards)
