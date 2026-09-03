from __future__ import annotations

import argparse

from src.agent_factory import make_agent
from src.environment import RescueEnvironment
from src.scenarios import get_scenario
from src.simulator import run_episode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        default="random",
        choices=["random", "simple", "model", "learning", "goal", "utility"],
    )
    parser.add_argument(
        "--scenario",
        default="simple",
        choices=["simple", "partial", "risky", "stochastic"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    env = RescueEnvironment(get_scenario(args.scenario), seed=args.seed)
    agent = make_agent(args.agent, seed=args.seed)
    result = run_episode(env, agent, render=args.render)

    print("\nMétricas finais")
    print("=" * 60)
    for key, value in result.metrics.items():
        print(f"{key:20s}: {value}")


if __name__ == "__main__":
    main()
