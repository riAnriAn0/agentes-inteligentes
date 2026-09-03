from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.agent_factory import make_agent
from src.environment import RescueEnvironment
from src.scenarios import get_scenario
from src.simulator import run_episode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["simple", "model", "learning"],
        choices=["random", "simple", "model", "learning", "goal", "utility"],
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["simple", "partial", "risky", "stochastic"],
        choices=["simple", "partial", "risky", "stochastic"],
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--train-episodes",
        type=int,
        default=100,
        help="Episódios de treinamento para cada agente de aprendizado.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Execuções independentes de cada configuração.",
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument(
        "--epsilon",
        type=float,
        nargs="+",
        default=[0.01, 0.10, 0.30],
        help="Valores de epsilon usados no treinamento.",
    )
    parser.add_argument(
        "--evaluation-epsilon",
        type=float,
        default=0.0,
        help="Exploração durante a avaliação; zero avalia a política gulosa.",
    )
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()

    if args.episodes <= 0 or args.train_episodes < 0 or args.runs <= 0:
        parser.error("episodes e runs devem ser positivos; train-episodes não pode ser negativo.")
    if not 0.0 < args.alpha <= 1.0:
        parser.error("alpha deve pertencer ao intervalo (0, 1].")
    if not 0.0 <= args.gamma <= 1.0:
        parser.error("gamma deve pertencer ao intervalo [0, 1].")
    if any(not 0.0 <= value <= 1.0 for value in args.epsilon):
        parser.error("todo epsilon deve pertencer ao intervalo [0, 1].")
    if not 0.0 <= args.evaluation_epsilon <= 1.0:
        parser.error("evaluation-epsilon deve pertencer ao intervalo [0, 1].")

    rows: list[dict] = []

    for scenario_name in args.scenarios:
        scenario = get_scenario(scenario_name)

        for agent_name in args.agents:
            epsilon_values = args.epsilon if agent_name == "learning" else [None]

            for training_epsilon in epsilon_values:
                for run in range(args.runs):
                    agent_seed = args.seed + run
                    agent = make_agent(
                        agent_name,
                        seed=agent_seed,
                        alpha=args.alpha,
                        gamma=args.gamma,
                        epsilon=(
                            training_epsilon
                            if training_epsilon is not None
                            else args.epsilon[0]
                        ),
                        evaluation_epsilon=args.evaluation_epsilon,
                    )

                    if agent_name == "learning":
                        for episode in range(args.train_episodes):
                            # As sementes de treinamento não reaparecem na avaliação.
                            episode_seed = args.seed + run * 100_000 + episode
                            env = RescueEnvironment(scenario, seed=episode_seed)
                            result = run_episode(
                                env, agent, render=False, training=True
                            )
                            row = dict(result.metrics)
                            row.update(
                                phase="training",
                                run=run,
                                episode=episode,
                                train_epsilon=training_epsilon,
                                evaluation_epsilon=args.evaluation_epsilon,
                                alpha=args.alpha,
                                gamma=args.gamma,
                            )
                            rows.append(row)
                            print(
                                f"{scenario_name:10s} | {agent_name:9s} | "
                                f"eps={training_epsilon:.2f} | run={run + 1} | "
                                f"treino {episode + 1:3d}/{args.train_episodes}"
                            )

                    for episode in range(args.episodes):
                        # Todas as arquiteturas recebem as mesmas sementes de
                        # avaliação, separadas das sementes de treinamento.
                        episode_seed = (
                            args.seed + 10_000_000 + run * 100_000 + episode
                        )
                        env = RescueEnvironment(scenario, seed=episode_seed)
                        result = run_episode(
                            env, agent, render=False, training=False
                        )
                        row = dict(result.metrics)
                        row.update(
                            phase="evaluation",
                            run=run,
                            episode=episode,
                            train_epsilon=(
                                training_epsilon if training_epsilon is not None else ""
                            ),
                            evaluation_epsilon=(
                                args.evaluation_epsilon
                                if agent_name == "learning"
                                else ""
                            ),
                            alpha=args.alpha if agent_name == "learning" else "",
                            gamma=args.gamma if agent_name == "learning" else "",
                        )
                        rows.append(row)
                        label = (
                            f"eps={training_epsilon:.2f} | "
                            if training_epsilon is not None
                            else ""
                        )
                        print(
                            f"{scenario_name:10s} | {agent_name:9s} | "
                            f"{label}run={run + 1} | "
                            f"avaliação {episode + 1:3d}/{args.episodes}"
                        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResultados salvos em: {output.resolve()}")


if __name__ == "__main__":
    main()
