from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as file:
        return list(csv.DictReader(file))


METRICS = [
    "reward",
    "steps",
    "rescued",
    "energy_used",
    "hazard_entries",
    "success",
]


def series_label(row: dict) -> str:
    agent = row["agent"]
    epsilon = row.get("train_epsilon", "")
    if agent == "learning" and epsilon != "":
        return f"learning (epsilon={float(epsilon):g})"
    return agent


def evaluation_rows(rows: list[dict]) -> list[dict]:
    """Mantém compatibilidade com CSVs antigos, que não possuem `phase`."""
    return [row for row in rows if row.get("phase", "evaluation") == "evaluation"]


def grouped_stats(
    rows: list[dict], metric: str
) -> dict[tuple[str, str], tuple[float, float]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["scenario"], series_label(row))].append(float(row[metric]))
    return {
        key: (mean(values), pstdev(values) if len(values) > 1 else 0.0)
        for key, values in grouped.items()
    }


def plot_metric(rows: list[dict], metric: str, output: Path) -> None:
    grouped = grouped_stats(rows, metric)
    scenarios = sorted({key[0] for key in grouped})
    agents = sorted({key[1] for key in grouped})

    x = list(range(len(scenarios)))
    width = 0.8 / max(1, len(agents))
    fig, ax = plt.subplots(figsize=(10, 5))

    for index, agent in enumerate(agents):
        values = [grouped.get((scenario, agent), (0.0, 0.0))[0] for scenario in scenarios]
        deviations = [
            grouped.get((scenario, agent), (0.0, 0.0))[1]
            for scenario in scenarios
        ]
        positions = [value + index * width for value in x]
        ax.bar(
            positions,
            values,
            width=width,
            yerr=deviations,
            capsize=3,
            label=agent,
        )

    centers = [value + width * (len(agents) - 1) / 2 for value in x]
    ax.set_xticks(centers)
    ax.set_xticklabels(scenarios)
    ax.set_xlabel("Cenário")
    ax.set_ylabel(metric)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_learning_curve(rows: list[dict], output: Path) -> None:
    learning_rows = [
        row
        for row in rows
        if row["agent"] == "learning"
        and row.get("phase", "training") == "training"
    ]
    if not learning_rows:
        return

    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in learning_rows:
        grouped[
            (row["scenario"], row.get("train_epsilon", ""), int(row["episode"]))
        ].append(float(row["reward"]))

    fig, ax = plt.subplots(figsize=(10, 5))
    configurations = sorted({(key[0], key[1]) for key in grouped})
    for scenario, epsilon in configurations:
        episodes = sorted(
            key[2]
            for key in grouped
            if key[0] == scenario and key[1] == epsilon
        )
        averages = [mean(grouped[(scenario, epsilon, episode)]) for episode in episodes]
        deviations = [
            pstdev(grouped[(scenario, epsilon, episode)])
            if len(grouped[(scenario, epsilon, episode)]) > 1
            else 0.0
            for episode in episodes
        ]
        label = f"{scenario} (epsilon={float(epsilon):g})" if epsilon != "" else scenario
        ax.plot(episodes, averages, label=label)
        if any(deviations):
            ax.fill_between(
                episodes,
                [value - deviation for value, deviation in zip(averages, deviations)],
                [value + deviation for value, deviation in zip(averages, deviations)],
                alpha=0.15,
            )

    ax.set_xlabel("Episódio")
    ax.set_ylabel("Recompensa")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_summary(rows: list[dict], output: Path) -> None:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["scenario"], series_label(row))].append(row)

    summary: list[dict] = []
    for (scenario, agent), values in sorted(grouped.items()):
        result: dict[str, str | float | int] = {
            "scenario": scenario,
            "agent": agent,
            "samples": len(values),
        }
        for metric in METRICS:
            numbers = [float(row[metric]) for row in values]
            result[f"{metric}_mean"] = mean(numbers)
            result[f"{metric}_std"] = pstdev(numbers) if len(numbers) > 1 else 0.0
        for metric in ["q_states", "q_entries"]:
            numbers = [
                float(row[metric])
                for row in values
                if row.get(metric, "") != ""
            ]
            if numbers:
                result[f"{metric}_mean"] = mean(numbers)
                result[f"{metric}_std"] = (
                    pstdev(numbers) if len(numbers) > 1 else 0.0
                )
        summary.append(result)

    with output.open("w", newline="", encoding="utf-8") as file:
        fieldnames = list(dict.fromkeys(key for row in summary for key in row))
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results.csv")
    parser.add_argument("--output-dir", default="plots")
    args = parser.parse_args()

    rows = load_rows(args.input)
    if not rows:
        raise SystemExit("O arquivo de resultados está vazio.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparable_rows = evaluation_rows(rows)
    if not comparable_rows:
        raise SystemExit("Não há episódios de avaliação no arquivo de resultados.")

    for metric in METRICS:
        plot_metric(comparable_rows, metric, output_dir / f"{metric}.png")

    plot_learning_curve(rows, output_dir / "learning_curve.png")
    write_summary(comparable_rows, output_dir / "summary.csv")
    print(f"Gráficos salvos em: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
