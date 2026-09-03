# Atividade Prática — Arquiteturas de Agentes Inteligentes

Toda a infraestrutura do ambiente, cenários, simulação, métricas e experimentação é fornecida. Você deve implementar apenas o arquivo:

`src/agents/student_agents.py`

Agentes obrigatórios:
1. `SimpleReflexAgent`
2. `ModelBasedAgent`
3. `LearningAgent`

Agentes opcionais:
4. `GoalBasedAgent`
5. `UtilityBasedAgent`

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Testar a infraestrutura

```bash
python -m src.run_episode --agent random --scenario simple --render
```

## Rodar os agentes

```bash
python -m src.run_episode --agent simple --scenario partial --render
python -m src.run_episode --agent model --scenario partial --render
python -m src.run_episode --agent learning --scenario stochastic --render
```

## Experimentos

```bash
python -m src.run_experiments --agents simple model learning --scenarios simple partial risky stochastic --episodes 100 --output results.csv
```

O comando acima treina o agente de aprendizado separadamente para
`epsilon = 0.01, 0.10, 0.30` e depois avalia as políticas sem atualizar a
Q-table. Por padrão, são usados 100 episódios de treinamento e 100 de avaliação.
Para obter estimativas mais estáveis, execute várias repetições independentes:

```bash
python -m src.run_experiments \
  --agents simple model learning \
  --scenarios simple partial risky stochastic \
  --train-episodes 100 \
  --episodes 100 \
  --epsilon 0.01 0.10 0.30 \
  --runs 10 \
  --output results.csv
```

Opções adicionais incluem `--alpha`, `--gamma`, `--seed` e
`--evaluation-epsilon`. As sementes de avaliação são iguais entre as
arquiteturas e não são reutilizadas no treinamento.

## Gráficos

```bash
python -m src.plot_results --input results.csv --output-dir plots
```

Além dos gráficos, o script produz `plots/summary.csv` com a média e
 o desvio-padrão das métricas, usando apenas os episódios de avaliação. A curva de
aprendizado utiliza apenas os episódios de treinamento.

## Verificação

```bash
python -m tests.check_submission
```
