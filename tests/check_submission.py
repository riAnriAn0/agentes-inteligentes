from __future__ import annotations

import copy
import inspect
import traceback

from src.agent_factory import make_agent
from src.environment import RescueEnvironment
from src.scenarios import get_scenario
from src.simulator import run_episode
from src.types import Action


MANDATORY = ["simple", "model", "learning"]
SCENARIOS = ["simple", "partial", "risky", "stochastic"]


def _student_state(agent) -> dict:
    """Estado persistente relevante, desconsiderando RNG e modo de execução."""
    return {
        key: copy.deepcopy(value)
        for key, value in vars(agent).items()
        if key not in {"rng", "training"}
    }


def _check_no_environment_access(agent) -> str | None:
    module = inspect.getmodule(type(agent))
    source = inspect.getsource(module) if module is not None else ""
    forbidden = [
        "from src.environment",
        "import src.environment",
        "from src.scenarios",
        "import src.scenarios",
        "SCENARIOS",
        "get_scenario",
    ]
    found = [token for token in forbidden if token in source]
    if found:
        return f"acesso proibido ao ambiente/cenários: {', '.join(found)}."
    return None


def check_agent(name: str) -> tuple[bool, str]:
    agent = make_agent(name, seed=123)

    try:
        access_error = _check_no_environment_access(agent)
        if access_error:
            return False, f"{name}: {access_error}"

        scenario = get_scenario("simple")
        env = RescueEnvironment(scenario, seed=123)
        perception = env.reset()
        state_before = _student_state(agent) if name == "simple" else None
        action = agent.act(perception)
        if not isinstance(action, Action):
            return False, f"{name}: act() deve retornar Action."
        if name == "simple" and _student_state(agent) != state_before:
            return False, f"{name}: act() modificou estado persistente proibido."

        for index, scenario_name in enumerate(SCENARIOS):
            env = RescueEnvironment(get_scenario(scenario_name), seed=123 + index)
            run_episode(env, agent, render=False, training=True)

        if name == "model":
            agent.reset()
            if getattr(agent, "known_map", None):
                return False, "model: reset() não limpou known_map."
            if getattr(agent, "visit_count", None):
                return False, "model: reset() não limpou visit_count."

        if name == "learning" and hasattr(agent, "q"):
            learned = dict(agent.q)
            agent.reset()
            if dict(agent.q) != learned:
                return False, "learning: reset() apagou ou modificou a Q-table."

            env = RescueEnvironment(get_scenario("stochastic"), seed=999)
            run_episode(env, agent, render=False, training=False)
            if dict(agent.q) != learned:
                return False, "learning: a avaliação modificou a Q-table."
    except NotImplementedError as exc:
        return False, f"{name}: ainda não implementado ({exc})."
    except Exception:
        return False, f"{name}: erro durante execução:\n{traceback.format_exc()}"

    return True, f"{name}: interface, restrições básicas e quatro cenários OK."


def check_environment() -> tuple[bool, str]:
    env = RescueEnvironment(get_scenario("risky"), seed=0)
    env.step(Action.EAST)
    env.step(Action.EAST)
    env.step(Action.RECHARGE)
    result = env.step(Action.RECHARGE)
    if result.reward > 0:
        return False, "ambiente: recarga desnecessária ainda gera recompensa positiva."
    return True, "ambiente: recarga não permite acumular recompensa indevida."


def main() -> None:
    failures = 0
    ok, message = check_environment()
    print(("[OK] " if ok else "[ERRO] ") + message + "\n")
    failures += int(not ok)
    print("Verificando agentes obrigatórios...\n")
    for name in MANDATORY:
        ok, message = check_agent(name)
        print(("[OK] " if ok else "[ERRO] ") + message)
        failures += int(not ok)

    if failures:
        raise SystemExit(f"\n{failures} agente(s) precisam de correção.")
    print("\nTodos os testes básicos passaram.")


if __name__ == "__main__":
    main()
