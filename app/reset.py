"""reset(task_id): put the shared sandbox state into a task's starting
condition, ready for an agent to act on it.

Used like: `from app.reset import reset; reset("disk_full_001")`
before handing a task's prompt to an LLM agent.
"""

from __future__ import annotations

from app.state import state, _initial_state
from app.tasks import get_task


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge `overrides` into `base`, returning the result."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def reset(task_id: str) -> dict:
    """Reset the shared sandbox state to the initial state of the given task."""
    task = get_task(task_id)

    fresh = _initial_state()
    _deep_merge(fresh, task["initial_state"])

    state.clear()
    state.update(fresh)

    return state