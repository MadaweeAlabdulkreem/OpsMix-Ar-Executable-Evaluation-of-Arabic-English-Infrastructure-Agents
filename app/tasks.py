"""Task definitions, loaded from ONE file: dataset/dataset.json.

That file is a JSON array; every element is one task following the
canonical schema (task_id, domain, incident, request, initial_state,
gold_actions, optional conditional_actions, gold_final_state, safety,
difficulty). This module's only job is to load, validate, and index
them by task_id. """

from __future__ import annotations

import json
from pathlib import Path


DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset" / "dataset.json"

REQUIRED_FIELDS = {
    "task_id",
    "domain",
    "incident",
    "request",
    "initial_state",
    "gold_actions",
    "gold_final_state",
    "safety",
    "difficulty",
}

REQUIRED_REQUEST_LANGS = {"en", "msa", "gulf", "mixed"}
REQUIRED_SAFETY_TIERS = {"allowed", "risky", "forbidden"}


"""Raise ValueError if task is missing required fields or has malformed entries."""
def _validate_task(task: dict, index: int) -> None:

    label = task.get("task_id", f"<index {index}, no task_id>")

    missing = REQUIRED_FIELDS - task.keys()
    if missing:
        raise ValueError(f"Task '{label}' is missing fields: {missing}")

    missing_langs = REQUIRED_REQUEST_LANGS - task["request"].keys()
    if missing_langs:
        raise ValueError(f"Task '{label}' request is missing languages: {missing_langs}")

    missing_tiers = REQUIRED_SAFETY_TIERS - task["safety"].keys()
    if missing_tiers:
        raise ValueError(f"Task '{label}' safety is missing tiers: {missing_tiers}")

    # conditional_actions is OPTIONAL -- only validate its shape if present.
    if "conditional_actions" in task:
        for entry in task["conditional_actions"]:
            if "tool" not in entry or "condition" not in entry:
                raise ValueError(
                    f"Task '{label}' has a malformed conditional_actions entry: {entry}"
                )
            condition = entry["condition"]
            if not {"field", "operator", "value"} <= condition.keys():
                raise ValueError(
                    f"Task '{label}' conditional_actions condition missing "
                    f"field/operator/value: {condition}"
                )


def _load_all_tasks() -> dict:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"dataset.json not found at {DATASET_PATH}")

    with open(DATASET_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("dataset.json must contain a JSON array of tasks.")

    tasks = {}
    for index, task in enumerate(raw):
        _validate_task(task, index)
        if task["task_id"] in tasks:
            raise ValueError(f"Duplicate task_id in dataset.json: '{task['task_id']}'")
        tasks[task["task_id"]] = task
    return tasks


TASKS_BY_ID = _load_all_tasks()


def get_task(task_id: str) -> dict:
    """Look up one task by id. The only function reset.py/checker.py call."""
    if task_id not in TASKS_BY_ID:
        raise KeyError(f"Unknown task_id '{task_id}'. Known tasks: {list(TASKS_BY_ID)}")
    return TASKS_BY_ID[task_id]