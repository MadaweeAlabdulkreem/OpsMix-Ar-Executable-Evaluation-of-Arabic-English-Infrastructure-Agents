"""
app/evaluate.py

OpsMix-Ar evaluation harness.

Purpose
-------
Evaluate externally generated LLM tool calls against the real
Tiny Infra Service sandbox.

Pipeline
--------
Qwen predictions
        |
        v
evaluate.py
        |
        v
HTTP API (main.py)
        |
        +--> reset task
        |
        +--> execute predicted tool calls
        |
        +--> collect /history
        |
        +--> collect /state
        |
        v
checker.py
        |
        v
evaluation_results.json
evaluation_summary.json

Qwen runs separately, for example in Google Colab.
The output from Qwen is saved as JSON and passed to this evaluator.

The evaluator:
1. Resets the sandbox.
2. Executes Qwen's predicted tool calls through HTTP.
3. Collects the REAL sandbox state and history.
4. Synchronizes those values into the local checker state.
5. Runs the existing checker.
6. Computes task-level and aggregate metrics.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import requests

from app.checker import check
from app.state import state as LOCAL_STATE
from app.tasks import TASKS_BY_ID, get_task


# ============================================================
# Configuration
# ============================================================

BASE_URL = os.getenv(
    "TINY_INFRA_BASE_URL",
    "http://127.0.0.1:8000",
)

DEFAULT_TIMEOUT = 30.0

RESULTS_DIR = Path("results")

LANGUAGES = ("en", "msa", "gulf", "mixed")


# ============================================================
# Tool -> HTTP endpoint mapping
# ============================================================

TOOL_ENDPOINTS: dict[str, tuple[str, str]] = {
    "check_disk": ("GET", "/check_disk"),
    "clear_cache": ("POST", "/clear_cache"),
    "restart_service": ("POST", "/restart_service"),
    "rotate_api_key": ("POST", "/rotate_api_key"),
    "scale_replicas": ("POST", "/scale_replicas"),
    "get_metrics": ("GET", "/get_metrics"),
    "rollback_deploy": ("POST", "/rollback_deploy"),
    "get_logs": ("GET", "/get_logs"),
    "kill_process": ("POST", "/kill_process"),
    "set_config": ("POST", "/set_config"),
}


SENSITIVE_KEYS = {
    "api_key",
}


class EvaluatorError(Exception):
    """Evaluator-level failure.

    Examples:
    - sandbox server is unreachable
    - reset endpoint failed
    - state/history cannot be collected

    A normal HTTP 400/404/500 returned by a tool is NOT an
    evaluator failure. It is a valid model/tool-call result.
    """


# ============================================================
# Sanitization
# ============================================================

def _sanitize(obj: Any) -> Any:
    """
    Recursively redact sensitive values.

    This creates a new object and never modifies the live state.
    """

    if isinstance(obj, dict):
        result = {}

        for key, value in obj.items():
            if key in SENSITIVE_KEYS and isinstance(value, str):
                result[key] = "***REDACTED***"
            else:
                result[key] = _sanitize(value)

        return result

    if isinstance(obj, list):
        return [_sanitize(item) for item in obj]

    return obj


# ============================================================
# HTTP helpers
# ============================================================

def _build_query_params(
    tool: str,
    args: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert the model's tool-call arguments into query parameters.

    main.py defines tool arguments as normal FastAPI parameters,
    so both GET and POST calls use query parameters.

    Special case:
    get_logs(limit=None) must OMIT the limit parameter entirely.
    """

    params = dict(args or {})

    if tool == "get_logs":
        if params.get("limit") is None:
            params.pop("limit", None)

    return params


def call_tool(
    session: requests.Session,
    tool: str,
    args: dict[str, Any],
    base_url: str = BASE_URL,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """
    Execute ONE predicted tool call through the HTTP sandbox.

    HTTP errors such as 400/404/500 are recorded as model/tool
    execution results rather than crashing the evaluator.
    """

    if not isinstance(tool, str):
        return {
            "tool": tool,
            "args": args,
            "status_code": None,
            "ok": False,
            "response": {
                "detail": "Tool name must be a string."
            },
        }

    if not isinstance(args, dict):
        return {
            "tool": tool,
            "args": args,
            "status_code": None,
            "ok": False,
            "response": {
                "detail": "Tool args must be an object."
            },
        }

    if tool not in TOOL_ENDPOINTS:
        return {
            "tool": tool,
            "args": args,
            "status_code": None,
            "ok": False,
            "response": {
                "detail": f"Unknown tool '{tool}'."
            },
        }

    method, path = TOOL_ENDPOINTS[tool]

    url = base_url.rstrip("/") + path

    params = _build_query_params(
        tool=tool,
        args=args,
    )

    try:
        if method == "GET":
            response = session.get(
                url,
                params=params,
                timeout=timeout,
            )

        elif method == "POST":
            response = session.post(
                url,
                params=params,
                timeout=timeout,
            )

        else:
            raise EvaluatorError(
                f"Unsupported HTTP method '{method}' "
                f"for tool '{tool}'."
            )

    except requests.exceptions.RequestException as exc:
        raise EvaluatorError(
            f"Could not reach sandbox at {url}: {exc}"
        ) from exc

    try:
        body = response.json()

    except ValueError:
        body = {
            "detail": response.text
        }

    return {
        "tool": tool,
        "args": args,
        "status_code": response.status_code,
        "ok": response.ok,
        "response": body,
    }


# ============================================================
# Sandbox endpoints
# ============================================================

def reset_task_http(
    session: requests.Session,
    task_id: str,
    base_url: str = BASE_URL,
    timeout: float = DEFAULT_TIMEOUT,
) -> None:
    """
    Reset the real FastAPI sandbox to the task's initial state.
    """

    url = (
        base_url.rstrip("/")
        + f"/reset/{task_id}"
    )

    try:
        response = session.post(
            url,
            timeout=timeout,
        )

    except requests.exceptions.RequestException as exc:
        raise EvaluatorError(
            f"Could not reach reset endpoint: {exc}"
        ) from exc

    if not response.ok:
        raise EvaluatorError(
            f"Reset failed for task '{task_id}': "
            f"HTTP {response.status_code} - {response.text}"
        )


def get_history_http(
    session: requests.Session,
    base_url: str = BASE_URL,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[dict[str, Any]]:
    """
    Retrieve the REAL sandbox history.
    """

    url = (
        base_url.rstrip("/")
        + "/history"
    )

    try:
        response = session.get(
            url,
            timeout=timeout,
        )
        response.raise_for_status()

    except requests.exceptions.RequestException as exc:
        raise EvaluatorError(
            f"Could not retrieve /history: {exc}"
        ) from exc

    try:
        history = response.json()

    except ValueError as exc:
        raise EvaluatorError(
            "The /history endpoint did not return valid JSON."
        ) from exc

    if not isinstance(history, list):
        raise EvaluatorError(
            "/history must return a JSON list."
        )

    return history


def get_state_http(
    session: requests.Session,
    base_url: str = BASE_URL,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """
    Retrieve the REAL sandbox state.
    """

    url = (
        base_url.rstrip("/")
        + "/state"
    )

    try:
        response = session.get(
            url,
            timeout=timeout,
        )
        response.raise_for_status()

    except requests.exceptions.RequestException as exc:
        raise EvaluatorError(
            f"Could not retrieve /state: {exc}"
        ) from exc

    try:
        sandbox_state = response.json()

    except ValueError as exc:
        raise EvaluatorError(
            "The /state endpoint did not return valid JSON."
        ) from exc

    if not isinstance(sandbox_state, dict):
        raise EvaluatorError(
            "/state must return a JSON object."
        )

    return sandbox_state


# ============================================================
# Synchronize remote sandbox with local checker
# ============================================================

def _sync_checker_state(
    remote_state: dict[str, Any],
) -> dict[str, Any]:
    """
    Synchronize the state returned by the REAL FastAPI server
    into the local state object used by checker.py.

    Why this is necessary
    ---------------------
    evaluate.py and uvicorn/main.py normally run in separate
    Python processes.

    Therefore:

        FastAPI state != evaluator process state

    unless we explicitly synchronize them.

    The checker itself remains the source of truth for grading.
    We only provide it with the actual state/history collected
    from the sandbox.
    """

    previous_state = copy.deepcopy(LOCAL_STATE)

    LOCAL_STATE.clear()
    LOCAL_STATE.update(
        copy.deepcopy(remote_state)
    )

    return previous_state


def _restore_checker_state(
    previous_state: dict[str, Any],
) -> None:
    """
    Restore the evaluator process's original local state.
    """

    LOCAL_STATE.clear()
    LOCAL_STATE.update(
        copy.deepcopy(previous_state)
    )


def _run_checker_against_remote_state(
    task_id: str,
    remote_state: dict[str, Any],
) -> dict[str, Any]:
    """
    Run the existing checker against the REAL sandbox state.

    We do not reimplement checker.py logic here.
    """

    previous_state = _sync_checker_state(
        remote_state
    )

    try:
        return check(task_id)

    finally:
        _restore_checker_state(
            previous_state
        )


# ============================================================
# Prediction validation
# ============================================================

def _validate_tool_calls(
    calls: Any,
    task_id: str,
) -> list[dict[str, Any]]:
    """
    Validate the JSON representation of Qwen's predicted calls.

    Expected:

    [
        {
            "tool": "check_disk",
            "args": {}
        },
        {
            "tool": "clear_cache",
            "args": {}
        }
    ]
    """

    if not isinstance(calls, list):
        raise ValueError(
            f"Predictions for '{task_id}' must be a list."
        )

    validated = []

    for index, call in enumerate(calls):

        if not isinstance(call, dict):
            raise ValueError(
                f"Invalid call at index {index} "
                f"for task '{task_id}': {call!r}"
            )

        tool = call.get("tool")

        args = call.get(
            "args",
            {},
        )

        if not isinstance(tool, str):
            raise ValueError(
                f"Invalid tool at index {index} "
                f"for task '{task_id}'."
            )

        if not isinstance(args, dict):
            raise ValueError(
                f"Args must be an object for "
                f"'{tool}' in task '{task_id}'."
            )

        validated.append(
            {
                "tool": tool,
                "args": args,
            }
        )

    return validated


def load_predictions(
    path: str,
) -> dict[str, list[dict[str, Any]]]:
    """
    Load Qwen predictions.

    Expected format:

    {
        "task_id_001": [
            {
                "tool": "check_disk",
                "args": {}
            }
        ],

        "task_id_002": [
            {
                "tool": "get_metrics",
                "args": {
                    "service": "api"
                }
            }
        ]
    }
    """

    prediction_path = Path(path)

    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {prediction_path}"
        )

    with prediction_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        predictions = json.load(file)

    if not isinstance(predictions, dict):
        raise ValueError(
            "Prediction JSON must be an object "
            "keyed by task_id."
        )

    validated = {}

    for task_id, calls in predictions.items():

        validated[task_id] = _validate_tool_calls(
            calls,
            task_id,
        )

    return validated


# ============================================================
# Tool / Argument Metrics
# ============================================================

def _tool_and_argument_metrics(
    predicted_calls: list[dict[str, Any]],
    gold_actions: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Calculate tool-selection and argument accuracy.

    We compare the first N predicted calls against the first N
    gold actions.

    This keeps the metric interpretable at the step level.

    Extra predicted calls are counted as incorrect selections
    rather than ignored.
    """

    gold_count = len(gold_actions)
    predicted_count = len(predicted_calls)

    if gold_count == 0:

        return {
            "tool_selection_correct": 0,
            "tool_selection_total": predicted_count,
            "tool_selection_accuracy": (
                100.0
                if predicted_count == 0
                else 0.0
            ),

            "argument_correct": 0,
            "argument_total": predicted_count,
            "argument_accuracy": (
                100.0
                if predicted_count == 0
                else 0.0
            ),
        }

    comparison_count = max(
        gold_count,
        predicted_count,
    )

    tool_correct = 0
    argument_correct = 0

    for index in range(comparison_count):

        predicted = (
            predicted_calls[index]
            if index < predicted_count
            else None
        )

        gold = (
            gold_actions[index]
            if index < gold_count
            else None
        )

        if predicted is None:
            continue

        if gold is None:
            # Extra model action.
            continue

        predicted_tool = predicted.get("tool")
        gold_tool = gold.get("tool")

        predicted_args = predicted.get(
            "args",
            {},
        )

        gold_args = gold.get(
            "args",
            {},
        )

        if predicted_tool == gold_tool:
            tool_correct += 1

            if predicted_args == gold_args:
                argument_correct += 1

    tool_total = comparison_count
    argument_total = comparison_count

    return {
        "tool_selection_correct": tool_correct,
        "tool_selection_total": tool_total,
        "tool_selection_accuracy": (
            round(
                100.0 * tool_correct / tool_total,
                2,
            )
            if tool_total
            else 0.0
        ),

        "argument_correct": argument_correct,
        "argument_total": argument_total,
        "argument_accuracy": (
            round(
                100.0 * argument_correct / argument_total,
                2,
            )
            if argument_total
            else 0.0
        ),
    }


# ============================================================
# Action Order Correctness + Precision / Recall
# ============================================================

def _longest_common_subsequence_len(
    a: list[str],
    b: list[str],
) -> int:
    """
    Standard LCS length over two sequences of tool names.

    Used to score how well the predicted call *order* matches the
    gold order, independent of extra/missing calls.
    """

    n, m = len(a), len(b)

    if n == 0 or m == 0:
        return 0

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[n][m]


def _order_and_set_metrics(
    predicted_calls: list[dict[str, Any]],
    gold_actions: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Action Order Correctness:
        - order_exact_match: True if predicted tool-name sequence is
          identical, position by position, to the gold sequence.
        - order_score: LCS(predicted, gold) / len(gold), a partial-credit
          measure that tolerates extra/missing calls elsewhere in the
          sequence while still rewarding correct relative ordering.

    Precision / Recall (multiset over tool names, order-independent):
        - precision: of the tools the model called, how many were
          actually required (accounting for repeats).
        - recall: of the tools required, how many did the model call.
    """

    predicted_tools = [c.get("tool") for c in predicted_calls]
    gold_tools = [g.get("tool") for g in gold_actions]

    order_exact_match = predicted_tools == gold_tools

    lcs_len = _longest_common_subsequence_len(predicted_tools, gold_tools)

    order_score = (
        round(100.0 * lcs_len / len(gold_tools), 2)
        if gold_tools
        else (100.0 if not predicted_tools else 0.0)
    )

    # Multiset precision/recall on tool names.
    remaining_gold = list(gold_tools)
    true_positives = 0

    for tool in predicted_tools:
        if tool in remaining_gold:
            remaining_gold.remove(tool)
            true_positives += 1

    precision = (
        round(100.0 * true_positives / len(predicted_tools), 2)
        if predicted_tools
        else (100.0 if not gold_tools else 0.0)
    )

    recall = (
        round(100.0 * true_positives / len(gold_tools), 2)
        if gold_tools
        else (100.0 if not predicted_tools else 0.0)
    )

    return {
        "order_exact_match": order_exact_match,
        "order_score": order_score,
        "precision": precision,
        "recall": recall,
    }


# ============================================================
# Single Task Evaluation
# ============================================================

def evaluate_task(
    session: requests.Session,
    task_id: str,
    predicted_calls: list[dict[str, Any]],
    language: str,
    base_url: str = BASE_URL,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """
    Evaluate ONE task.

    Pipeline:

        reset
          ↓
        Qwen predicted calls
          ↓
        actual HTTP execution
          ↓
        collect history/state
          ↓
        checker
          ↓
        metrics
    """

    if task_id not in TASKS_BY_ID:
        raise EvaluatorError(
            f"Unknown task_id '{task_id}'."
        )

    task = get_task(task_id)

    result: dict[str, Any] = {
        "task_id": task_id,
        "language": language,

        "domain": task.get(
            "domain",
            "",
        ),

        "difficulty": task.get(
            "difficulty",
            "",
        ),

        "passed": False,

        "gold_actions_correct": False,
        "state_match": False,

        "conditional_violations": [],

        "forbidden_action": False,
        "risky_action": False,
        "unexpected_action": False,

        "forbidden_calls": [],
        "risky_calls": [],
        "unexpected_calls": [],

        "predicted_calls": predicted_calls,
        "executed_calls": [],

        "history": [],
        "final_state": {},

        "execution_errors": [],

        "tool_selection_correct": 0,
        "tool_selection_total": 0,
        "tool_selection_accuracy": 0.0,

        "argument_correct": 0,
        "argument_total": 0,
        "argument_accuracy": 0.0,

        "order_exact_match": False,
        "order_score": 0.0,

        "precision": 0.0,
        "recall": 0.0,
    }

    # --------------------------------------------------------
    # 1. Reset
    # --------------------------------------------------------

    try:
        reset_task_http(
            session=session,
            task_id=task_id,
            base_url=base_url,
            timeout=timeout,
        )

    except Exception as exc:
        result["execution_errors"].append(
            f"Reset error: {exc}"
        )

        return result

    # --------------------------------------------------------
    # 2. Tool / argument metrics
    # --------------------------------------------------------

    gold_actions = task.get(
        "gold_actions",
        [],
    )

    metric_result = _tool_and_argument_metrics(
        predicted_calls=predicted_calls,
        gold_actions=gold_actions,
    )

    result.update(
        metric_result
    )

    order_result = _order_and_set_metrics(
        predicted_calls=predicted_calls,
        gold_actions=gold_actions,
    )

    result.update(
        order_result
    )

    # --------------------------------------------------------
    # 3. Execute Qwen's calls
    # --------------------------------------------------------

    for index, call in enumerate(
        predicted_calls
    ):

        try:
            execution = call_tool(
                session=session,
                tool=call["tool"],
                args=call["args"],
                base_url=base_url,
                timeout=timeout,
            )

            result["executed_calls"].append(
                execution
            )

        except EvaluatorError as exc:

            result["execution_errors"].append(
                f"Call {index}: {exc}"
            )

            # Server-level failure.
            # Do not continue trying to reach a dead server.
            break

        except Exception as exc:

            result["execution_errors"].append(
                f"Call {index}: {type(exc).__name__}: {exc}"
            )

    # --------------------------------------------------------
    # 4. Collect actual state + history
    # --------------------------------------------------------

    try:

        remote_history = get_history_http(
            session=session,
            base_url=base_url,
            timeout=timeout,
        )

        remote_state = get_state_http(
            session=session,
            base_url=base_url,
            timeout=timeout,
        )

        result["history"] = _sanitize(
            remote_history
        )

        result["final_state"] = _sanitize(
            remote_state
        )

    except Exception as exc:

        result["execution_errors"].append(
            f"State/history collection error: {exc}"
        )

        return result

    # --------------------------------------------------------
    # 5. Run canonical checker
    # --------------------------------------------------------

    try:

        graded = _run_checker_against_remote_state(
            task_id=task_id,
            remote_state=remote_state,
        )

        result["passed"] = graded[
            "passed"
        ]

        result["gold_actions_correct"] = graded[
            "gold_actions_correct"
        ]

        result["state_match"] = graded[
            "state_match"
        ]

        result["conditional_violations"] = graded[
            "conditional_violations"
        ]

        safety = graded[
            "safety"
        ]

        result["forbidden_action"] = safety[
            "forbidden_action"
        ]

        result["forbidden_calls"] = safety[
            "forbidden_calls"
        ]

        result["risky_action"] = safety[
            "risky_action"
        ]

        result["risky_calls"] = safety[
            "risky_calls"
        ]

        result["unexpected_action"] = safety[
            "unexpected_action"
        ]

        result["unexpected_calls"] = safety[
            "unexpected_calls"
        ]

    except Exception as exc:

        result["execution_errors"].append(
            f"Checker error: "
            f"{type(exc).__name__}: {exc}"
        )

    return result


# ============================================================
# Batch evaluation
# ============================================================

def evaluate_tasks(
    predictions: dict[str, list[dict[str, Any]]],
    language: str,
    base_url: str = BASE_URL,
    timeout: float = DEFAULT_TIMEOUT,
    selected_task_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Evaluate multiple tasks.

    Every task is independently reset.
    """

    if selected_task_ids is None:
        task_ids = list(
            predictions.keys()
        )

    else:
        task_ids = selected_task_ids

    results = []

    with requests.Session() as session:

        for index, task_id in enumerate(
            task_ids,
            start=1,
        ):

            print(
                f"[{index}/{len(task_ids)}] "
                f"Evaluating {task_id} "
                f"({language})..."
            )

            predicted_calls = predictions.get(
                task_id,
                [],
            )

            try:

                result = evaluate_task(
                    session=session,
                    task_id=task_id,
                    predicted_calls=predicted_calls,
                    language=language,
                    base_url=base_url,
                    timeout=timeout,
                )

            except Exception as exc:

                result = {
                    "task_id": task_id,
                    "language": language,
                    "passed": False,
                    "execution_errors": [
                        f"{type(exc).__name__}: {exc}"
                    ],
                }

            results.append(
                result
            )

    return results


# ============================================================
# Aggregate metrics
# ============================================================

def _percentage(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return round(
        100.0 * numerator / denominator,
        2,
    )


def summarize(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Aggregate evaluation metrics.

    Metrics:

    SR  = Task Success Rate
    SMR = State Match Rate
    TSA = Tool Selection Accuracy
    ArgA = Argument Accuracy

    Also reports safety violations and execution errors.
    """

    total = len(results)

    passed = sum(
        1
        for result in results
        if result.get("passed", False)
    )

    state_correct = sum(
        1
        for result in results
        if result.get(
            "state_match",
            False,
        )
    )

    forbidden = sum(
        1
        for result in results
        if result.get(
            "forbidden_action",
            False,
        )
    )

    risky = sum(
        1
        for result in results
        if result.get(
            "risky_action",
            False,
        )
    )

    unexpected = sum(
        1
        for result in results
        if result.get(
            "unexpected_action",
            False,
        )
    )

    execution_errors = sum(
        1
        for result in results
        if result.get(
            "execution_errors"
        )
    )

    tool_correct = sum(
        result.get(
            "tool_selection_correct",
            0,
        )
        for result in results
    )

    tool_total = sum(
        result.get(
            "tool_selection_total",
            0,
        )
        for result in results
    )

    argument_correct = sum(
        result.get(
            "argument_correct",
            0,
        )
        for result in results
    )

    argument_total = sum(
        result.get(
            "argument_total",
            0,
        )
        for result in results
    )

    # Safety Violation Rate: a task counts as a violation if it triggered
    # ANY forbidden, risky, or unexpected action.
    safety_violations = sum(
        1
        for result in results
        if result.get("forbidden_action", False)
        or result.get("risky_action", False)
        or result.get("unexpected_action", False)
    )

    order_exact_matches = sum(
        1
        for result in results
        if result.get("order_exact_match", False)
    )

    order_score_sum = sum(
        result.get("order_score", 0.0)
        for result in results
    )

    precision_sum = sum(
        result.get("precision", 0.0)
        for result in results
    )

    recall_sum = sum(
        result.get("recall", 0.0)
        for result in results
    )

    return {
        "total_tasks": total,

        "passed_tasks": passed,

        "failed_tasks": (
            total - passed
        ),

        "task_success_rate": _percentage(
            passed,
            total,
        ),

        "state_match_rate": _percentage(
            state_correct,
            total,
        ),

        "tool_selection_accuracy": _percentage(
            tool_correct,
            tool_total,
        ),

        "argument_accuracy": _percentage(
            argument_correct,
            argument_total,
        ),

        "tool_selection_correct": tool_correct,
        "tool_selection_total": tool_total,

        "argument_correct": argument_correct,
        "argument_total": argument_total,

        "forbidden_tasks": forbidden,
        "risky_tasks": risky,
        "unexpected_tasks": unexpected,

        "safety_violation_tasks": safety_violations,
        "safety_violation_rate": _percentage(
            safety_violations,
            total,
        ),

        "order_exact_match_tasks": order_exact_matches,
        "order_exact_match_rate": _percentage(
            order_exact_matches,
            total,
        ),
        "avg_order_score": (
            round(order_score_sum / total, 2)
            if total
            else 0.0
        ),

        "avg_precision": (
            round(precision_sum / total, 2)
            if total
            else 0.0
        ),
        "avg_recall": (
            round(recall_sum / total, 2)
            if total
            else 0.0
        ),

        "execution_error_tasks": execution_errors,
    }


def summarize_by_language(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Produce metrics grouped by language.

    Useful when running:

        predictions_en.json
        predictions_msa.json
        predictions_gulf.json
        predictions_mixed.json
    """

    summary = {}

    for language in LANGUAGES:

        language_results = [
            result
            for result in results
            if result.get(
                "language"
            ) == language
        ]

        if not language_results:
            continue

        summary[language] = summarize(
            language_results
        )

    return summary


DIFFICULTIES = ("easy", "medium", "hard")


def summarize_by_difficulty(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Produce metrics grouped by task difficulty (easy / medium / hard).
    """

    summary = {}

    for difficulty in DIFFICULTIES:

        difficulty_results = [
            result
            for result in results
            if result.get("difficulty") == difficulty
        ]

        if not difficulty_results:
            continue

        summary[difficulty] = summarize(
            difficulty_results
        )

    return summary


def cross_language_gap(
    by_language_summary: dict[str, dict[str, Any]],
    metric: str = "task_success_rate",
) -> dict[str, Any]:
    """
    Cross-Language Performance Gap (Delta).

    Reports, for the given metric (default: Task Success Rate), the
    max-vs-min gap across languages plus the per-language values used
    to compute it.
    """

    values = {
        language: lang_summary.get(metric, 0.0)
        for language, lang_summary in by_language_summary.items()
    }

    if not values:
        return {
            "metric": metric,
            "values": {},
            "max_language": None,
            "min_language": None,
            "gap": 0.0,
        }

    max_language = max(values, key=values.get)
    min_language = min(values, key=values.get)

    return {
        "metric": metric,
        "values": values,
        "max_language": max_language,
        "min_language": min_language,
        "gap": round(values[max_language] - values[min_language], 2),
    }


# ============================================================
# Save results
# ============================================================

def save_results(
    results: list[dict[str, Any]],
    output_dir: str | Path = RESULTS_DIR,
) -> tuple[Path, Path]:
    """
    Save detailed results and aggregate summary.
    """

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    detailed_path = (
        output_dir
        / "evaluation_results.json"
    )

    summary_path = (
        output_dir
        / "evaluation_summary.json"
    )

    sanitized_results = [
        _sanitize(result)
        for result in results
    ]

    summary = summarize(
        results
    )

    summary["by_language"] = (
        summarize_by_language(
            results
        )
    )

    summary["by_difficulty"] = (
        summarize_by_difficulty(
            results
        )
    )

    summary["cross_language_gap"] = cross_language_gap(
        summary["by_language"]
    )

    with detailed_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            sanitized_results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return (
        detailed_path,
        summary_path,
    )


# ============================================================
# Console summary
# ============================================================

def print_summary(
    results: list[dict[str, Any]],
) -> None:

    summary = summarize(
        results
    )

    print()
    print("=" * 70)
    print("OpsMix-Ar Evaluation")
    print("=" * 70)

    print(
        f"Total tasks:              "
        f"{summary['total_tasks']}"
    )

    print(
        f"Passed:                   "
        f"{summary['passed_tasks']}"
    )

    print(
        f"Failed:                   "
        f"{summary['failed_tasks']}"
    )

    print(
        f"Task Success Rate:        "
        f"{summary['task_success_rate']:.2f}%"
    )

    print(
        f"State Match Rate:         "
        f"{summary['state_match_rate']:.2f}%"
    )

    print(
        f"Tool Selection Accuracy:  "
        f"{summary['tool_selection_accuracy']:.2f}%"
    )

    print(
        f"Argument Accuracy:        "
        f"{summary['argument_accuracy']:.2f}%"
    )

    print(
        f"Order Exact-Match Rate:   "
        f"{summary['order_exact_match_rate']:.2f}%"
    )

    print(
        f"Avg Order Score:          "
        f"{summary['avg_order_score']:.2f}%"
    )

    print(
        f"Avg Precision:            "
        f"{summary['avg_precision']:.2f}%"
    )

    print(
        f"Avg Recall:               "
        f"{summary['avg_recall']:.2f}%"
    )

    print(
        f"Safety Violation Rate:    "
        f"{summary['safety_violation_rate']:.2f}%"
    )

    print(
        f"Forbidden tasks:          "
        f"{summary['forbidden_tasks']}"
    )

    print(
        f"Risky tasks:              "
        f"{summary['risky_tasks']}"
    )

    print(
        f"Unexpected-action tasks:  "
        f"{summary['unexpected_tasks']}"
    )

    print(
        f"Execution-error tasks:    "
        f"{summary['execution_error_tasks']}"
    )

    print("=" * 70)

    by_difficulty = summarize_by_difficulty(results)

    if by_difficulty:
        print()
        print(f"{'Difficulty':<10} {'SR%':>7} {'SMR%':>7} {'TSA%':>7} "
              f"{'ArgA%':>7} {'Order%':>7} {'SVR%':>7} {'N':>5}")
        print("-" * 70)

        for difficulty in DIFFICULTIES:
            s = by_difficulty.get(difficulty)
            if not s:
                continue
            print(
                f"{difficulty:<10} "
                f"{s['task_success_rate']:>6.2f}% "
                f"{s['state_match_rate']:>6.2f}% "
                f"{s['tool_selection_accuracy']:>6.2f}% "
                f"{s['argument_accuracy']:>6.2f}% "
                f"{s['order_exact_match_rate']:>6.2f}% "
                f"{s['safety_violation_rate']:>6.2f}% "
                f"{s['total_tasks']:>5}"
            )

    by_language = summarize_by_language(results)

    if by_language:
        gap = cross_language_gap(by_language)
        print()
        print(f"Cross-Language Gap (SR%): {gap['gap']:.2f} pts "
              f"(max={gap['max_language']}, min={gap['min_language']})")
        print(f"  Values: {gap['values']}")


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Qwen/LLM tool-call predictions "
            "against the OpsMix-Ar Tiny Infra Service."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "JSON file containing model predictions."
        ),
    )

    parser.add_argument(
        "--language",
        required=True,
        choices=LANGUAGES,
        help=(
            "Language represented by the prediction file."
        ),
    )

    parser.add_argument(
        "--task",
        help=(
            "Evaluate one task."
        ),
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        help=(
            "Evaluate selected task IDs."
        ),
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Evaluate every task in the input JSON."
        ),
    )

    parser.add_argument(
        "--base-url",
        default=BASE_URL,
        help=(
            "Tiny Infra Service URL."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=(
            "HTTP timeout in seconds."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            RESULTS_DIR
        ),
        help=(
            "Directory where evaluation results "
            "will be saved."
        ),
    )

    return parser


# ============================================================
# Main
# ============================================================

def main(
    argv: list[str] | None = None,
) -> int:

    parser = build_parser()

    args = parser.parse_args(
        argv
    )

    # --------------------------------------------------------
    # Validate task selector
    # --------------------------------------------------------

    selectors = sum(
        [
            args.task is not None,
            args.tasks is not None,
            args.all,
        ]
    )

    if selectors > 1:
        parser.error(
            "Use only one of "
            "--task, --tasks, or --all."
        )

    # --------------------------------------------------------
    # Load predictions
    # --------------------------------------------------------

    try:

        predictions = load_predictions(
            args.input
        )

    except Exception as exc:

        print(
            f"ERROR loading predictions: {exc}",
            file=sys.stderr,
        )

        return 1

    # --------------------------------------------------------
    # Select tasks
    # --------------------------------------------------------

    if args.task:

        if args.task not in predictions:
            print(
                f"ERROR: Task '{args.task}' "
                f"is not present in predictions.",
                file=sys.stderr,
            )

            return 1

        task_ids = [
            args.task
        ]

    elif args.tasks:

        missing = [
            task_id
            for task_id in args.tasks
            if task_id not in predictions
        ]

        if missing:

            print(
                "ERROR: These tasks are missing "
                f"from predictions: {missing}",
                file=sys.stderr,
            )

            return 1

        task_ids = args.tasks

    else:

        # Default behaviour = all tasks in prediction file.
        task_ids = list(
            predictions.keys()
        )

    if not task_ids:

        print(
            "ERROR: No tasks found.",
            file=sys.stderr,
        )

        return 1

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    results = evaluate_tasks(
        predictions=predictions,
        language=args.language,
        base_url=args.base_url,
        timeout=args.timeout,
        selected_task_ids=task_ids,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    detailed_path, summary_path = save_results(
        results=results,
        output_dir=args.output_dir,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_summary(
        results
    )

    print()
    print(
        f"Detailed results: {detailed_path}"
    )

    print(
        f"Summary:          {summary_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )