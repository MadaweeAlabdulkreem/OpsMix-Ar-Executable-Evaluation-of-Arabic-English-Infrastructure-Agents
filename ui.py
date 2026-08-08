import copy
import importlib
import json
import os
import time
from pathlib import Path
from typing import Optional

import streamlit as st

from app.agent import decide
from app.tasks import TASKS_BY_ID
from app.reset import reset
from app.state import state
from app.checker import check


# ---------------------------------------------------------------------------
# Results logging
# ---------------------------------------------------------------------------

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "results.jsonl"


def _current_model_name() -> str:
    """
    Best-effort model label for the results dataset.

    Priority:
    1) MODEL_NAME environment variable
    2) OPENAI_MODEL environment variable
    3) OPENAI_MODEL exported by app.agent
    4) simulated-agent
    """
    if os.environ.get("MODEL_NAME"):
        return os.environ["MODEL_NAME"]

    if os.environ.get("OPENAI_MODEL"):
        return os.environ["OPENAI_MODEL"]

    try:
        agent_module = importlib.import_module("app.agent")
        model_name = getattr(agent_module, "OPENAI_MODEL", None)
        if model_name:
            return str(model_name)
    except Exception:
        pass

    return "simulated-agent"


def _trajectory_from_trace(trace: list) -> list:
    """Convert the UI trace into the compact benchmark trajectory schema."""
    trajectory = []

    for step in trace:
        tool_name = step.get("executed_tool") or step.get("requested_tool")
        trajectory.append(
            {
                "tool": tool_name,
                "args": step.get("args", {}),
                "obs": step.get("result"),
            }
        )

    return trajectory


def _safety_violation(check_result: Optional[dict]):
    """Return a compact safety label such as risky:restart_service."""
    if not check_result:
        return None

    safety = check_result.get("safety", {})

    forbidden = safety.get("forbidden_calls", [])
    if forbidden:
        return f"forbidden:{forbidden[0]}"

    unexpected = safety.get("unexpected_calls", [])
    if unexpected:
        return f"unexpected:{unexpected[0]}"

    risky = safety.get("risky_calls", [])
    if risky:
        return f"risky:{risky[0]}"

    return None


def _first_wrong_step(gold_actions: list, actual_calls: list):
    """
    Return the zero-based index of the first mismatch.

    A mismatch can be:
    - wrong tool
    - wrong args
    - missing expected call
    - extra unexpected call
    """
    common = min(len(gold_actions), len(actual_calls))

    for index in range(common):
        gold = gold_actions[index]
        actual = actual_calls[index]

        if gold.get("tool") != actual.get("tool"):
            return index

        if gold.get("args", {}) != actual.get("args", {}):
            return index

    if len(gold_actions) != len(actual_calls):
        return common

    return None


def _failure_mode(
    check_result: Optional[dict],
    gold_actions: list,
    actual_calls: list,
    trace: list,
):
    """
    Classify the model failure using the benchmark categories:
    wrong_tool | wrong_arg | unsafe | gave_up | answered_in_text

    `answered_in_text` is reserved for a future agent adapter that exposes
    free-text answers. The current tool-call contract does not return them.
    """
    if check_result and check_result.get("passed"):
        return None

    safety = (check_result or {}).get("safety", {})

    if safety.get("forbidden_action"):
        return "unsafe"

    if not actual_calls:
        return "gave_up"

    common = min(len(gold_actions), len(actual_calls))

    for index in range(common):
        gold = gold_actions[index]
        actual = actual_calls[index]

        if gold.get("tool") != actual.get("tool"):
            return "wrong_tool"

        if gold.get("args", {}) != actual.get("args", {}):
            return "wrong_arg"

    if len(actual_calls) > len(gold_actions):
        return "wrong_tool"

    if len(actual_calls) < len(gold_actions):
        return "gave_up"

    # If tool path is exact but the final state still does not match,
    # this is likely an execution/state issue rather than a clear model
    # selection error, so do not mislabel it.
    return None


def _append_result(record: dict) -> None:
    """Append one model attempt as one JSON line."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="OpsMix-Ar Agent Evaluation",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------

LANGUAGES = {
    "English": "en",
    "Modern Standard Arabic (MSA)": "msa",
    "Gulf Arabic": "gulf",
    "Mixed Arabic-English": "mixed",
}


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

TOOL_ALIASES = {
    "check_disk_usage": "check_disk",
}


# ---------------------------------------------------------------------------
# Argument normalization
# ---------------------------------------------------------------------------

def _normalise_args(tool_name: str, args: dict) -> dict:
    args = dict(args or {})

    if tool_name in {
        "restart_service",
        "get_metrics",
        "get_logs",
    }:
        if "service_name" in args and "service" not in args:
            args["service"] = args.pop("service_name")

    if tool_name == "clear_cache":
        args.pop("service_name", None)
        args.pop("service", None)

    if tool_name == "rotate_api_key":
        args = {}

    return args


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def execute_tool(tool_name: str, args: dict):
    """
    Execute one real sandbox tool from app.main.
    """

    tools = importlib.import_module("app.main")

    actual_tool_name = TOOL_ALIASES.get(
        tool_name,
        tool_name,
    )

    tool_registry = {
        "check_disk": tools.check_disk,
        "clear_cache": tools.clear_cache,
        "restart_service": tools.restart_service,
        "rotate_api_key": tools.rotate_api_key,
        "scale_replicas": tools.scale_replicas,
        "get_metrics": tools.get_metrics,
        "rollback_deploy": tools.rollback_deploy,
        "get_logs": tools.get_logs,
        "kill_process": tools.kill_process,
        "set_config": tools.set_config,
    }

    if actual_tool_name not in tool_registry:
        raise ValueError(
            f"Agent requested unsupported tool '{tool_name}'. "
            f"Available real tools: {', '.join(tool_registry)}"
        )

    normalised_args = _normalise_args(
        actual_tool_name,
        args,
    )

    result = tool_registry[actual_tool_name](
        **normalised_args
    )

    return {
        "requested_tool": tool_name,
        "executed_tool": actual_tool_name,
        "executed_args": normalised_args,
        "result": result,
    }


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "active_task_id" not in st.session_state:
    st.session_state.active_task_id = None

if "run_output" not in st.session_state:
    st.session_state.run_output = None


# ---------------------------------------------------------------------------
# Sidebar — Experiment Settings
# ---------------------------------------------------------------------------

with st.sidebar:

    st.header("Experiment Settings")

    language_label = st.selectbox(
        "Language",
        list(LANGUAGES.keys()),
    )

    language_code = LANGUAGES[language_label]

    task_id = st.selectbox(
        "Task",
        list(TASKS_BY_ID.keys()),
    )


selected_task = TASKS_BY_ID[task_id]


# ---------------------------------------------------------------------------
# Reset automatically when task changes
# ---------------------------------------------------------------------------

if st.session_state.active_task_id != task_id:

    reset(task_id)

    st.session_state.active_task_id = task_id
    st.session_state.run_output = None


# ---------------------------------------------------------------------------
# Sidebar — Task metadata
# ---------------------------------------------------------------------------

with st.sidebar:

    st.markdown("---")

    st.caption(
        f"Domain: {selected_task.get('domain', '-')}"
    )

    st.caption(
        f"Incident: {selected_task.get('incident', '-')}"
    )

    st.caption(
        f"Difficulty: {selected_task.get('difficulty', '-')}"
    )

    st.caption(
        f"Model: {_current_model_name()}"
    )

    if st.button(
        "Reset current task",
        use_container_width=True,
    ):
        reset(task_id)
        st.session_state.run_output = None
        st.rerun()


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

st.title("🧪 OpsMix-Ar Tool-Calling Evaluation")

st.caption(
    "Task → Initial State → User Request → Agent → "
    "Tool Calls → State → Checker → PASS / FAIL"
)


# ---------------------------------------------------------------------------
# Selected task
# ---------------------------------------------------------------------------

st.subheader("Selected Task")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Task ID",
        selected_task["task_id"],
    )

with c2:
    st.metric(
        "Domain",
        selected_task.get("domain", "-"),
    )

with c3:
    st.metric(
        "Difficulty",
        selected_task.get("difficulty", "-"),
    )


st.markdown(
    f"**Incident:** `{selected_task.get('incident', '-')}`"
)


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

with st.expander("🔍 View Ground Truth"):

    st.markdown("### Initial State")
    st.json(
        selected_task.get(
            "initial_state",
            {},
        )
    )

    st.markdown("### Gold Actions")
    st.json(
        selected_task.get(
            "gold_actions",
            [],
        )
    )

    if selected_task.get("conditional_actions"):

        st.markdown("### Conditional Actions")

        st.json(
            selected_task.get(
                "conditional_actions",
                [],
            )
        )

    st.markdown("### Gold Final State")

    st.json(
        selected_task.get(
            "gold_final_state",
            {},
        )
    )

    st.markdown("### Safety Rules")

    st.json(
        selected_task.get(
            "safety",
            {},
        )
    )


# ---------------------------------------------------------------------------
# State before execution
# ---------------------------------------------------------------------------

st.subheader("State — Before Execution")

st.json(state)


# ---------------------------------------------------------------------------
# User request
# ---------------------------------------------------------------------------

st.subheader("User Request")


task_requests = selected_task.get(
    "request",
    {},
)

default_request = task_requests.get(
    language_code,
    "",
)


user_request = st.text_area(
    "Request",
    value=default_request,
    height=110,
    key=f"request_{task_id}_{language_code}",
)


st.caption(
    f"Language variant: `{language_code}`"
)


# ---------------------------------------------------------------------------
# Run agent
# ---------------------------------------------------------------------------

run_clicked = st.button(
    "▶ Run Agent",
    type="primary",
)


if run_clicked:

    if not user_request.strip():

        st.warning("اكتبي طلب أول.")

    else:

        # Always start this experiment from the canonical task state.
        reset(task_id)

        initial_state = copy.deepcopy(state)
        attempt_started = time.perf_counter()

        # ---------------------------------------------------------------
        # Agent decision
        # ---------------------------------------------------------------

        trace = []
        execution_error = None
        agent_error = None

        try:
            steps, intent, service = decide(
                user_request,
                language_code,
                copy.deepcopy(state),
            )
        except Exception as exc:
            steps = []
            intent = None
            service = None
            agent_error = str(exc)
            execution_error = f"Agent error: {exc}"

        # ---------------------------------------------------------------
        # Execute tool calls
        # ---------------------------------------------------------------

        for step in steps:

            try:

                call = execute_tool(
                    step["tool"],
                    step.get("args", {}),
                )

                trace.append(
                    {
                        "requested_tool": call[
                            "requested_tool"
                        ],
                        "executed_tool": call[
                            "executed_tool"
                        ],
                        "args": call[
                            "executed_args"
                        ],
                        "reasoning": step.get(
                            "reasoning",
                            "",
                        ),
                        "result": call[
                            "result"
                        ],
                    }
                )

            except Exception as exc:

                execution_error = str(exc)

                trace.append(
                    {
                        "requested_tool": step.get(
                            "tool"
                        ),
                        "executed_tool": None,
                        "args": step.get(
                            "args",
                            {},
                        ),
                        "reasoning": step.get(
                            "reasoning",
                            "",
                        ),
                        "result": (
                            f"ERROR: {exc}"
                        ),
                    }
                )

                break


        # ---------------------------------------------------------------
        # Checker
        # ---------------------------------------------------------------

        checker_error = None
        check_result = None

        try:

            check_result = check(task_id)

        except Exception as exc:

            checker_error = str(exc)

        latency_s = round(time.perf_counter() - attempt_started, 3)

        actual_calls = [
            {
                "tool": entry.get("tool"),
                "args": entry.get("args", {}),
            }
            for entry in state.get("history", [])
        ]

        gold_actions = selected_task.get("gold_actions", [])

        result_record = {
            "task_id": task_id,
            "model": _current_model_name(),
            "lang": language_code,
            "trajectory": _trajectory_from_trace(trace),
            "passed": bool(check_result and check_result.get("passed")),
            "final_state_match": (
                check_result.get("state_match")
                if check_result
                else False
            ),
            "safety_violation": _safety_violation(check_result),
            "first_wrong_step": _first_wrong_step(
                gold_actions,
                actual_calls,
            ),
            "failure_mode": _failure_mode(
                check_result,
                gold_actions,
                actual_calls,
                trace,
            ),
            "latency_s": latency_s,
        }

        # Keep API / execution errors visible in the attempt record without
        # changing the canonical fields above.
        if agent_error or execution_error or checker_error:
            result_record["error"] = (
                agent_error
                or execution_error
                or checker_error
            )

        try:
            _append_result(result_record)
            result_log_error = None
        except Exception as exc:
            result_log_error = str(exc)


        # ---------------------------------------------------------------
        # Store output
        # ---------------------------------------------------------------

        st.session_state.run_output = {
            "task_id": task_id,
            "language": language_label,
            "language_code": language_code,
            "user_request": user_request,
            "intent": intent,
            "service": service,
            "initial_state": initial_state,
            "trace": trace,
            "final_state": copy.deepcopy(state),
            "check_result": check_result,
            "execution_error": execution_error,
            "checker_error": checker_error,
            "result_record": result_record,
            "result_log_error": result_log_error,
            "latency_s": latency_s,
            "model": _current_model_name(),
        }


# ---------------------------------------------------------------------------
# Experiment output
# ---------------------------------------------------------------------------

output = st.session_state.run_output


if output:

    st.markdown("---")

    st.header("Experiment Result")


    # -------------------------------------------------------------------
    # Agent decision
    # -------------------------------------------------------------------

    st.subheader("1. Agent Decision")

    st.write(
        "Detected intent:",
        output["intent"],
    )

    st.write(
        "Detected service:",
        output["service"],
    )


    # -------------------------------------------------------------------
    # Tool calls
    # -------------------------------------------------------------------

    st.subheader("2. Tool Call Trace")

    if not output["trace"]:

        st.warning(
            "Agent made no tool calls."
        )

    else:

        for i, step in enumerate(
            output["trace"],
            start=1,
        ):

            label = (
                step["executed_tool"]
                or step["requested_tool"]
            )

            with st.expander(
                f"Step {i}: {label}",
                expanded=True,
            ):

                if (
                    step["requested_tool"]
                    != step["executed_tool"]
                ):

                    st.write(
                        "Agent requested:",
                        step["requested_tool"],
                    )

                    st.write(
                        "Executed as:",
                        step["executed_tool"],
                    )

                st.write("Arguments:")

                st.json(
                    step["args"]
                )

                st.write("Reasoning:")

                st.write(
                    step["reasoning"]
                )

                st.write("Result:")

                st.write(
                    step["result"]
                )


    # -------------------------------------------------------------------
    # Final state
    # -------------------------------------------------------------------

    st.subheader("3. Final State")

    st.json(
        output["final_state"]
    )


    # -------------------------------------------------------------------
    # Expected vs actual
    # -------------------------------------------------------------------

    st.subheader(
        "4. Expected vs Actual Tool Path"
    )

    col_a, col_b = st.columns(2)


    with col_a:

        st.markdown(
            "**Expected / Gold**"
        )

        st.json(
            selected_task.get(
                "gold_actions",
                [],
            )
        )


    with col_b:

        st.markdown(
            "**Actual Recorded Calls**"
        )

        st.json(
            [
                {
                    "tool": entry.get(
                        "tool"
                    ),
                    "args": entry.get(
                        "args",
                        {},
                    ),
                }
                for entry in state.get(
                    "history",
                    [],
                )
            ]
        )


    # -------------------------------------------------------------------
    # Checker
    # -------------------------------------------------------------------

    st.subheader("5. Checker")


    if output["checker_error"]:

        st.error(
            f"Checker error: "
            f"{output['checker_error']}"
        )

    elif output["check_result"]:

        if output[
            "check_result"
        ]["passed"]:

            st.success(
                "✅ PASS"
            )

        else:

            st.error(
                "❌ FAIL"
            )

        st.json(
            output[
                "check_result"
            ]
        )

    st.subheader("6. Results Dataset Record")

    st.json(
        output.get(
            "result_record",
            {},
        )
    )

    if output.get("result_log_error"):
        st.error(
            "Could not append to results/results.jsonl: "
            f"{output['result_log_error']}"
        )
    else:
        st.success(
            "Result saved to results/results.jsonl"
        )


    # -------------------------------------------------------------------
    # Execution error
    # -------------------------------------------------------------------

    if output["execution_error"]:

        st.error(
            f"Execution Error: "
            f"{output['execution_error']}"
        )