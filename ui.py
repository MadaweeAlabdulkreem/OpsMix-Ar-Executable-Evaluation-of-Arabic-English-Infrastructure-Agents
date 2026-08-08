import copy
import importlib
import sys

import streamlit as st

from app.agent import decide
from app.tasks import TASKS_BY_ID
from app.reset import reset
from app.state import state
from app.checker import check

st.set_page_config(page_title="OpsMix-Ar Agent Evaluation", layout="wide")

LANGUAGES = {
    "English": "en",
    "Modern Standard Arabic (MSA)": "msa",
    "Gulf Arabic": "gulf",
    "Mixed Arabic-English": "mixed",
}

TOOL_ALIASES = {
    "check_disk_usage": "check_disk",
}

def _normalise_args(tool_name: str, args: dict) -> dict:
    args = dict(args or {})

    if tool_name in {"restart_service", "get_metrics", "get_logs"}:
        if "service_name" in args and "service" not in args:
            args["service"] = args.pop("service_name")

    if tool_name == "clear_cache":
        args.pop("service_name", None)
        args.pop("service", None)

    if tool_name == "rotate_api_key":
        args = {}

    return args

def execute_tool(tool_name: str, args: dict):
    if sys.version_info < (3, 10):
        raise RuntimeError(
            "Tool execution requires Python 3.10+ for the current app/main.py. "
            f"Current interpreter: Python {sys.version_info.major}.{sys.version_info.minor}."
        )

    tools = importlib.import_module("app.main")
    actual_tool_name = TOOL_ALIASES.get(tool_name, tool_name)

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

    normalised_args = _normalise_args(actual_tool_name, args)
    result = tool_registry[actual_tool_name](**normalised_args)

    return {
        "requested_tool": tool_name,
        "executed_tool": actual_tool_name,
        "executed_args": normalised_args,
        "result": result,
    }

if "active_task_id" not in st.session_state:
    st.session_state.active_task_id = None

if "run_output" not in st.session_state:
    st.session_state.run_output = None

with st.sidebar:
    st.header("Experiment Settings")
    language_label = st.selectbox("Language", list(LANGUAGES.keys()))
    language_code = LANGUAGES[language_label]
    task_id = st.selectbox("Task", list(TASKS_BY_ID.keys()))

selected_task = TASKS_BY_ID[task_id]

if st.session_state.active_task_id != task_id:
    reset(task_id)
    st.session_state.active_task_id = task_id
    st.session_state.run_output = None

with st.sidebar:
    st.markdown("---")
    st.caption(f"Domain: {selected_task.get('domain', '-')}")
    st.caption(f"Incident: {selected_task.get('incident', '-')}")
    st.caption(f"Difficulty: {selected_task.get('difficulty', '-')}")
    if st.button("Reset current task", use_container_width=True):
        reset(task_id)
        st.session_state.run_output = None
        st.rerun()

st.title("🧪 OpsMix-Ar Tool-Calling Evaluation")
st.caption(
    "Task → Initial State → User Request → Agent → "
    "Tool Calls → State → Checker → PASS / FAIL"
)

if sys.version_info < (3, 10):
    st.warning(
        "The UI can open on this interpreter, but real tool execution needs "
        f"Python 3.10+. Current interpreter: "
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

st.subheader("Selected Task")
c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Task ID", selected_task["task_id"])
with c2:
    st.metric("Domain", selected_task.get("domain", "-"))
with c3:
    st.metric("Difficulty", selected_task.get("difficulty", "-"))

st.markdown(f"**Incident:** `{selected_task.get('incident', '-')}`")

with st.expander("🔍 View Ground Truth"):
    st.markdown("### Initial State")
    st.json(selected_task.get("initial_state", {}))

    st.markdown("### Gold Actions")
    st.json(selected_task.get("gold_actions", []))

    if selected_task.get("conditional_actions"):
        st.markdown("### Conditional Actions")
        st.json(selected_task.get("conditional_actions", []))

    st.markdown("### Gold Final State")
    st.json(selected_task.get("gold_final_state", {}))

    st.markdown("### Safety Rules")
    st.json(selected_task.get("safety", {}))

st.subheader("State — Before Execution")
st.json(state)

st.subheader("User Request")
user_request = st.text_input(
    "Write the request manually",
    placeholder="مثال: مفتاح الـAPI انكشف، غيّره فوراً",
)

run_clicked = st.button("▶ Run Agent", type="primary")

if run_clicked:
    if not user_request.strip():
        st.warning("اكتبي طلب أول.")
    else:
        initial_state = copy.deepcopy(state)

        steps, intent, service = decide(
            user_request,
            language_code,
            copy.deepcopy(state),
        )

        trace = []
        execution_error = None

        for step in steps:
            try:
                call = execute_tool(step["tool"], step.get("args", {}))
                trace.append(
                    {
                        "requested_tool": call["requested_tool"],
                        "executed_tool": call["executed_tool"],
                        "args": call["executed_args"],
                        "reasoning": step.get("reasoning", ""),
                        "result": call["result"],
                    }
                )
            except Exception as exc:
                execution_error = str(exc)
                trace.append(
                    {
                        "requested_tool": step.get("tool"),
                        "executed_tool": None,
                        "args": step.get("args", {}),
                        "reasoning": step.get("reasoning", ""),
                        "result": f"ERROR: {exc}",
                    }
                )
                break

        checker_error = None
        check_result = None
        try:
            check_result = check(task_id)
        except Exception as exc:
            checker_error = str(exc)

        st.session_state.run_output = {
            "task_id": task_id,
            "language": language_label,
            "user_request": user_request,
            "intent": intent,
            "service": service,
            "initial_state": initial_state,
            "trace": trace,
            "final_state": copy.deepcopy(state),
            "check_result": check_result,
            "execution_error": execution_error,
            "checker_error": checker_error,
        }

output = st.session_state.run_output

if output:
    st.markdown("---")
    st.header("Experiment Result")

    st.subheader("1. Agent Decision")
    st.write("Detected intent:", output["intent"])
    st.write("Detected service:", output["service"])

    st.subheader("2. Tool Call Trace")
    if not output["trace"]:
        st.warning("Agent made no tool calls.")
    else:
        for i, step in enumerate(output["trace"], start=1):
            label = step["executed_tool"] or step["requested_tool"]
            with st.expander(f"Step {i}: {label}", expanded=True):
                if step["requested_tool"] != step["executed_tool"]:
                    st.write("Agent requested:", step["requested_tool"])
                    st.write("Executed as:", step["executed_tool"])
                st.write("Arguments:")
                st.json(step["args"])
                st.write("Reasoning:")
                st.write(step["reasoning"])
                st.write("Result:")
                st.write(step["result"])

    st.subheader("3. Final State")
    st.json(output["final_state"])

    st.subheader("4. Expected vs Actual Tool Path")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Expected / Gold**")
        st.json(selected_task.get("gold_actions", []))

    with col_b:
        st.markdown("**Actual Recorded Calls**")
        st.json(
            [
                {"tool": entry.get("tool"), "args": entry.get("args", {})}
                for entry in state.get("history", [])
            ]
        )

    st.subheader("5. Checker")

    if output["checker_error"]:
        st.error(f"Checker error: {output['checker_error']}")
    elif output["check_result"]:
        if output["check_result"]["passed"]:
            st.success("✅ PASS")
        else:
            st.error("❌ FAIL")
        st.json(output["check_result"])

    if output["execution_error"]:
        st.error(f"Execution Error: {output['execution_error']}")