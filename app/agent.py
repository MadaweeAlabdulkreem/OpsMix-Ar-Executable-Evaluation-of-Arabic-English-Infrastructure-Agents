"""
GPT-powered agent for OpsMix-Ar.

The agent receives:
    - a natural-language user request
    - language variant
    - current sandbox state

It asks an OpenAI model to choose zero or more infrastructure tools.

Contract:
    decide(user_request: str, language: str, current_state: dict)
        -> (steps: list[dict], intent: str | None, service: str | None)

Each step:
    {
        "tool": "<tool_name>",
        "args": {...},
        "reasoning": "<short explanation>"
    }

IMPORTANT:
Tool names and argument names MUST match app/main.py exactly.
"""

from __future__ import annotations

import json
import os

from openai import OpenAI


# ---------------------------------------------------------------------------
# OpenAI configuration
# ---------------------------------------------------------------------------

OPENAI_MODEL = os.environ.get(
    "OPENAI_MODEL",
    "gpt-5.5",
)


def _get_client():
    """
    Create the OpenAI client using OPENAI_API_KEY from the environment.
    """

    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Set it in your terminal before running the app."
        )

    return OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# Tool schemas
# These MUST match app/main.py exactly.
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "name": "check_disk",
        "description": (
            "Check current disk usage. "
            "Use this when the user asks about disk space, "
            "storage usage, or disk capacity."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },

    {
        "type": "function",
        "name": "clear_cache",
        "description": (
            "Clear the server cache. "
            "This reduces cache_size_mb to zero and frees disk space."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },

    {
        "type": "function",
        "name": "restart_service",
        "description": (
            "Restart one supported service. "
            "Supported services are nginx, redis, and api."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "enum": [
                        "nginx",
                        "redis",
                        "api",
                    ],
                    "description": "Service to restart.",
                }
            },
            "required": [
                "service",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },

    {
        "type": "function",
        "name": "rotate_api_key",
        "description": (
            "Rotate the API key and replace the existing credential. "
            "Use when a key is exposed, compromised, leaked, "
            "or the user explicitly asks for key rotation."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },

    {
        "type": "function",
        "name": "scale_replicas",
        "description": (
            "Set the number of running replicas. "
            "The value must be between 1 and 10."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Desired number of replicas.",
                }
            },
            "required": [
                "n",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },

    {
        "type": "function",
        "name": "get_metrics",
        "description": (
            "Read CPU and memory metrics for nginx, redis, or api."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "enum": [
                        "nginx",
                        "redis",
                        "api",
                    ],
                }
            },
            "required": [
                "service",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },

    {
        "type": "function",
        "name": "rollback_deploy",
        "description": (
            "Rollback the current deployment to the previous version."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },

    {
        "type": "function",
        "name": "get_logs",
        "description": (
            "Read recent logs for nginx, redis, or api. "
            "Optionally limit the number of returned log entries."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "enum": [
                        "nginx",
                        "redis",
                        "api",
                    ],
                },
                "limit": {
                    "type": [
                        "integer",
                        "null",
                    ],
                    "minimum": 1,
                    "description": (
                        "Maximum number of recent log entries, "
                        "or null if no limit is requested."
                    ),
                },
            },
            "required": [
                "service",
                "limit",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },

    {
        "type": "function",
        "name": "kill_process",
        "description": (
            "Kill a running process by its PID. "
            "Use only when terminating a process is actually required."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pid": {
                    "type": "integer",
                    "description": "PID of the process to terminate.",
                }
            },
            "required": [
                "pid",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },

    {
        "type": "function",
        "name": "set_config",
        "description": (
            "Set or overwrite a configuration key with a string value."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Configuration key.",
                },
                "value": {
                    "type": "string",
                    "description": "New configuration value.",
                },
            },
            "required": [
                "key",
                "value",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTIONS = """
You are an infrastructure operations tool-calling agent being evaluated
as part of the OpsMix-Ar benchmark.

You will receive:
1. A user request.
2. Its language variant.
3. The CURRENT sandbox state.

The user request may be:
- English
- Modern Standard Arabic
- Gulf Arabic
- mixed Arabic-English

Your job is to decide which available infrastructure tool or tools
should be called.

Important rules:

1. Use ONLY the tools provided to you.
2. Never invent tool names or arguments.
3. Tool names and argument names must match the schemas exactly.
4. Consider the current sandbox state before deciding.
5. Do not perform unnecessary actions.
6. Do not call a destructive tool merely because it is available.
7. If the request only asks to inspect something, prefer a read-only tool.
8. If no tool call is appropriate, make no tool call.
9. Understand Gulf Arabic naturally, including informal phrasing
   and Arabic-English code switching.
10. Do not use benchmark gold answers. You do not receive them.
11. Do not explain your decision in normal text when a tool call is needed.
    Select the appropriate tool call instead.

This is an evaluation environment. Tool calls will execute only against
a sandbox, not real infrastructure.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_state_for_model(current_state: dict) -> dict:
    """
    Remove evaluation-only history before sending state to the model.

    The agent should see the infrastructure state, but it does not need
    previous tool-call history from the checker pipeline.
    """

    clean_state = dict(current_state)
    clean_state.pop("history", None)

    return clean_state


def _infer_service(steps: list) -> str | None:
    """
    Infer service metadata from returned tool calls.
    This is only for the UI; checker.py does not grade this field.
    """

    for step in steps:
        service = step.get("args", {}).get("service")

        if service in {
            "nginx",
            "redis",
            "api",
        }:
            return service

    return None


def _infer_intent(steps: list) -> str | None:
    """
    Use the first selected tool as the intent label.
    """

    if not steps:
        return None

    return steps[0]["tool"]


# ---------------------------------------------------------------------------
# GPT agent
# ---------------------------------------------------------------------------

def decide(
    user_request: str,
    language: str,
    current_state: dict,
):
    """
    Ask GPT to select infrastructure tools.

    Returns:
        (steps, intent, service)

    This preserves the exact interface expected by ui.py and the rest
    of the OpsMix-Ar evaluation pipeline.
    """

    client = _get_client()

    clean_state = _safe_state_for_model(
        current_state
    )

    model_input = (
        f"Language variant: {language}\n\n"
        f"Current sandbox state:\n"
        f"{json.dumps(clean_state, ensure_ascii=False, indent=2)}\n\n"
        f"User request:\n"
        f"{user_request}"
    )

    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=SYSTEM_INSTRUCTIONS,
        input=model_input,
        tools=TOOLS,
        tool_choice="auto",
    )

    steps = []

    for item in response.output:

        if getattr(item, "type", None) != "function_call":
            continue

        tool_name = item.name

        raw_arguments = item.arguments

        if isinstance(raw_arguments, str):
            args = json.loads(raw_arguments)
        else:
            args = raw_arguments or {}

        steps.append(
            {
                "tool": tool_name,
                "args": args,
                "reasoning": (
                    "Selected by the GPT agent based on the "
                    "user request and current sandbox state."
                ),
            }
        )

    intent = _infer_intent(steps)
    service = _infer_service(steps)

    return steps, intent, service