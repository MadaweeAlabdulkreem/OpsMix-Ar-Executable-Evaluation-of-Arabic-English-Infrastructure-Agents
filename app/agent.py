"""
agent.py
--------
The "agent" decides which tool(s) to call given a natural-language
request. This file currently contains a SIMULATED agent (deterministic
keyword matching across English / MSA / Gulf Arabic / Mixed).

WHY simulated first:
  - It is fast, free, and 100% reproducible for demoing the pipeline.
  - It defines the exact CONTRACT (input -> list of tool calls) that a
    real LLM must satisfy later. Swap `decide()` for `decide_with_llm()`
    below and nothing else in the project needs to change.

Contract:
    decide(user_request: str, language: str, current_state: dict)
        -> (steps: list[dict], intent: str|None, service: str|None)

    Each step in `steps` looks like:
        {"tool": "<tool_name>", "args": {...}, "reasoning": "<why>"}
"""

# ---------------------------------------------------------------------------
# Keyword dictionaries covering English, MSA, Gulf Arabic, and mixed usage.
# Matching is deliberately simple substring matching -- good enough for a
# deterministic demo agent, and language-agnostic on purpose: real mixed
# Arabic-English sentences often blend scripts mid-sentence.
# ---------------------------------------------------------------------------
SERVICE_KEYWORDS = {
    "payment": ["payment", "الدفع", "دفع", "الپايمنت", "payment service"],
    "database": ["database", "db ", " db", "قاعدة البيانات", "قاعدة بيانات", "الداتابيس", "داتا بيس"],
    "api": ["api", "الـ api", "خدمة api", "خدمة ال api"],
}

INTENT_KEYWORDS = {
    # "turn it on / restart / fix it / it's down, start it"
    "restart": [
        "restart", "turn it on", "turn on", "start it", "start the",
        "شغل", "شغلها", "شغّل", "شغّلها", "فعل", "فعّل", "شغلها لي", "قم بتشغيل",
    ],
    "stop": [
        "stop", "shut down", "shutdown", "turn it off", "turn off",
        "وقف", "أوقف", "اوقف", "وقفها", "أوقفها",
    ],
    "check_status": [
        "status", "check status", "what's the status", "is it running",
        "وش وضع", "ايش وضع", "تحقق من", "الحالة", "وضع الخدمة", "شو وضع",
    ],
    "clear_cache": [
        "cache", "clear cache", "clear the cache",
        "كاش", "امسح الكاش", "نظف الكاش", "امسح الكاش تبع",
    ],
    "rotate_key": [
        "rotate key", "rotate the api key", "api key", "new key",
        "مفتاح", "غير المفتاح", "غيّر المفتاح", "دور المفتاح", "غير مفتاح",
    ],
    "disk_usage": [
        "disk", "disk usage", "storage",
        "مساحة القرص", "مساحة التخزين", "القرص", "مساحة الهارد",
    ],
}

# Extra "problem" signal words (Gulf/MSA) that reinforce a restart intent,
# e.g. "خدمة الدفع طافية" = "the payment service is down/off".
DOWN_SIGNAL_WORDS = ["طافية", "طافي", "واقفة", "واقف", "down", "not working", "مو شغالة", "معطلة"]


def _normalize(text: str) -> str:
    return text.strip().lower()


def detect_service(text: str):
    for service, keywords in SERVICE_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return service
    return None


def detect_intent(text: str):
    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return intent
    # If nothing matched but a "down" signal word is present, assume restart
    for kw in DOWN_SIGNAL_WORDS:
        if kw in text:
            return "restart"
    return None


def decide(user_request: str, language: str, current_state: dict):
    """
    SIMULATED AGENT.
    `language` is accepted for interface parity with the real-LLM version
    (a real LLM adapter would use it to pick a system prompt / few-shot
    examples per language variant). The simulated agent itself matches
    keywords across all variants regardless of the selected language.
    """
    text = _normalize(user_request)
    service = detect_service(text)
    intent = detect_intent(text)

    steps = []

    # disk_usage is a server-level check, not tied to a specific service
    if intent == "disk_usage":
        steps = [
            {"tool": "check_disk_usage", "args": {"server_name": "main"},
             "reasoning": "User asked about disk usage."},
        ]
        return steps, intent, service

    if service is None or intent is None:
        return steps, intent, service

    if intent == "restart":
        steps = [
            {"tool": "check_service_status", "args": {"service_name": service},
             "reasoning": "Verify current status before restarting."},
            {"tool": "restart_service", "args": {"service_name": service},
             "reasoning": "User asked to turn the service back on."},
        ]
    elif intent == "stop":
        steps = [
            {"tool": "check_service_status", "args": {"service_name": service},
             "reasoning": "Verify current status before stopping."},
            {"tool": "stop_service", "args": {"service_name": service},
             "reasoning": "User asked to stop the service."},
        ]
    elif intent == "check_status":
        steps = [
            {"tool": "check_service_status", "args": {"service_name": service},
             "reasoning": "User asked for a status check only."},
        ]
    elif intent == "clear_cache":
        steps = [
            {"tool": "check_disk_usage", "args": {"server_name": "main"},
             "reasoning": "Check disk usage before clearing cache."},
            {"tool": "clear_cache", "args": {"service_name": service},
             "reasoning": "User asked to clear the cache."},
        ]
    elif intent == "rotate_key":
        steps = [
        {
            "tool": "rotate_api_key",
            "args": {},
            "reasoning": "User asked to rotate the API key."
        },
    ]
    return steps, intent, service


# ---------------------------------------------------------------------------
# =====================  LLM INTEGRATION POINT (RunPod)  =====================
# To replace the simulated agent with a real open-source LLM served on
# RunPod (e.g. an OpenAI-compatible vLLM / TGI endpoint), implement a
# function with the SAME return contract as `decide()` above, and swap the
# call site in app.py from `agent.decide(...)` to `agent.decide_with_llm(...)`.
#
# Sketch (left unimplemented on purpose -- no live network calls in this demo):
#
# import requests
#
# RUNPOD_ENDPOINT = "https://<your-runpod-id>.proxy.runpod.net/v1/chat/completions"
#
# TOOL_SCHEMA = [
#     {"name": "check_service_status", "parameters": {"service_name": "str"}},
#     {"name": "restart_service", "parameters": {"service_name": "str"}},
#     {"name": "stop_service", "parameters": {"service_name": "str"}},
#     {"name": "check_disk_usage", "parameters": {"server_name": "str"}},
#     {"name": "clear_cache", "parameters": {"service_name": "str"}},
#     {"name": "rotate_api_key", "parameters": {"service_name": "str"}},
# ]
#
# def decide_with_llm(user_request: str, language: str, current_state: dict):
#     system_prompt = build_system_prompt(language, TOOL_SCHEMA, current_state)
#     response = requests.post(
#         RUNPOD_ENDPOINT,
#         json={
#             "model": "your-open-source-model",
#             "messages": [
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_request},
#             ],
#             "tools": TOOL_SCHEMA,
#         },
#         timeout=60,
#     )
#     tool_calls = parse_tool_calls(response.json())  # -> list of {"tool", "args", "reasoning"}
#     intent, service = infer_intent_service_from_calls(tool_calls)  # for the validator
#     return tool_calls, intent, service
#
# Everything downstream (sandbox execution, trace UI, validator) is already
# written against the `(steps, intent, service)` contract, so no other file
# needs to change when you plug this in.
# ==============================================================================
