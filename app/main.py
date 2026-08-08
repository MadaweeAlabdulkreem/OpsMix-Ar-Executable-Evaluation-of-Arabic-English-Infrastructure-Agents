"""FastAPI app exposing the tiny infra service's tools as HTTP endpoints."""

from __future__ import annotations

import copy
import secrets

from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException

from app.state import state, SERVICE_NAMES


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Tiny Infra Service")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_REPLICAS = 1
MAX_REPLICAS = 10


# ---------------------------------------------------------------------------
# History helper
# ---------------------------------------------------------------------------

def _record(
    tool: str,
    args: dict,
    timestamp: Optional[str] = None,
    state_before: Optional[dict] = None,
) -> None:
    """
    Record a tool call in the shared sandbox history.

    state_before stores a snapshot of the sandbox immediately before
    the tool call. checker.py uses this for conditional actions.
    """

    state["history"].append(
        {
            "tool": tool,
            "args": args,
            "timestamp": timestamp
            or datetime.now(timezone.utc).isoformat(),
            "state_before": state_before,
        }
    )


# ---------------------------------------------------------------------------
# 1. check_disk()
# ---------------------------------------------------------------------------

@app.get("/check_disk")
def check_disk():
    """Return current disk usage. Read-only."""

    state_before = copy.deepcopy(state)

    disk_usage_percent = (
        state["disk_used_gb"] / state["disk_total_gb"]
    ) * 100

    _record(
        "check_disk",
        {},
        state_before=state_before,
    )

    return {
        "disk_total_gb": state["disk_total_gb"],
        "disk_used_gb": state["disk_used_gb"],
        "disk_usage_percent": round(disk_usage_percent, 2),
    }


# ---------------------------------------------------------------------------
# 2. clear_cache()
# ---------------------------------------------------------------------------

@app.post("/clear_cache")
def clear_cache():
    """
    Clear the cache and reduce disk usage accordingly.
    """

    state_before = copy.deepcopy(state)

    # Convert MB -> GB
    freed_gb = state["cache_size_mb"] / 1024

    state["disk_used_gb"] = max(
        0,
        state["disk_used_gb"] - freed_gb,
    )

    state["cache_size_mb"] = 0

    _record(
        "clear_cache",
        {},
        state_before=state_before,
    )

    disk_usage_percent = (
        state["disk_used_gb"] / state["disk_total_gb"]
    ) * 100

    return {
        "status": "success",
        "cache_size_mb": state["cache_size_mb"],
        "disk_used_gb": round(state["disk_used_gb"], 2),
        "disk_usage_percent": round(disk_usage_percent, 2),
    }


# ---------------------------------------------------------------------------
# 3. restart_service(service)
# ---------------------------------------------------------------------------

@app.post("/restart_service")
def restart_service(service: str):
    """
    Restart one of:
        nginx
        redis
        api
    """

    service = service.strip().lower()

    if service not in SERVICE_NAMES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Service '{service}' is not supported. "
                f"Supported services: {', '.join(SERVICE_NAMES)}."
            ),
        )

    state_before = copy.deepcopy(state)

    now = datetime.now(timezone.utc).isoformat()

    state["services"][service]["status"] = "running"
    state["services"][service]["last_restart"] = now
    state["services"][service]["restart_count"] += 1

    _record(
        "restart_service",
        {"service": service},
        timestamp=now,
        state_before=state_before,
    )

    return {
        "status": "success",
        "service": service,
        "last_restart": now,
    }


# ---------------------------------------------------------------------------
# 4. rotate_api_key()
# ---------------------------------------------------------------------------

@app.post("/rotate_api_key")
def rotate_api_key():
    """
    Generate a new API key and replace the existing one.
    The actual key is intentionally not returned.
    """

    state_before = copy.deepcopy(state)

    new_key = secrets.token_hex(32)
    rotated_at = datetime.now(timezone.utc).isoformat()

    state["api_key"] = new_key
    state["api_key_last_rotated"] = rotated_at

    _record(
        "rotate_api_key",
        {},
        state_before=state_before,
    )

    return {
        "status": "success",
        "api_key_rotated": True,
        "api_key_last_rotated": rotated_at,
    }


# ---------------------------------------------------------------------------
# 5. scale_replicas(n)
# ---------------------------------------------------------------------------

@app.post("/scale_replicas")
def scale_replicas(n: int):
    """
    Scale the number of running replicas.
    """

    if n < MIN_REPLICAS or n > MAX_REPLICAS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Replica count must be between "
                f"{MIN_REPLICAS} and {MAX_REPLICAS}."
            ),
        )

    current_replicas = state["replicas"]

    # Detect unnecessary repeated call
    if current_replicas == n:
        return {
            "status": "no_op",
            "message": f"Replica count is already {n}.",
            "replicas": current_replicas,
        }

    state_before = copy.deepcopy(state)

    previous_replicas = current_replicas
    state["replicas"] = n

    _record(
        "scale_replicas",
        {"n": n},
        state_before=state_before,
    )

    return {
        "status": "success",
        "previous_replicas": previous_replicas,
        "replicas": state["replicas"],
    }


# ---------------------------------------------------------------------------
# 6. get_metrics(service)
# ---------------------------------------------------------------------------

@app.get("/get_metrics")
def get_metrics(service: str):
    """
    Return CPU and memory metrics for a service.
    """

    service = service.strip().lower()

    if service not in SERVICE_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Service '{service}' is not supported.",
        )

    state_before = copy.deepcopy(state)

    _record(
        "get_metrics",
        {"service": service},
        state_before=state_before,
    )

    return {
        "service": service,
        "metrics": state["metrics"][service],
    }


# ---------------------------------------------------------------------------
# 7. rollback_deploy()
# ---------------------------------------------------------------------------

@app.post("/rollback_deploy")
def rollback_deploy():
    """
    Swap the current deployment version with the previous version.
    """

    state_before = copy.deepcopy(state)

    current = state["deployment"]["current_version"]
    previous = state["deployment"]["previous_version"]

    state["deployment"]["current_version"] = previous
    state["deployment"]["previous_version"] = current

    _record(
        "rollback_deploy",
        {},
        state_before=state_before,
    )

    return {
        "status": "success",
        "rolled_back_to": previous,
    }


# ---------------------------------------------------------------------------
# 8. get_logs(service, limit=None)
# ---------------------------------------------------------------------------

@app.get("/get_logs")
def get_logs(
    service: str,
    limit: Optional[int] = None,
):
    """
    Return recent log lines for a service.

    Read-only operation.
    """

    service = service.strip().lower()

    if service not in SERVICE_NAMES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Service '{service}' is not supported. "
                f"Supported services: {', '.join(SERVICE_NAMES)}."
            ),
        )

    logs = state["logs"][service]

    if limit is not None:
        if limit < 1:
            raise HTTPException(
                status_code=400,
                detail="limit must be a positive integer.",
            )

        logs = logs[-limit:]

    state_before = copy.deepcopy(state)

    _record(
        "get_logs",
        {
            "service": service,
            "limit": limit,
        },
        state_before=state_before,
    )

    return {
        "service": service,
        "logs": logs,
    }


# ---------------------------------------------------------------------------
# 9. kill_process(pid)
# ---------------------------------------------------------------------------

@app.post("/kill_process")
def kill_process(pid: int):
    """
    Kill a running process by PID.
    """

    process = state["processes"].get(pid)

    if process is None:
        raise HTTPException(
            status_code=404,
            detail=f"No process found with pid {pid}.",
        )

    # Detect unnecessary repeated call
    if process["status"] == "killed":
        return {
            "status": "no_op",
            "message": f"Process {pid} is already killed.",
            "pid": pid,
        }

    state_before = copy.deepcopy(state)

    previous_status = process["status"]
    now = datetime.now(timezone.utc).isoformat()

    process["status"] = "killed"
    process["killed_at"] = now

    _record(
        "kill_process",
        {"pid": pid},
        state_before=state_before,
    )

    return {
        "status": "success",
        "pid": pid,
        "service": process["service"],
        "previous_status": previous_status,
        "killed_at": now,
    }


# ---------------------------------------------------------------------------
# 10. set_config(key, value)
# ---------------------------------------------------------------------------

@app.post("/set_config")
def set_config(
    key: str,
    value: str,
):
    """
    Set or overwrite a configuration value.
    """

    key = key.strip()

    if not key:
        raise HTTPException(
            status_code=400,
            detail="key must not be empty.",
        )

    state_before = copy.deepcopy(state)

    previous_value = state["config"].get(key)

    state["config"][key] = value

    _record(
        "set_config",
        {
            "key": key,
            "value": value,
        },
        state_before=state_before,
    )

    return {
        "status": "success",
        "key": key,
        "value": value,
        "previous_value": previous_value,
    }


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

@app.get("/history")
def get_history():
    """Return all recorded tool calls."""

    return state["history"]