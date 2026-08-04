"""FastAPI app exposing the tiny infra service's tools as HTTP endpoints.

Each tool from the project spec becomes exactly one endpoint here.
This step only adds check_disk() -- clear_cache() and
restart_service() come in later steps.
"""
from __future__ import annotations
from fastapi import FastAPI, HTTPException
from app.state import state, SERVICE_NAMES
from datetime import datetime, timezone


# resourse: https://fastapi.tiangolo.com/tutorial/body/
app = FastAPI(title="Tiny Infra Service")

def _record(tool: str, args: dict, timestamp: str | None = None) -> None:
    
    state["history"].append({
        "tool": tool,
        "args": args,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
    })

# The first tool: check_disk() Return current disk usage. Read-only
@app.get("/check_disk")
def check_disk():
    disk_usage_percent = (state["disk_used_gb"] / state["disk_total_gb"]) * 100
    _record("check_disk", {})

    return {"disk_total_gb": state["disk_total_gb"],
             "disk_used_gb": state["disk_used_gb"],
             "disk_usage_percent": round(disk_usage_percent, 2),}


# The second tool: clear_cache() is a write operation Clear the cache: empty it and reduce disk usage accordingly.
# resource: https://onlinetoolsforge.com/en/tools/disk-usage-calculator/
@app.post("/clear_cache")
def clear_cache():
    freed_gb = state["cache_size_mb"] / 1024  # convert MB freed -> GB freed
 
    state["disk_used_gb"] = max(0, state["disk_used_gb"] - freed_gb)
    state["cache_size_mb"] = 0
    _record("check_disk", {})

    disk_usage_percent = (state["disk_used_gb"] / state["disk_total_gb"]) * 100
 
    return {
        "status": "success",
        "cache_size_mb": state["cache_size_mb"],
        "disk_used_gb": round(state["disk_used_gb"], 2),
        "disk_usage_percent": round(disk_usage_percent, 2),
    }


# The third tool: restart_service() Restart one of nginx, redis, or api. Rejects any other value.
@app.post("/restart_service")
def restart_service(service: str):
    service = service.strip().lower()
    if service not in SERVICE_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Service '{service}' is not supported. Supported services: {', '.join(SERVICE_NAMES)}."
        )
 
    now = datetime.now(timezone.utc).isoformat()
    state["services"][service]["status"] = "running"
    state["services"][service]["last_restart"] = now
    _record("check_disk", {})

    return {
        "status": "success",
        "service": service,
        "last_restart": now,
    }

@app.get("/history")
def get_history():
    return state["history"]