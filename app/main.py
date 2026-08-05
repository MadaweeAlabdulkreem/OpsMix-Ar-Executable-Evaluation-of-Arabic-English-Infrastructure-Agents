"""FastAPI app exposing the tiny infra service's tools as HTTP endpoints.

Each tool from the project spec becomes exactly one endpoint here.
This step only adds check_disk() -- clear_cache() and
restart_service() come in later steps.
"""

from fastapi import FastAPI, HTTPException
from app.state import state, SERVICE_NAMES
from datetime import datetime, timezone

# resourse: https://fastapi.tiangolo.com/tutorial/body/
app = FastAPI(title="Tiny Infra Service")


# The first tool: check_disk() Return current disk usage. Read-only
@app.get("/check_disk")
def check_disk():
    disk_usage_percent = (state["disk_used_gb"] / state["disk_total_gb"]) * 100

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
 
    return {
        "status": "success",
        "service": service,
        "last_restart": now,
    }

# The fourth tool: rotate_api_key()
# Generate a new API key and replace the existing one.
import secrets

@app.post("/rotate_api_key")
def rotate_api_key():
    new_key = secrets.token_hex(32)

    state["api_key"] = new_key
    state["api_key_last_rotated"] = datetime.now(timezone.utc).isoformat()

    return {
        "status": "success",
        "api_key": new_key,
        "api_key_last_rotated": state["api_key_last_rotated"],
    }


# The fifth tool: scale_replicas(n)
# Scale the number of running replicas.
@app.post("/scale_replicas")
def scale_replicas(n: int):
    if n < 1:
        raise HTTPException(
            status_code=400,
            detail="Replica count must be at least 1."
        )

    state["replicas"] = n

    return {
        "status": "success",
        "replicas": state["replicas"],
    }