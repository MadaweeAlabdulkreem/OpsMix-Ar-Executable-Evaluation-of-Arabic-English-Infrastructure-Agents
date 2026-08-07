"""In-memory state for the tiny infrastructure service.

This module IS the "server." There is no real disk, cache, or
services here -- just plain Python data structures that our three
tools read and mutate to simulate what a real ops action would do.
"""

# Names of the three services this tiny server "runs".
# A tuple, not a list, because this set of names never changes at runtime.
SERVICE_NAMES = ("nginx", "redis", "api")


def _initial_state() -> dict:
    """Build one fresh copy of the starting state.

    This is a function -- not a bare dictionary -- so that later
    (outside today's scope) something could call it again to reset
    the server back to a known starting point.
    """
    return {
        "disk_total_gb": 10, # Total disk capacity
        "disk_used_gb": 8.2, # Current disk usage
        "cache_size_mb": 512,

        # New state for the additional tools
        # API key state
        "api_key": "initial-api-key",
        "api_key_last_rotated": None,

        # Scaling state
        "replicas": 1,

        # Service state
        "services": {
            name: {
                "status": "running",
                "last_restart": None,
                "restart_count": 0,
            }
            for name in SERVICE_NAMES
        },
    }


# The single source of truth for the whole service.
# main.py will import THIS object directly, and every tool function
# will read from and write to it -- never create its own copy.
state = _initial_state()