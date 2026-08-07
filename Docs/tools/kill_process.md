{
  "name": "kill_process",
  "parameters": {
    "type": "object",
    "properties": {
      "pid": {"type": "integer", "description": "process ID, e.g. from get_metrics/get_logs context"}
    },
    "required": ["pid"]
  }
}