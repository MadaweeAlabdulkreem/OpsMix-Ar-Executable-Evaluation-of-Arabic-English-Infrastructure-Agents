{
  "name": "get_logs",
  "parameters": {
    "type": "object",
    "properties": {
      "service": {"type": "string", "enum": ["nginx", "redis", "api"]},
      "limit": {"type": "integer", "minimum": 1, "description": "optional, most recent N lines"}
    },
    "required": ["service"]
  }
}