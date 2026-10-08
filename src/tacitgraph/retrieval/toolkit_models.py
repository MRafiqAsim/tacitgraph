"""Result and tool-descriptor types shared by the retrieval toolkit."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    """Result from a tool execution."""

    tool_name: str
    success: bool
    data: Any
    message: str = ""
    execution_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "data": self.data,
            "message": self.message,
            "execution_time": self.execution_time,
        }


@dataclass
class Tool:
    """Definition of a tool available to the ReAct agent."""

    name: str
    description: str
    parameters: dict[str, dict[str, Any]]  # param_name -> {type, description, required}
    function: Callable

    def to_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function schema."""
        properties = {}
        required = []

        for param_name, param_info in self.parameters.items():
            properties[param_name] = {
                "type": param_info.get("type", "string"),
                "description": param_info.get("description", ""),
            }
            if param_info.get("required", False):
                required.append(param_name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        }
