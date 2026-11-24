from typing import Any

from .base import Tool
from .registry import ToolRegistry


class ListToolsTool(Tool):
    name = "list_tools"
    description = "查询当前可用工具及其用途和参数。"
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if arguments:
            return {"error": "list_tools 不接受参数"}
        return {"tools": [definition["function"] for definition in self._registry.definitions()]}
