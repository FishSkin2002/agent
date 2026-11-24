from typing import Any

from .base import Tool
from .paper_retrieval import SearchPapers


class ListTools(Tool):
    name = "list_tools"
    description = "查询当前可用工具及其用途和参数。"
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if arguments:
            return {"error": "list_tools 不接受参数"}
        return {"tools": [definition["function"] for definition in tool_definitions()]}


TOOL_REGISTRY: dict[str, Tool] = {tool.name: tool for tool in (ListTools(), SearchPapers())}


def tool_definitions() -> list[dict[str, Any]]:
    return [tool.definition() for tool in TOOL_REGISTRY.values()]


def execute_tool(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function")
    if call.get("type") != "function" or not isinstance(function, dict):
        return {"error": "无效的函数调用"}

    name = function.get("name")
    if not isinstance(name, str) or name not in TOOL_REGISTRY:
        return {"error": "未知工具，可以调用 list_tools 查看当前工具清单"}

    return TOOL_REGISTRY[name].invoke(function.get("arguments"))
