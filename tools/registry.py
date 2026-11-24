from typing import Any

from .base import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具 {tool.name} 已注册")
        self._tools[tool.name] = tool

    def definitions(self) -> list[dict[str, Any]]:
        return [tool.definition() for tool in self._tools.values()]

    def invoke(self, call: dict[str, Any]) -> dict[str, Any]:
        function = call.get("function")
        if call.get("type") != "function" or not isinstance(function, dict):
            return {"error": "无效的函数调用"}

        name = function.get("name")
        if not isinstance(name, str) or name not in self._tools:
            return {"error": "未知工具，请查询当前工具清单"}

        return self._tools[name].invoke(function.get("arguments"))
