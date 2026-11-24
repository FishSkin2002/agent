import json
from abc import ABC, abstractmethod
from typing import Any, ClassVar


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    parameters: ClassVar[dict[str, Any]]

    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def invoke(self, arguments_json: str | None) -> dict[str, Any]:
        if not isinstance(arguments_json, str):
            return {"error": "工具参数必须是 JSON 对象"}
        try:
            arguments = json.loads(arguments_json)
        except ValueError:
            return {"error": "工具参数必须是 JSON 对象"}
        if not isinstance(arguments, dict):
            return {"error": "工具参数必须是 JSON 对象"}
        return self.run(arguments)

    @abstractmethod
    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """校验本工具的参数、执行操作并返回结果。"""
