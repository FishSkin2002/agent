import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .paper_retrieval import MAX_QUERIES, search_papers_tool


@dataclass(frozen=True)
class Tool:
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]


def list_tools(arguments: dict[str, Any]) -> dict[str, Any]:
    if arguments:
        return {"error": "list_tools 不接受参数"}
    return {"tools": [definition["function"] for definition in tool_definitions()]}


TOOL_REGISTRY = {
    "list_tools": Tool(
        description="查询当前可用工具及其用途和参数。",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        handler=list_tools,
    ),
    "search_papers": Tool(
        description="在线检索 arXiv 论文，返回标题、作者、摘要和版本链接。",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": '主查询，用英文关键词或 arXiv 查询式，例如 ti:"Synergizing Reasoning and Acting"。',
                },
                "alternatives": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": MAX_QUERIES - 1,
                    "description": "补充查询，可填写其他拼写、近义词或不同的关键词组合。",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        handler=search_papers_tool,
    ),
}


def tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for name, tool in TOOL_REGISTRY.items()
    ]


def execute_tool(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function")
    if call.get("type") != "function" or not isinstance(function, dict):
        return {"error": "无效的函数调用"}

    name = function.get("name")
    if not isinstance(name, str) or name not in TOOL_REGISTRY:
        return {"error": "未知工具，可以调用 list_tools 查看当前工具清单"}

    try:
        arguments = json.loads(function["arguments"])
    except (KeyError, TypeError, ValueError):
        return {"error": "工具参数必须是 JSON 对象"}
    if not isinstance(arguments, dict):
        return {"error": "工具参数必须是 JSON 对象"}

    return TOOL_REGISTRY[name].handler(arguments)
