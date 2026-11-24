from typing import Any

from memory import MemoryStore

from .base import Tool


class SearchMemoryTool(Tool):
    name = "search_memory"
    description = "按关键词查询短期和长期记忆，最多返回 5 条。结果包含可用于修改和淘汰的记忆 ID。"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "选择可能出现在记忆原文中的短词组，省略时查看优先级最高的记忆。",
            },
        },
        "required": [],
        "additionalProperties": False,
    }

    def __init__(self, memory: MemoryStore) -> None:
        self._memory = memory

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = arguments.get("query", "")
        if not isinstance(query, str) or arguments.keys() - {"query"}:
            return {"error": "search_memory 只接受字符串参数 query"}
        return {"memories": self._memory.search(query.strip())}
