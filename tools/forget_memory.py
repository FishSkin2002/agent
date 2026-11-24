from typing import Any

from memory import MemoryStore

from .base import Tool


class ForgetMemoryTool(Tool):
    name = "forget_memory"
    description = "按 ID 淘汰已过时、错误或重复的短期或长期记忆。先查询并核对要淘汰的记录。"
    parameters = {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "integer",
                "minimum": 1,
                "description": "查询得到的目标记忆 ID。",
            },
        },
        "required": ["memory_id"],
        "additionalProperties": False,
    }

    def __init__(self, memory: MemoryStore) -> None:
        self._memory = memory

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        memory_id = arguments.get("memory_id")
        if type(memory_id) is not int or not 1 <= memory_id < 2**63 or len(arguments) != 1:
            return {"error": "forget_memory 只接受有效的正整数参数 memory_id"}
        return {"forgotten": self._memory.forget(memory_id), "memory_id": memory_id}
