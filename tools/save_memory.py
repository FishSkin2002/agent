from typing import Any

from memory import LONG_TERM_IMPORTANCE, MemoryStore

from .base import Tool


class SaveMemoryTool(Tool):
    name = "save_memory"
    description = (
        "保存或修改对后续对话有用的信息，例如用户偏好、任务约束或已确认的事实。"
        "先查询相关记忆，修改已有记录时传入其 ID。"
        f"重要性达到 {LONG_TERM_IMPORTANCE} 时晋升为长期记忆。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "简洁、完整的记忆内容。"},
            "importance": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "根据稳定性和未来价值评估的重要性，范围 0 到 1。",
            },
            "memory_id": {
                "type": "integer",
                "minimum": 1,
                "description": "修改已有记忆时填写查询得到的 ID，新增时省略。",
            },
        },
        "required": ["content", "importance"],
        "additionalProperties": False,
    }

    def __init__(self, memory: MemoryStore) -> None:
        self._memory = memory

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        content = arguments.get("content")
        importance = arguments.get("importance")
        memory_id = arguments.get("memory_id")
        if not isinstance(content, str) or not content.strip():
            return {"error": "content 必须是非空字符串"}
        if type(importance) not in (int, float) or not 0 <= importance <= 1:
            return {"error": "importance 必须是 0 到 1 的数值"}
        if "memory_id" in arguments and (type(memory_id) is not int or not 1 <= memory_id < 2**63):
            return {"error": "memory_id 必须是有效的正整数"}
        if arguments.keys() - self.parameters["properties"].keys():
            return {"error": "save_memory 收到了未知参数"}

        record = self._memory.save(content.strip(), importance, memory_id)
        if record is None:
            return {"error": "未找到该记忆，请重新查询"}
        return {"memory": record}
