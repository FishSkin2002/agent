from typing import Any

from .base import Tool


class CreatePlanTool(Tool):
    name = "create_plan"
    description = (
        "为需要多步取证或比较多篇论文的问题制定本轮计划。"
        "按依赖顺序列出步骤，系统会逐步交给模型执行。每轮只能创建一个计划。"
        "能直接回答或一次检索就能解决的问题可以跳过计划。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "按执行顺序列出 2 到 10 个具体的子任务。",
            },
        },
        "required": ["steps"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self.steps: tuple[str, ...] = ()

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.steps:
            return {"error": "本轮已有计划，请继续当前步骤"}
        steps = arguments.get("steps")
        if (
            arguments.keys() != {"steps"}
            or not isinstance(steps, list)
            or not 2 <= len(steps) <= 10
            or any(not isinstance(step, str) or not step.strip() for step in steps)
        ):
            return {"error": "请提供 2 到 10 个非空的计划步骤"}
        self.steps = tuple(step.strip() for step in steps)
        return {"steps": self.steps}
