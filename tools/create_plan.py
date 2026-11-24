from collections.abc import Callable
from typing import Any

from .base import Tool


class CreatePlanTool(Tool):
    name = "create_plan"
    description = (
        "为需要多步取证或比较多篇论文的问题启动计划阶段。"
        "系统会制定本轮计划并逐步交给模型执行。每轮只能创建一个计划。"
        "能直接回答或一次检索就能解决的问题可以跳过计划。"
    )
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def __init__(self, generate_plan: Callable[[], str]) -> None:
        self._generate_plan = generate_plan
        self.steps: tuple[str, ...] = ()

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.steps:
            return {"error": "本轮已有计划，请继续当前步骤"}
        if arguments:
            return {"error": "启动计划阶段时请传入空对象"}
        try:
            plan = self._generate_plan()
        except RuntimeError as error:
            return {"error": f"计划生成失败（{error}）"}

        lines = [line.strip() for line in plan.splitlines() if line.strip()]
        if not 2 <= len(lines) <= 10:
            return {"error": "计划应包含 2 到 10 个步骤"}
        steps = []
        for index, line in enumerate(lines, start=1):
            number, _, step = line.partition(".")
            if number != str(index) or not step.strip():
                return {"error": "计划步骤应按顺序编号，每行一个非空步骤"}
            steps.append(step.strip())
        self.steps = tuple(steps)
        return {"steps": self.steps}
