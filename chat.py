import json
import os
import sqlite3
import sys
from collections import deque
from contextlib import ExitStack, closing
from http.client import HTTPException
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from memory import DEFAULT_SHORT_TERM_TURNS, MemoryStore
from tools.forget_memory import ForgetMemoryTool
from tools.list_tools import ListToolsTool
from tools.mcp_client import MCPConfigError, register_mcp_tools
from tools.paper_library import PaperLibrary
from tools.registry import ToolRegistry
from tools.save_memory import SaveMemoryTool
from tools.search_memory import SearchMemoryTool
from tools.search_paper_library import SearchPaperLibraryTool
from tools.search_paper_passages import SearchPaperPassagesTool
from tools.search_papers import SearchPapersTool


MODEL_ENDPOINTS = {
    "glm-4.7-flash": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
    "deepseek-flash": "https://api.deepseek.com/chat/completions",
}
LEGACY_KEY_MODELS = {
    "GLM_API_KEY": "glm-4.7-flash",
    "DEEPSEEK_API_KEY": "deepseek-flash",
}
MAX_TOOL_CALLS = 10
SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "你是论文检索助手，也可以回答一般概念问题。"
        "根据问题决定直接回答还是调用工具。可以查询可用工具，并根据描述和参数选择。"
        "用获取的证据回答并引用来源。查不到证据或证据不足时，请如实说明。"
        "回答论文方法或实验细节时，查找论文原文片段，并引用返回的版本和页码。"
        "需要以前的信息时主动查找记忆。主动保存对后续对话有用的信息并评估重要性，"
        "发现已有记忆需要更正或已经失效时，自主修改或淘汰。"
    ),
}
Message = dict[str, Any]


def report_progress(stage: str, message: str) -> None:
    print(f"[{stage}] {message}", file=sys.stderr, flush=True)


def read_question() -> str:
    label = "【 用户 】"
    if sys.stdout.isatty() and os.environ.get("TERM") != "dumb" and "NO_COLOR" not in os.environ:
        label = "\x1b[1;30;106m  用户  \x1b[0m"
    print(f"\n{label} ❯ ", end="", flush=True)
    return input().strip()


def create_tool_registry(memory: MemoryStore, papers: PaperLibrary) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ListToolsTool(registry))
    registry.register(SearchPapersTool(papers))
    registry.register(SearchPaperLibraryTool(papers))
    registry.register(SearchPaperPassagesTool(papers))
    registry.register(SaveMemoryTool(memory))
    registry.register(SearchMemoryTool(memory))
    registry.register(ForgetMemoryTool(memory))
    return registry


def load_config(path: Path) -> tuple[str, str, str, int]:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError("请在 chat.py 同目录准备可读的 UTF-8 格式 .env 文件。") from error

    values = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        if separator:
            values[name.strip()] = value.strip()

    model = values.get("MODEL", "")
    api_key = values.get("API_KEY", "")

    # 已有 .env 只填了一个旧密钥时，继续使用该密钥对应的模型。
    if "MODEL" not in values and "API_KEY" not in values:
        legacy_keys = [name for name in LEGACY_KEY_MODELS if values.get(name)]
        if len(legacy_keys) == 1:
            key_name = legacy_keys[0]
            model = LEGACY_KEY_MODELS[key_name]
            api_key = values[key_name]

    if model not in MODEL_ENDPOINTS:
        supported_models = "、".join(MODEL_ENDPOINTS)
        raise ValueError(f"请在 .env 的 MODEL 中选择 {supported_models}。")
    if not api_key:
        raise ValueError("请在 .env 的 API_KEY 中填写所选模型对应的密钥。")
    try:
        short_term_turns = int(values.get("MEMORY_SHORT_TERM_TURNS", DEFAULT_SHORT_TERM_TURNS))
    except ValueError as error:
        raise ValueError("请在 .env 的 MEMORY_SHORT_TERM_TURNS 中填写正整数。") from error
    if not 1 <= short_term_turns < 2**63:
        raise ValueError("请在 .env 的 MEMORY_SHORT_TERM_TURNS 中填写有效的正整数。")
    return MODEL_ENDPOINTS[model], model, api_key, short_term_turns


def call_model(
    url: str, model: str, api_key: str, messages: list[Message], registry: ToolRegistry
) -> Message:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "thinking": {"type": "disabled"},
        "max_tokens": 1024,
        "tools": registry.definitions(),
        "tool_choice": "auto",
    }
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            body = response.read()
    except HTTPError as error:
        error.close()
        raise RuntimeError(f"API 返回 HTTP {error.code}") from error
    except (OSError, HTTPException, ValueError) as error:
        raise RuntimeError("请求无法发送或响应读取失败") from error

    try:
        choice = json.loads(body)["choices"][0]
        message = choice["message"]
        finish_reason = choice["finish_reason"]
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise RuntimeError("API 响应格式有误") from error

    if finish_reason == "length":
        raise RuntimeError("回复达到输出长度上限，请缩短问题或要求简短回答")
    if finish_reason not in ("stop", "tool_calls") or not isinstance(message, dict):
        raise RuntimeError("API 未返回完整的回复")

    reply = message.get("content")
    tool_calls = message.get("tool_calls")
    if tool_calls or finish_reason == "tool_calls":
        if not isinstance(tool_calls, list) or not tool_calls:
            raise RuntimeError("API 未返回有效的工具调用列表")
        return {"role": "assistant", "content": reply, "tool_calls": tool_calls}

    if not isinstance(reply, str) or not reply.strip():
        raise RuntimeError("API 未返回完整的文本回复")
    return {"role": "assistant", "content": reply}


def run_turn(
    url: str, model: str, api_key: str, history: list[Message], question: str, registry: ToolRegistry
) -> list[Message]:
    turn = [{"role": "user", "content": question}]
    remaining_calls = MAX_TOOL_CALLS

    while True:
        if remaining_calls == MAX_TOOL_CALLS:
            report_progress("模型", f"正在等待 {model} 处理问题")
        else:
            report_progress("模型", f"正在等待 {model} 处理工具结果")
        message = call_model(url, model, api_key, [SYSTEM_MESSAGE] + history + turn, registry)
        turn.append(message)
        tool_calls = message.get("tool_calls", [])
        if not tool_calls:
            return turn
        if len(tool_calls) > remaining_calls:
            raise RuntimeError("本轮已达到工具调用上限，请缩小问题范围后重试")
        remaining_calls -= len(tool_calls)

        # 每个调用必须有一个能与结果对应的 ID。
        try:
            call_ids = [call["id"] for call in tool_calls]
        except (KeyError, TypeError) as error:
            raise RuntimeError("API 缺少工具调用 ID") from error
        if any(not isinstance(call_id, str) or not call_id for call_id in call_ids):
            raise RuntimeError("API 返回了无效的工具调用 ID")
        if len(set(call_ids)) != len(call_ids):
            raise RuntimeError("API 返回了重复的工具调用 ID")

        for call, call_id in zip(tool_calls, call_ids):
            function = call.get("function")
            name = function.get("name") if isinstance(function, dict) else None
            tool_name = name if isinstance(name, str) and name.isprintable() else "未知工具"
            report_progress("工具", f"正在执行 {tool_name}")
            result = registry.invoke(call)
            if "error" in result or result.get("is_error"):
                report_progress("工具", f"{tool_name} 返回错误")
            elif isinstance(papers := result.get("papers"), list):
                report_progress("工具", f"{tool_name} 返回 {len(papers)} 篇论文")
            else:
                report_progress("工具", f"{tool_name} 已返回结果")
            turn.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(result, ensure_ascii=False),
            })


def chat(url: str, model: str, api_key: str, memory: MemoryStore, registry: ToolRegistry) -> int:
    history: deque[list[Message]] = deque(maxlen=memory.short_term_turns)
    print(f"当前模型为 {model}。输入 /exit 结束对话。")
    while True:
        question = read_question()
        if question == "/exit":
            return 0
        if not question:
            continue

        recent_messages = [message for recent_turn in history for message in recent_turn]
        try:
            with memory.turn():
                turn = run_turn(url, model, api_key, recent_messages, question, registry)
        except (RuntimeError, sqlite3.Error) as error:
            print(f"调用失败（{error}）。可以重新输入后再试。", file=sys.stderr)
            continue

        history.append(turn)
        print(f"\n【助手】\n{turn[-1]['content']}")


def main() -> int:
    project_dir = Path(__file__).parent
    try:
        url, model, api_key, short_term_turns = load_config(project_dir / ".env")
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1

    data_dir = project_dir / ".agent-data"
    with ExitStack() as stack:
        try:
            data_dir.mkdir(exist_ok=True)
            memory = stack.enter_context(closing(MemoryStore(data_dir / "memory.sqlite3", short_term_turns)))
            papers = stack.enter_context(closing(PaperLibrary(data_dir / "papers.sqlite3")))
        except (OSError, sqlite3.Error) as error:
            print(f"无法打开本地数据库（{error}）。", file=sys.stderr)
            return 1

        registry = create_tool_registry(memory, papers)
        try:
            stack.enter_context(register_mcp_tools(project_dir / "mcp.json", registry))
        except MCPConfigError as error:
            print(error, file=sys.stderr)
            return 1
        return chat(url, model, api_key, memory, registry)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EOFError, KeyboardInterrupt):
        print()
