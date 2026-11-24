import json
import sys
from http.client import HTTPException
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from tools.registry import execute_tool, tool_definitions


MODEL_ENDPOINTS = {
    "glm-4.7-flash": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
    "deepseek-flash": "https://api.deepseek.com/chat/completions",
}
LEGACY_KEY_MODELS = {
    "GLM_API_KEY": "glm-4.7-flash",
    "DEEPSEEK_API_KEY": "deepseek-flash",
}
MAX_TOOL_CALLS = 3
SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "你是论文检索助手，也可以回答一般概念问题。"
        "根据问题决定直接回答还是调用工具。可以查询可用工具，并根据描述和参数选择。"
        "用获取的证据回答并引用来源。查不到证据或证据不足时，请如实说明。"
    ),
}
Message = dict[str, Any]


def load_config(path: Path) -> tuple[str, str, str]:
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
    return MODEL_ENDPOINTS[model], model, api_key


def call_model(url: str, model: str, api_key: str, messages: list[Message]) -> Message:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "thinking": {"type": "disabled"},
        "max_tokens": 1024,
        "tools": tool_definitions(),
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
    url: str, model: str, api_key: str, history: list[Message], question: str
) -> list[Message]:
    turn = [{"role": "user", "content": question}]
    remaining_calls = MAX_TOOL_CALLS

    while True:
        message = call_model(url, model, api_key, [SYSTEM_MESSAGE] + history + turn)
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
            result = execute_tool(call)
            turn.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(result, ensure_ascii=False),
            })


def main() -> int:
    try:
        url, model, api_key = load_config(Path(__file__).with_name(".env"))
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1

    history: list[Message] = []
    print(f"当前模型为 {model}。输入 /exit 结束对话。")
    while True:
        question = input("你 > ").strip()
        if question == "/exit":
            return 0
        if not question:
            continue

        try:
            turn = run_turn(url, model, api_key, history, question)
        except RuntimeError as error:
            print(f"调用失败（{error}）。可以重新输入后再试。", file=sys.stderr)
            continue

        history.extend(turn)
        print(f"助手 > {turn[-1]['content']}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EOFError, KeyboardInterrupt):
        print()
