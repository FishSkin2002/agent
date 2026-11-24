import json
from collections.abc import Callable, Iterable, Iterator
from http.client import HTTPException
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from tools.registry import ToolRegistry


DEFAULT_MAX_OUTPUT_TOKENS = 2048
Message = dict[str, Any]


def _event_data(lines: Iterable[bytes]) -> Iterator[str]:
    data = []
    for raw_line in lines:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if not line and data:
            yield "\n".join(data)
            data.clear()
        field, _, value = line.partition(":")
        if field == "data":
            data.append(value.removeprefix(" "))
    if data:
        yield "\n".join(data)


def _merge_tool_calls(calls: dict[int, Message], deltas: list[Message]) -> None:
    for position, delta in enumerate(deltas):
        index = delta.get("index", position)
        call = calls.setdefault(index, {"type": "function", "function": {"arguments": ""}})
        for key in ("id", "type"):
            if delta.get(key) is not None:
                call[key] = delta[key]
        function = delta.get("function") or {}
        if function.get("name"):
            call["function"]["name"] = function["name"]
        call["function"]["arguments"] += function.get("arguments") or ""


def _read_stream(
    lines: Iterable[bytes], on_text: Callable[[str], None] | None, on_reasoning: Callable[[str], None] | None,
) -> tuple[Message, str | None]:
    text = []
    reasoning = []
    calls: dict[int, Message] = {}
    finish_reason = None
    for data in _event_data(lines):
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        if "error" in chunk:
            raise RuntimeError("API 返回流式错误，回复已中断")
        if not (choices := chunk.get("choices")):
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}
        _merge_tool_calls(calls, delta.get("tool_calls") or [])
        if thought := delta.get("reasoning_content"):
            if not isinstance(thought, str):
                raise ValueError("invalid reasoning delta")
            reasoning.append(thought)
            if on_reasoning is not None:
                on_reasoning(thought)
        if content := delta.get("content"):
            if not isinstance(content, str):
                raise ValueError("invalid content delta")
            text.append(content)
            if on_text is not None and not calls:
                on_text(content)
        finish_reason = choice.get("finish_reason") or finish_reason

    message: Message = {"content": "".join(text) or None}
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    if calls:
        message["tool_calls"] = [calls[index] for index in sorted(calls)]
    return message, finish_reason


def call_model(
    url: str, model: str, api_key: str, messages: list[Message], registry: ToolRegistry, *,
    max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS, on_text: Callable[[str], None] | None = None,
    on_reasoning: Callable[[str], None] | None = None,
) -> Message:
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "thinking": {"type": "enabled"},
        "max_tokens": max_tokens,
    }
    if definitions := registry.definitions():
        payload["tools"] = definitions
        if model != "deepseek-flash":
            payload["tool_choice"] = "auto"
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            if response.headers.get_content_type() == "text/event-stream":
                message, finish_reason = _read_stream(response, on_text, on_reasoning)
            else:
                choice = json.loads(response.read())["choices"][0]
                message, finish_reason = choice["message"], choice["finish_reason"]
                if isinstance(message, dict) and on_reasoning is not None:
                    if isinstance(thought := message.get("reasoning_content"), str) and thought:
                        on_reasoning(thought)
    except HTTPError as error:
        error.close()
        raise RuntimeError(f"API 返回 HTTP {error.code}") from error
    except (OSError, HTTPException) as error:
        raise RuntimeError("请求无法发送或响应读取失败") from error
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as error:
        raise RuntimeError("API 响应格式有误") from error

    if finish_reason == "length":
        raise RuntimeError("回复达到长度上限，请调大 .env 中的 MAX_OUTPUT_TOKENS 或要求简短回答")
    if finish_reason not in ("stop", "tool_calls") or not isinstance(message, dict):
        raise RuntimeError("API 未返回完整的回复")

    reply = message.get("content")
    assistant: Message = {"role": "assistant", "content": reply}
    if isinstance(thought := message.get("reasoning_content"), str) and thought:
        assistant["reasoning_content"] = thought
    tool_calls = message.get("tool_calls")
    if tool_calls or finish_reason == "tool_calls":
        if not isinstance(tool_calls, list) or not tool_calls:
            raise RuntimeError("API 未返回有效的工具调用列表")
        assistant["tool_calls"] = tool_calls
        return assistant

    if not isinstance(reply, str) or not reply.strip():
        raise RuntimeError("API 未返回完整的文本回复")
    return assistant
