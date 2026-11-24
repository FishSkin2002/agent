import json
import unittest
from email.message import Message
from unittest.mock import patch

from llm.client import call_model
from tools.list_tools import ListToolsTool
from tools.registry import ToolRegistry


class StreamResponse:
    def __init__(self, lines):
        self.lines = lines
        self.headers = Message()
        self.headers["Content-Type"] = "text/event-stream; charset=utf-8"

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        return False

    def __iter__(self):
        return iter(self.lines)


def event(delta, finish_reason=None):
    chunk = {"choices": [{"delta": delta, "finish_reason": finish_reason}]}
    return [f"data: {json.dumps(chunk, ensure_ascii=False)}\r\n".encode(), b"\r\n"]


class ModelStreamTests(unittest.TestCase):
    def test_text_arrives_before_the_final_event(self):
        received = []
        thoughts = []

        def lines():
            yield b": keep-alive\n"
            yield b"\n"
            yield from event({"reasoning_content": "先考虑"})
            self.assertEqual(thoughts, ["先考虑"])
            yield from event({"reasoning_content": "相关证据。"})
            self.assertEqual(received, [])
            yield from event({"content": "第一段"})
            self.assertEqual(received, ["第一段"])
            yield from event({"content": "第二段"}, "stop")
            yield b'data: {"choices": [], "usage": {"prompt_tokens": 5}}\n'
            yield b"\n"
            yield b"data: [DONE]\n"
            yield b"\n"

        for model in ("deepseek-flash", "glm-4.7-flash"):
            with self.subTest(model=model), patch("llm.client.urlopen", return_value=StreamResponse(lines())) as request:
                received.clear()
                thoughts.clear()
                reply = call_model(
                    "https://example.invalid", model, "test-key", [], ToolRegistry(),
                    max_tokens=3072, on_text=received.append, on_reasoning=thoughts.append,
                )
                payload = json.loads(request.call_args.args[0].data)
                self.assertIs(payload["stream"], True)
                self.assertEqual(payload["max_tokens"], 3072)
                self.assertEqual(payload["thinking"], {"type": "enabled"})
                self.assertEqual(received, ["第一段", "第二段"])
                self.assertEqual(thoughts, ["先考虑", "相关证据。"])
                self.assertEqual(reply, {
                    "role": "assistant", "content": "第一段第二段", "reasoning_content": "先考虑相关证据。",
                })

    def test_tool_fragments_are_reassembled_before_dispatch(self):
        registry = ToolRegistry()
        registry.register(ListToolsTool(registry))
        lines = [
            *event({"reasoning_content": "先查看工具。"}),
            *event({"tool_calls": [
                {"index": 0, "id": "call-a", "type": "function", "function": {"name": "list_tools", "arguments": "{"}},
                {"index": 1, "id": "call-b", "type": "function", "function": {"name": "list_tools", "arguments": "{}"}},
            ]}),
            *event({"tool_calls": [{"index": 0, "function": {"arguments": "}"}}]}),
            *event({}, "tool_calls"),
        ]
        received = []
        with patch("llm.client.urlopen", return_value=StreamResponse(lines)):
            reply = call_model(
                "https://example.invalid", "deepseek-flash", "test-key", [], registry, on_text=received.append,
            )
        self.assertEqual(received, [])
        self.assertEqual(reply["reasoning_content"], "先查看工具。")
        self.assertEqual([call["id"] for call in reply["tool_calls"]], ["call-a", "call-b"])
        for call in reply["tool_calls"]:
            self.assertEqual(call["function"]["arguments"], "{}")
            self.assertIn("tools", registry.invoke(call))

    def test_length_error_points_to_the_output_setting(self):
        with (
            patch("llm.client.urlopen", return_value=StreamResponse(event({"content": "未结束"}, "length"))),
            self.assertRaisesRegex(RuntimeError, "MAX_OUTPUT_TOKENS"),
        ):
            call_model("https://example.invalid", "deepseek-flash", "test-key", [], ToolRegistry())
