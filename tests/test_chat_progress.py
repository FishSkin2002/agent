import io
import json
import unittest
from contextlib import closing, redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import main
from memory import MemoryStore
from tools.registry import ToolRegistry


class ChatProgressTests(unittest.TestCase):
    def test_direct_reply_prints_chunks_before_the_model_returns(self):
        output = io.StringIO()
        progress = io.StringIO()

        def model_response(url, model, api_key, messages, registry, *, max_tokens, on_text, on_reasoning):
            self.assertEqual(max_tokens, 3072)
            on_reasoning("先分析")
            self.assertTrue(progress.getvalue().endswith("[思考]\n先分析"))
            on_reasoning("问题。")
            on_text("第一段")
            self.assertTrue(progress.getvalue().endswith("[思考]\n先分析问题。\n"))
            self.assertIn("【 助手 】 ❯ 第一段", output.getvalue())
            on_text("第二段")
            return {"role": "assistant", "content": "第一段第二段"}

        with (
            closing(MemoryStore(Path(":memory:"))) as memory,
            patch.object(main, "call_model", side_effect=model_response),
            patch("builtins.input", side_effect=["你好", "/exit"]),
            redirect_stdout(output),
            redirect_stderr(progress),
        ):
            self.assertEqual(
                main.chat("ignored", "deepseek-flash", "test-key", memory, ToolRegistry(), max_output_tokens=3072), 0,
            )
        self.assertEqual(output.getvalue().count("【 助手 】 ❯ 第一段第二段"), 1)

    def test_waiting_stage_is_visible_before_the_model_and_tool_return(self):
        progress = io.StringIO()
        output = io.StringIO()
        query = 'ti:"On-Policy Self-Distillation"'
        paper = {"paper_id": "2605.18740v4", "title": "Vision-OPD", "abstract": "Test abstract"}

        def model_response(url, model, api_key, messages, registry, *, max_tokens, on_text, on_reasoning):
            if not registry.definitions():
                self.assertIn("正在核对答复与检索证据", progress.getvalue())
                self.assertIsNone(on_text)
                self.assertEqual(json.loads(messages[-1]["content"])["draft"], "搜到了 Vision-OPD。")
                return {"role": "assistant", "content": '{"answer":"", "missing_evidence":""}'}
            if messages[-1]["role"] == "user":
                self.assertIn("正在等待 deepseek-flash 处理问题", progress.getvalue())
                on_reasoning("先检索论文。")
                return {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "先检索论文。",
                    "tool_calls": [{
                        "id": "search-1",
                        "type": "function",
                        "function": {
                            "name": "search_papers",
                            "arguments": json.dumps({"query": query}),
                        },
                    }],
                }
            self.assertIn("search_papers 返回 1 篇论文\n  Vision-OPD\n", progress.getvalue())
            self.assertNotIn(paper["paper_id"], progress.getvalue())
            self.assertIn("正在等待 deepseek-flash 处理工具结果", progress.getvalue())
            self.assertEqual(messages[-2]["reasoning_content"], "先检索论文。")
            self.assertEqual(json.loads(messages[-1]["content"]), {"papers": [paper]})
            self.assertIsNone(on_text)
            return {"role": "assistant", "content": "搜到了 Vision-OPD。"}

        def tool_response(call):
            self.assertIn("正在执行 search_papers", progress.getvalue())
            self.assertEqual(json.loads(call["function"]["arguments"]), {"query": query})
            return {"papers": [paper]}

        with TemporaryDirectory() as directory:
            registry = ToolRegistry()
            path = Path(directory) / "memory.sqlite3"
            with (
                closing(MemoryStore(path)) as memory,
                patch.object(main, "call_model", side_effect=model_response),
                patch.object(registry, "invoke", side_effect=tool_response),
                patch("builtins.input", side_effect=[f"搜索 {query}", "/exit"]),
                redirect_stdout(output),
                redirect_stderr(progress),
            ):
                self.assertEqual(main.chat("ignored", "deepseek-flash", "test-key", memory, registry), 0)

        self.assertIn("【 用户 】 ❯", output.getvalue())
        self.assertIn("【 助手 】 ❯ 搜到了 Vision-OPD。", output.getvalue())
