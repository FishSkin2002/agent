import io
import json
import unittest
from contextlib import closing, redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import chat
from memory import MemoryStore
from tools.registry import ToolRegistry


class ChatProgressTests(unittest.TestCase):
    def test_waiting_stage_is_visible_before_the_model_and_tool_return(self):
        progress = io.StringIO()
        output = io.StringIO()
        query = 'ti:"On-Policy Self-Distillation"'

        def model_response(url, model, api_key, messages, registry):
            if messages[-1]["role"] == "user":
                self.assertIn("正在等待 deepseek-flash 处理问题", progress.getvalue())
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "search-1",
                        "type": "function",
                        "function": {
                            "name": "search_papers",
                            "arguments": json.dumps({"query": query}),
                        },
                    }],
                }
            self.assertIn("search_papers 返回 0 篇论文", progress.getvalue())
            self.assertIn("正在等待 deepseek-flash 处理工具结果", progress.getvalue())
            return {"role": "assistant", "content": "这次没有搜到论文。"}

        def tool_response(call):
            self.assertIn("正在执行 search_papers", progress.getvalue())
            self.assertEqual(json.loads(call["function"]["arguments"]), {"query": query})
            return {"papers": []}

        with TemporaryDirectory() as directory:
            registry = ToolRegistry()
            path = Path(directory) / "memory.sqlite3"
            with (
                closing(MemoryStore(path)) as memory,
                patch.object(chat, "call_model", side_effect=model_response),
                patch.object(registry, "invoke", side_effect=tool_response),
                patch("builtins.input", side_effect=[f"搜索 {query}", "/exit"]),
                redirect_stdout(output),
                redirect_stderr(progress),
            ):
                self.assertEqual(chat.chat("ignored", "deepseek-flash", "test-key", memory, registry), 0)

        self.assertIn("【 你 · 输入问题 】 ❯", output.getvalue())
        self.assertIn("【助手】\n这次没有搜到论文。", output.getvalue())
