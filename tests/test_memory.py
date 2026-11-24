import io
import json
import unittest
from contextlib import closing, redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import chat
from memory import MemoryStore


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        data_dir = TemporaryDirectory()
        self.addCleanup(data_dir.cleanup)
        self.path = Path(data_dir.name) / "memory.sqlite3"

    def test_short_term_expires_after_completed_turns_across_restarts(self):
        with closing(MemoryStore(self.path, 2)) as memory:
            with memory.turn():
                memory.save("本次检索只看 ReAct 的 arXiv v3", 0.4)
                memory.save("以后优先找顶会论文", 0.8)

        with closing(MemoryStore(self.path, 2)) as memory:
            with self.assertRaisesRegex(RuntimeError, "模型请求失败"):
                with memory.turn():
                    memory.save("失败轮次中的内容", 0.6)
                    raise RuntimeError("模型请求失败")
            self.assertEqual(memory.search("失败轮次"), [])

            with memory.turn():
                self.assertEqual(len(memory.search("react")), 1)
            self.assertEqual(len(memory.search("react")), 1)

            with memory.turn():
                self.assertEqual(len(memory.search("react")), 1)
            self.assertEqual(memory.search("react"), [])
            self.assertEqual(memory.search("顶会")[0]["kind"], "long_term")

    def test_revision_promotes_and_both_kinds_can_be_forgotten(self):
        with closing(MemoryStore(self.path, 1)) as memory:
            with memory.turn():
                preference = memory.save("优先顶会", 0.6)
                temporary = memory.save("这次只看 arXiv", 0.5)
            with memory.turn():
                promoted = memory.save("以后优先检索顶会论文", 0.8, preference["id"])
                self.assertEqual(promoted["id"], preference["id"])
                self.assertEqual(promoted["kind"], "long_term")
                self.assertTrue(memory.forget(temporary["id"]))

            with memory.turn():
                self.assertEqual(memory.search("顶会"), [promoted])
                self.assertTrue(memory.forget(promoted["id"]))
                self.assertFalse(memory.forget(promoted["id"]))
            self.assertEqual(memory.search(""), [])

    def test_chat_counts_tool_loop_as_one_turn_and_uses_persisted_memory(self):
        questions = [
            "我以后优先看顶会论文，这一次只看 ReAct 的 arXiv v3",
            "你好",
            "我有哪些偏好？",
            "顶会优先这个偏好已经失效了。",
        ]
        with TemporaryDirectory() as clean_project:
            project_dir = Path(clean_project)
            (project_dir / ".env").write_text(
                "MODEL=deepseek-flash\nAPI_KEY=test-key\nMEMORY_SHORT_TERM_TURNS=1\n",
                encoding="utf-8",
            )
            saved = self.tool_message("save_memory", [
                {"content": "优先看顶会论文", "importance": 0.9},
                {"content": "这次只看 ReAct 的 arXiv v3", "importance": 0.5},
            ])
            searched = self.tool_message("search_memory", [{}])
            forgotten = self.tool_message("forget_memory", [{"memory_id": 1}])
            responses = [
                saved,
                {"role": "assistant", "content": "已记录。"},
                {"role": "assistant", "content": "你好。"},
                searched,
                {"role": "assistant", "content": "你优先看顶会论文。"},
                forgotten,
                {"role": "assistant", "content": "已淘汰这条偏好。"},
            ]
            with (
                patch.object(chat, "__file__", str(project_dir / "chat.py")),
                patch.object(chat, "call_model", side_effect=responses) as model,
                patch("builtins.input", side_effect=[questions[0], "/exit", *questions[1:], "/exit"]),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()) as errors,
            ):
                self.assertEqual(chat.main(), 0)
                self.assertEqual(chat.main(), 0)
                self.assertEqual(errors.getvalue(), "")

            first_tool_results = model.call_args_list[1].args[3][-2:]
            records = [json.loads(message["content"])["memory"] for message in first_tool_results]
            self.assertEqual([record["kind"] for record in records], ["long_term", "short_term"])
            self.assertEqual([record["written_turn"] for record in records], [1, 1])
            follow_up = model.call_args_list[4].args[3]
            self.assertEqual(json.loads(follow_up[-1]["content"])["memories"], records[:1])
            self.assertNotIn(questions[0], [message["content"] for message in follow_up])
            forget_request = model.call_args_list[5].args[3]
            self.assertNotIn(questions[1], [message["content"] for message in forget_request])
            forget_result = model.call_args_list[6].args[3][-1]["content"]
            self.assertEqual(json.loads(forget_result), {"forgotten": True, "memory_id": 1})

            with closing(MemoryStore(project_dir / ".agent-data" / "memory.sqlite3")) as memory:
                self.assertEqual(memory.search(""), [])

    @staticmethod
    def tool_message(name, arguments):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "id": f"{name}-{index}",
                    "function": {"name": name, "arguments": json.dumps(values)},
                }
                for index, values in enumerate(arguments)
            ],
        }


if __name__ == "__main__":
    unittest.main()
