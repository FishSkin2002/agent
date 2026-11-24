import io
import json
import unittest
from contextlib import redirect_stderr
from unittest.mock import Mock, patch

import chat
from tools.create_plan import CreatePlanTool
from tools.registry import ToolRegistry
from tools.search_paper_passages import SearchPaperPassagesTool


QUESTION = "ReAct 在 HotpotQA 和 FEVER 上分别用了几个 few-shot 示例？请引用原文页码。"
PAPER_ID = "2210.03629v3"
PAGE_URL = f"https://arxiv.org/pdf/{PAPER_ID}#page=4"
PASSAGE_TOOL = SearchPaperPassagesTool.name
HOT_POT_QA = {"paper_id": PAPER_ID, "page": 4, "text": "HotpotQA: 6 examples.", "url": PAGE_URL}
FEVER = {"paper_id": PAPER_ID, "page": 4, "text": "FEVER: 3 examples.", "url": PAGE_URL}


def tool_message(name, call_id, **arguments):
    return {"role": "assistant", "content": None, "tool_calls": [{
        "id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)},
    }]}


def text_message(content):
    return {"role": "assistant", "content": content}


def review_message(answer="", missing_evidence=""):
    return text_message(json.dumps({"answer": answer, "missing_evidence": missing_evidence}))


class PaperWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.progress = self.enterContext(redirect_stderr(io.StringIO()))

    def test_plan_advances_steps_before_review_and_supplements_with_existing_evidence(self):
        steps = ["核对 HotpotQA 的示例数", "核对 FEVER 的示例数"]
        draft = f"HotpotQA 有 6 个示例（{PAGE_URL}），FEVER 待核实。"
        final = f"HotpotQA 用了 6 个，FEVER 用了 3 个。见 ReAct v3 第 4 页（{PAGE_URL}）。"
        responses = [
            tool_message(CreatePlanTool.name, "plan", steps=steps),
            tool_message(PASSAGE_TOOL, "hotpotqa", query="HotpotQA examples", paper_id=PAPER_ID),
            text_message("HotpotQA 步骤记录"),
            text_message("FEVER 步骤记录"),
            text_message(draft),
            review_message("HotpotQA 用了 6 个。", "查找 FEVER 示例数量"),
            tool_message(PASSAGE_TOOL, "fever", query="FEVER examples", paper_id=PAPER_ID),
            text_message(final),
            review_message(),
        ]
        seen = []

        def model_response(url, model, api_key, messages, registry, **kwargs):
            names = [definition["function"]["name"] for definition in registry.definitions()]
            index = len(seen)
            seen.append((messages, names, kwargs))
            return responses[index]

        library = Mock()
        library.has_passages.return_value = True
        library.search_passages.side_effect = [[HOT_POT_QA], [FEVER]]
        passage_tool = SearchPaperPassagesTool(library)
        self.registry.register(passage_tool)
        with patch.object(chat, "call_model", side_effect=model_response):
            turn = chat.run_turn("unused", "deepseek-flash", "test-key", [], QUESTION, self.registry)

        self.assertEqual(len(seen), len(responses))
        self.assertIn(CreatePlanTool.name, seen[0][1])
        self.assertEqual(seen[6][1], [PASSAGE_TOOL])
        self.assertEqual(seen[5][1], [])
        self.assertIn(steps[0], seen[1][0][-1]["content"])
        self.assertIn(steps[1], seen[3][0][-1]["content"])
        self.assertIn("HotpotQA 步骤记录", seen[4][0][-1]["content"])
        self.assertIn("查找 FEVER 示例数量", seen[6][0][-1]["content"])
        self.assertEqual(json.loads(seen[6][0][-2]["content"]), {"passages": [HOT_POT_QA]})
        self.assertEqual(json.loads(seen[8][0][-1]["content"])["retrieval_results"], [
            {"tool": PASSAGE_TOOL, "result": {"passages": [HOT_POT_QA]}},
            {"tool": PASSAGE_TOOL, "result": {"passages": [FEVER]}},
        ])
        self.assertEqual([message["tool_call_id"] for message in turn if message["role"] == "tool"], [
            "plan", "hotpotqa", "fever",
        ])
        self.assertEqual([message for message in turn if message["role"] == "assistant" and message["content"]], [
            text_message(final),
        ])
        self.assertEqual(self.registry.definitions(), [passage_tool.definition()])
        self.assertIn("[计划] 第 2/2 步", self.progress.getvalue())
        self.assertIn("[复核] 正在补查", self.progress.getvalue())

    def test_budget_counts_plan_alongside_paper_tools_and_caps_a_batch(self):
        batch = tool_message(CreatePlanTool.name, "plan", steps=["核对 HotpotQA", "核对 FEVER"])
        for call_id in ["hotpotqa", "fever"]:
            batch["tool_calls"] += tool_message(PASSAGE_TOOL, call_id, query=call_id)["tool_calls"]
        responses = [
            batch, text_message("步骤一"), text_message("步骤二"), text_message("草稿"),
            review_message("HotpotQA 用了 6 个。", "缺少 FEVER 的原文。"),
        ]
        passages = Mock(return_value=[HOT_POT_QA])
        library = Mock(search_passages=passages)
        self.registry.register(SearchPaperPassagesTool(library))
        with (
            patch.object(chat, "MAX_TOOL_CALLS", 2),
            patch.object(chat, "call_model", side_effect=responses) as model,
        ):
            turn = chat.run_turn("unused", "glm-4.7-flash", "test-key", [], QUESTION, self.registry)
        self.assertEqual(model.call_count, 5)
        self.assertEqual(model.call_args_list[1].args[-1].definitions(), [])
        passages.assert_called_once_with("hotpotqa", None)
        skipped = next(message for message in turn if message.get("tool_call_id") == "fever")
        self.assertIn("未执行此调用", skipped["content"])
        self.assertIn("证据缺口：缺少 FEVER 的原文。", turn[-1]["content"])

    def test_repeated_passage_with_different_similarity_stops_supplementing(self):
        responses = [
            tool_message(PASSAGE_TOOL, "first", query="HotpotQA and FEVER examples"),
            text_message("草稿"),
            review_message("HotpotQA 用了 6 个。", "缺少 FEVER 的原文。"),
            tool_message(PASSAGE_TOOL, "retry", query="FEVER demonstrations"),
            text_message("补查后的草稿"),
            review_message("HotpotQA 用了 6 个。", "缺少 FEVER 的原文。"),
        ]
        library = Mock()
        library.search_passages.side_effect = [[{**HOT_POT_QA, "similarity": score}] for score in [0.9, 0.4]]
        self.registry.register(SearchPaperPassagesTool(library))
        with patch.object(chat, "call_model", side_effect=responses) as model:
            turn = chat.run_turn("unused", "deepseek-flash", "test-key", [], QUESTION, self.registry)
        self.assertEqual(model.call_count, 6)
        self.assertIn("补查未取得新证据", self.progress.getvalue())
        self.assertIn("证据缺口：缺少 FEVER 的原文。", turn[-1]["content"])

    def test_invalid_review_keeps_a_visible_warning(self):
        responses = [
            tool_message(PASSAGE_TOOL, "first", query="HotpotQA and FEVER examples"),
            text_message("HotpotQA 用了 6 个。"),
            text_message("未按格式提供复核结果。"),
        ]
        self.registry.register(SearchPaperPassagesTool(Mock(search_passages=Mock(return_value=[HOT_POT_QA]))))
        with patch.object(chat, "call_model", side_effect=responses):
            turn = chat.run_turn("unused", "glm-4.7-flash", "test-key", [], QUESTION, self.registry)
        self.assertIn("本轮未完成证据复核", turn[-1]["content"])
