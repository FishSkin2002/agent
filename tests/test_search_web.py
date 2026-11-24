import io
import json
import unittest
from contextlib import redirect_stderr
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

import main
from tools.search_web import SearchWebTool


class SearchWebTests(unittest.TestCase):
    def test_web_search_results_reach_the_agent_and_the_reviewer(self):
        query = "OPSD 首发论文"
        api_key = "test-serpapi-key"
        registry = main.create_tool_registry(Mock(), Mock(), serpapi_api_key=api_key)
        self.assertNotIn(
            SearchWebTool.name,
            [tool["function"]["name"] for tool in main.create_tool_registry(Mock(), Mock()).definitions()],
        )
        call = {"role": "assistant", "content": None, "tool_calls": [{
            "id": "web-1", "type": "function", "function": {
                "name": SearchWebTool.name, "arguments": json.dumps({"query": query}),
            },
        }]}
        hits = [{"title": "无有效链接", "link": "javascript:void(0)"}] + [
            {"title": f"网页 {i}", "link": f"https://example.org/{i}", "snippet": f"搜索摘要 {i}"}
            for i in range(7)
        ]
        provider_body = json.dumps({"organic_results": hits}).encode()
        output = {"role": "assistant", "content": "已找到相关网页。"}
        review = {"role": "assistant", "content": '{"answer":"", "missing_evidence":""}'}

        with (
            patch("tools.search_web.urlopen", return_value=io.BytesIO(provider_body)) as request,
            patch.object(main, "call_model", side_effect=[call, output, review]) as model,
            redirect_stderr(io.StringIO()) as progress,
        ):
            turn = main.run_turn("unused", "deepseek-flash", "test-key", [], query, registry)

        url = urlsplit(request.call_args.args[0])
        self.assertEqual((url.scheme, url.netloc, url.path), ("https", "serpapi.com", "/search.json"))
        self.assertEqual(parse_qs(url.query), {"engine": ["google"], "q": [query], "api_key": [api_key]})
        observation = json.loads(model.call_args_list[1].args[3][-1]["content"])
        self.assertEqual(observation, {"results": [
            {"title": hit["title"], "url": hit["link"], "snippet": hit["snippet"]} for hit in hits[1:6]
        ]})
        review_request = json.loads(model.call_args_list[2].args[3][-1]["content"])
        self.assertEqual(review_request["retrieval_results"], [{"tool": SearchWebTool.name, "result": observation}])
        self.assertNotIn(api_key, json.dumps(model.call_args_list[2].args[3]))
        self.assertIn("search_web 返回 5 条网页结果", progress.getvalue())
        self.assertEqual(turn[-1], output)

    def test_http_error_does_not_return_the_authenticated_request_url(self):
        error = HTTPError("https://serpapi.com/search.json?api_key=secret", 429, "Too Many Requests", {}, None)
        with patch("tools.search_web.urlopen", side_effect=error):
            self.assertEqual(SearchWebTool("secret").run({"query": "ReAct"}), {"error": "SerpAPI 返回 HTTP 429"})
