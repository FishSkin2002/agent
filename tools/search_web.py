import json
from http.client import HTTPException
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

from .base import Tool


class SearchWebTool(Tool):
    name = "search_web"
    description = (
        "用 SerpAPI 搜索 Google 网页，返回标题、链接和检索摘要。"
        "用于补充知识，或在论文检索无法确定标题、作者或发布地址时寻找线索。"
        "论文的方法和实验细节应到论文原文核对。"
    )
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "用于 Google 搜索的关键词或问题。"}},
        "required": ["query"],
        "additionalProperties": False,
    }
    _MAX_RESULTS = 5

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = arguments.get("query")
        if arguments.keys() != {"query"} or not isinstance(query, str) or not query.strip():
            return {"error": "请提供一个非空字符串 query"}

        params = urlencode({"engine": "google", "q": query.strip(), "api_key": self._api_key})
        try:
            with urlopen(f"https://serpapi.com/search.json?{params}", timeout=30) as response:
                data = json.load(response)
        except HTTPError as error:
            error.close()
            return {"error": f"SerpAPI 返回 HTTP {error.code}"}
        except (OSError, HTTPException, ValueError):
            return {"error": "SerpAPI 网页检索失败"}

        if not isinstance(data, dict) or data.get("error"):
            return {"error": "SerpAPI 未返回有效的网页检索结果"}
        hits = data.get("organic_results", [])
        if not isinstance(hits, list):
            return {"error": "SerpAPI 返回的网页结果格式有误"}

        results = []
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            title, url, snippet = hit.get("title"), hit.get("link"), hit.get("snippet")
            if not isinstance(title, str) or not isinstance(url, str) or not url.startswith(("https://", "http://")):
                continue
            results.append({"title": title, "url": url, "snippet": snippet if isinstance(snippet, str) else ""})
            if len(results) == self._MAX_RESULTS:
                break
        return {"results": results}
