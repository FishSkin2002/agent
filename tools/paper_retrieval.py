import re
import sys
from itertools import zip_longest
from typing import Any

from arxiv import ArxivError

from .base import Tool
from .paper_search import MAX_RESULTS, search_papers


MAX_QUERIES = 3


def retrieve_papers(queries: list[str]) -> dict[str, Any]:
    """执行模型给出的查询式，合并相关论文并保留部分失败信息。"""
    queries = list(dict.fromkeys(query.strip() for query in queries if query.strip()))
    if not queries or len(queries) > MAX_QUERIES:
        raise ValueError(f"请提供 1 至 {MAX_QUERIES} 条检索式")

    result_lists = []
    failed_queries = []
    for query in queries:
        try:
            result_lists.append(search_papers(query))
        except (ArxivError, OSError):
            failed_queries.append(query)

    if not result_lists:
        return {"error": "arXiv 检索失败，可以稍后重试", "failed_queries": failed_queries}

    papers_by_id = {}
    # 按排名轮流取各查询的结果，让每条查询的高排名论文都有机会进入结果。
    for same_rank in zip_longest(*result_lists):
        for paper in same_rank:
            if paper is None:
                continue
            paper_id = re.sub(r"v[0-9]+$", "", str(paper["paper_id"]))
            papers_by_id.setdefault(paper_id, paper)

    return {"papers": list(papers_by_id.values())[:MAX_RESULTS], "failed_queries": failed_queries}


class SearchPapers(Tool):
    name = "search_papers"
    description = (
        "在线查找 arXiv 论文，返回标题、作者、摘要和版本链接。"
        "题名或术语可能拼错或有歧义时，提供一到两条补充查询以覆盖不同写法。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": '将问题转换为英文关键词或 arXiv 查询式，例如 ti:"Synergizing Reasoning and Acting"。',
            },
            "alternatives": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": MAX_QUERIES - 1,
                "description": "可独立检索的补充查询，可用纠正后的拼写、近义词、缩写或补全的题名。",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if arguments.keys() - {"query", "alternatives"}:
            return {"error": "请提供 query 和可选的 alternatives 参数"}
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return {"error": "query 必须是非空字符串"}
        alternatives = arguments.get("alternatives", [])
        if not isinstance(alternatives, list) or len(alternatives) >= MAX_QUERIES:
            return {"error": f"alternatives 必须是最多包含 {MAX_QUERIES - 1} 项的数组"}
        if any(not isinstance(item, str) or not item.strip() for item in alternatives):
            return {"error": "补充查询必须是非空字符串"}

        queries = [query, *alternatives]
        print("检索 arXiv > " + " | ".join(queries), file=sys.stderr)
        try:
            return retrieve_papers(queries)
        except ValueError as error:
            return {"error": str(error)}
