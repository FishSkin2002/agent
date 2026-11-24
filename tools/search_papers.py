import re
import sys
from typing import Any

import arxiv

from .base import Tool
from .paper_library import Paper, PaperLibrary


class SearchPapersTool(Tool):
    _MAX_RESULTS = 5
    _FIELD_LABELS = {
        "ti": "论文标题",
        "au": "作者",
        "abs": "摘要",
        "co": "备注",
        "jr": "期刊引用",
        "cat": "学科分类",
        "rn": "报告编号",
        "id": "论文编号",
        "all": "全部字段",
        "submittedDate": "提交日期",
    }
    # Match quoted phrases as a unit before looking for field prefixes.
    _QUERY_TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"?|(^|[\s(])([A-Za-z]+):', re.DOTALL)

    name = "search_papers"
    description = (
        "用一条查询检索 arXiv，返回论文标题、作者、摘要和版本链接。"
        "检索结果会保存到论文知识库。"
        "如果结果为空或证据不足，可以根据返回结果改写查询，再次调用本工具。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": '将问题转换为英文关键词或 arXiv 查询式，例如 ti:"Synergizing Reasoning and Acting"。',
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, library: PaperLibrary) -> None:
        self._library = library
        self._client = arxiv.Client(page_size=self._MAX_RESULTS)

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if arguments.keys() != {"query"}:
            return {"error": "请仅提供 query 参数"}
        query = arguments["query"]
        if not isinstance(query, str) or not query.strip():
            return {"error": "query 必须是非空字符串"}

        query = query.strip()
        print(f"检索 arXiv > {self._display_query(query)}", file=sys.stderr, flush=True)
        try:
            papers = self._search_arxiv(query)
        except (arxiv.ArxivError, OSError):
            return {"error": "arXiv 检索失败，可以稍后重试"}
        except ValueError as error:
            return {"error": str(error)}
        self._library.save_papers(papers)
        return {"papers": papers}

    @classmethod
    def _display_query(cls, query: str) -> str:
        def display_token(match: re.Match[str]) -> str:
            boundary, field = match.groups()
            if field in cls._FIELD_LABELS:
                return f"{boundary}{cls._FIELD_LABELS[field]}："
            return match.group(0)

        return cls._QUERY_TOKEN.sub(display_token, query)

    def _search_arxiv(self, query: str) -> list[Paper]:
        search = arxiv.Search(query=query, max_results=self._MAX_RESULTS)
        papers = []
        for result in self._client.results(search):
            paper_id = result.get_short_id()
            papers.append({
                "paper_id": paper_id,
                "title": " ".join(result.title.split()),
                "authors": [author.name for author in result.authors],
                "submitted_at": result.published.date().isoformat(),
                "abstract": " ".join(result.summary.split()),
                "url": f"https://arxiv.org/abs/{paper_id}",
                "journal_ref": result.journal_ref,
            })
        return papers
