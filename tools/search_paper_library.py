from typing import Any

from .base import Tool
from .paper_library import EmbeddingError, PaperLibrary


class SearchPaperLibraryTool(Tool):
    name = "search_paper_library"
    description = (
        "在以前通过 arXiv 搜到并保存的论文中，按标题和摘要做语义检索，"
        "最多返回 5 篇论文及其带版本的 arXiv ID。可用于找回跨会话的论文。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "用英文描述要找的研究主题、方法或论文标题。"},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, library: PaperLibrary) -> None:
        self._library = library

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = arguments.get("query")
        if arguments.keys() != {"query"} or not isinstance(query, str) or not query.strip():
            return {"error": "请提供一个非空字符串 query"}
        try:
            return {"papers": self._library.search_papers(query.strip())}
        except EmbeddingError as error:
            return {"error": str(error)}
