import re
import sys
from http.client import HTTPException
from shutil import copyfileobj
from tempfile import TemporaryFile
from typing import Any
from urllib.request import urlopen

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from .base import Tool
from .paper_library import EmbeddingError, PaperLibrary


class SearchPaperPassagesTool(Tool):
    _VERSIONED_ARXIV_ID = re.compile(r"(?:[0-9]{4}\.[0-9]{4,5}|[a-z][a-z0-9.-]*/[0-9]{7})v[1-9][0-9]*")
    _CHUNK_WORDS = 100
    _OVERLAP_WORDS = 30

    name = "search_paper_passages"
    description = (
        "语义检索论文正文，返回最多 3 段原文以及页码、版本链接。"
        "指定带版本的 arXiv ID 时可在线下载并收录该论文。"
        "省略 ID 时在所有已收录的全文片段中检索。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "用英文描述要找的具体论据，保留论文中的方法、数据集或指标名称。",
            },
            "paper_id": {"type": "string", "description": "可选的 arXiv 版本 ID，例如 2210.03629v3。"},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, library: PaperLibrary) -> None:
        self._library = library

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = arguments.get("query")
        paper_id = arguments.get("paper_id")
        if arguments.keys() - {"query", "paper_id"} or not isinstance(query, str) or not query.strip():
            return {"error": "请提供非空字符串 query 和可选的 paper_id"}
        if "paper_id" in arguments and (
            not isinstance(paper_id, str)
            or len(paper_id) > 80
            or not self._VERSIONED_ARXIV_ID.fullmatch(paper_id)
        ):
            return {"error": "paper_id 应是带版本的 arXiv ID，例如 2210.03629v3"}

        try:
            if paper_id is not None and not self._library.has_passages(paper_id):
                self._library.save_passages(paper_id, self._load_passages(paper_id))
            return {"passages": self._library.search_passages(query.strip(), paper_id)}
        except EmbeddingError as error:
            return {"error": str(error)}
        except (OSError, HTTPException, PyPdfError, ValueError) as error:
            return {"error": f"论文正文导入失败（{error}）"}

    def _load_passages(self, paper_id: str) -> list[tuple[int, str]]:
        print(f"读取 arXiv PDF > {paper_id}", file=sys.stderr)
        passages = []
        stride = self._CHUNK_WORDS - self._OVERLAP_WORDS
        with TemporaryFile() as pdf:
            with urlopen(f"https://arxiv.org/pdf/{paper_id}", timeout=30) as response:
                copyfileobj(response, pdf)
            pdf.seek(0)
            reader = PdfReader(pdf)
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text(extraction_mode="layout") or page.extract_text() or ""
                for paragraph in re.split(r"\n\s*\n", text):
                    words = paragraph.split()
                    for start in range(0, len(words), stride):
                        end = start + self._CHUNK_WORDS
                        passages.append((page_number, " ".join(words[start:end])))
                        if end >= len(words):
                            break
        if not passages:
            raise ValueError("PDF 中没有可提取的文本")
        return passages
