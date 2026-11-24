import io
import json
import unittest
from contextlib import closing, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from chat import create_tool_registry
from memory import MemoryStore
from tools.paper_library import PaperLibrary
from tools.search_papers import SearchPapersTool


class PaperRAGTests(unittest.TestCase):
    def test_arxiv_log_translates_fields_and_preserves_the_api_query(self):
        query = (
            'ti:"On-Policy Self-Distillation with abs: in the title" '
            'AND (au:Chen OR abs:distillation) AND co:revised AND jr:ICLR AND cat:cs.CL '
            'AND rn:123 AND id:2210.03629v3 AND all:OPSD '
            'AND submittedDate:[202301010000 TO 202612312359]'
        )
        with TemporaryDirectory() as directory, redirect_stderr(io.StringIO()) as log:
            with (
                closing(PaperLibrary(Path(directory) / "papers.sqlite3")) as library,
                patch("tools.search_papers.arxiv.Client.results", return_value=[]) as results,
            ):
                result = SearchPapersTool(library).run({"query": query})

        self.assertEqual(result, {"papers": []})
        self.assertEqual(results.call_args.args[0].query, query)
        self.assertIn(
            '论文标题："On-Policy Self-Distillation with abs: in the title" '
            'AND (作者：Chen OR 摘要：distillation) AND 备注：revised AND 期刊引用：ICLR AND 学科分类：cs.CL '
            'AND 报告编号：123 AND 论文编号：2210.03629v3 AND 全部字段：OPSD '
            'AND 提交日期：[202301010000 TO 202612312359]',
            log.getvalue(),
        )

    def test_registry_persists_catalog_and_page_evidence(self):
        paper_id = "2210.03629v3"
        pdf_text = [
            "Introduction to language models.",
            "For HotpotQA and Fever we chose six and three few-shot exemplars, respectively.",
        ]
        article = SimpleNamespace(
            get_short_id=lambda: paper_id,
            title="ReAct: Synergizing Reasoning and Acting in Language Models",
            authors=[SimpleNamespace(name="Shunyu Yao")],
            published=datetime(2022, 10, 6, tzinfo=timezone.utc),
            summary="A language model interleaves reasoning and acting for question answering.",
            journal_ref=None,
        )
        pdf = SimpleNamespace(
            pages=[SimpleNamespace(extract_text=lambda text=text, **kwargs: text) for text in pdf_text]
        )
        paraphrase = "How many demonstrations did the two benchmarks use?"
        with TemporaryDirectory() as directory, redirect_stderr(io.StringIO()):
            project = Path(directory)
            library_path = project / "papers.sqlite3"
            with (
                closing(MemoryStore(project / "memory.sqlite3")) as memory,
                patch("tools.search_papers.arxiv.Client.results", side_effect=[[article], [article]]),
                patch.object(PaperLibrary, "_embed", side_effect=self.vectors) as embed,
                patch("tools.search_paper_passages.urlopen", return_value=io.BytesIO(b"pdf")) as download,
                patch("tools.search_paper_passages.PdfReader", return_value=pdf),
            ):
                with closing(PaperLibrary(library_path)) as library:
                    registry = create_tool_registry(memory, library)
                    for _ in range(2):
                        self.invoke(registry, "search_papers", query="ReAct")
                    embed.assert_not_called()
                    result = self.invoke(
                        registry, "search_paper_passages", query=paraphrase, paper_id=paper_id
                    )
                    evidence = result["passages"][0]
                    self.assertEqual(evidence["text"], pdf_text[1])
                    self.assertEqual(evidence["page"], 2)
                    self.assertEqual(evidence["url"], f"https://arxiv.org/pdf/{paper_id}#page=2")

                with closing(PaperLibrary(library_path)) as library:
                    registry = create_tool_registry(memory, library)
                    restored = self.invoke(registry, "search_paper_library", query="reasoning and acting")
                    self.assertEqual([paper["paper_id"] for paper in restored["papers"]], [paper_id])
                    self.assertTrue(restored["papers"][0]["has_full_text"])
                    for arguments in ({}, {"paper_id": paper_id}):
                        result = self.invoke(registry, "search_paper_passages", query=paraphrase, **arguments)
                        self.assertEqual(result["passages"][0], evidence)
                download.assert_called_once_with(f"https://arxiv.org/pdf/{paper_id}", timeout=30)

    def test_passage_search_can_rank_across_papers_or_filter_one_version(self):
        first_id = "2210.03629v2"
        second_id = "2210.03629v3"
        with TemporaryDirectory() as directory, patch.object(PaperLibrary, "_embed", side_effect=self.vectors):
            with closing(PaperLibrary(Path(directory) / "papers.sqlite3")) as library:
                library.save_passages(first_id, [(1, "Introduction to language models")])
                library.save_passages(second_id, [(4, "6 and 3 few-shot exemplars")])
                hits = library.search_passages("How many demonstrations?")
                self.assertEqual([hit["paper_id"] for hit in hits], [second_id, first_id])
                restricted = library.search_passages("How many demonstrations?", first_id)
                self.assertEqual([hit["paper_id"] for hit in restricted], [first_id])

    def test_invalid_id_does_not_trigger_a_download(self):
        with TemporaryDirectory() as directory:
            project = Path(directory)
            with (
                closing(MemoryStore(project / "memory.sqlite3")) as memory,
                closing(PaperLibrary(project / "papers.sqlite3")) as library,
                patch("tools.search_paper_passages.urlopen") as download,
            ):
                registry = create_tool_registry(memory, library)
                result = self.invoke(
                    registry, "search_paper_passages", query="ReAct", paper_id="2210.03629"
                )
                self.assertIn("error", result)
                download.assert_not_called()

    @staticmethod
    def vectors(texts):
        vectors = np.zeros((len(texts), 384), dtype="<f4")
        for index, text in enumerate(texts):
            dimension = int("Introduction" in text)
            vectors[index, dimension] = 1
        return vectors

    @staticmethod
    def invoke(registry, name, **arguments):
        return registry.invoke({
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        })
