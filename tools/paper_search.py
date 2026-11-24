import arxiv


MAX_RESULTS = 5
client = arxiv.Client(page_size=MAX_RESULTS)


def search_papers(query: str) -> list[dict[str, str | list[str] | None]]:
    """从 arXiv 返回与查询相关的论文及其可核验的摘要链接。"""
    query = query.strip()
    if not query:
        raise ValueError("请提供论文检索关键词或 arXiv 查询式。")

    search = arxiv.Search(query=query, max_results=MAX_RESULTS)
    papers = []
    for result in client.results(search):
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
