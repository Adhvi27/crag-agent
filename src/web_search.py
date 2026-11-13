from __future__ import annotations

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.documents import Document


def web_search(query: str, max_results: int = 5) -> list[Document]:
    tool = DuckDuckGoSearchResults(max_results=max_results, output_format="list")
    raw = tool.invoke(query)
    if not raw:
        return []
    docs: list[Document] = []
    for i, item in enumerate(raw):
        if isinstance(item, dict):
            content = item.get("snippet") or item.get("body") or str(item)
            title = item.get("title", "")
            link = item.get("link") or item.get("href") or ""
            text = f"{title}\n{content}".strip()
            meta = {"source": "web", "url": link, "rank": i}
        else:
            text = str(item)
            meta = {"source": "web", "rank": i}
        if text.strip():
            docs.append(Document(page_content=text, metadata=meta))
    return docs
