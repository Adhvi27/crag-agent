from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_vector_store(
    knowledge_path: Path,
    persist_dir: Path | None = None,
) -> Chroma:
    text = knowledge_path.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=c, metadata={"source": knowledge_path.name}) for c in chunks]
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    kwargs: dict = {"collection_name": "crag_kb", "embedding_function": embeddings}
    if persist_dir:
        persist_dir.mkdir(parents=True, exist_ok=True)
        kwargs["persist_directory"] = str(persist_dir)
    return Chroma.from_documents(documents=docs, **kwargs)


def retrieve(store: Chroma, query: str, k: int = 4) -> list[Document]:
    return store.similarity_search(query, k=k)
