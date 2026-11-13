from __future__ import annotations

from pathlib import Path
from typing import Literal

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from src.graders import grade_retrieval_batch
from src.retrieval import build_vector_store, retrieve
from src.state import GraphState
from src.web_search import web_search


def _format_docs(docs: list[Document]) -> str:
    if not docs:
        return "(no context)"
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        url = d.metadata.get("url", "")
        header = f"[{i}] source={src}" + (f" url={url}" if url else "")
        parts.append(f"{header}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


class CRAGAgent:
    """Corrective RAG loop: grade retrieval, rewrite query or search web, then generate."""

    def __init__(
        self,
        knowledge_path: Path,
        *,
        model: str | None = None,
        persist_dir: Path | None = None,
        max_loops: int = 2,
    ) -> None:
        self.model = model or "gpt-4o-mini"
        self.max_loops = max_loops
        self.store: Chroma = build_vector_store(knowledge_path, persist_dir)
        self.llm = ChatOpenAI(model=self.model, temperature=0)
        self.app = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(GraphState)

        graph.add_node("retrieve", self._retrieve)
        graph.add_node("grade_documents", self._grade_documents)
        graph.add_node("rewrite_query", self._rewrite_query)
        graph.add_node("web_search", self._web_search_node)
        graph.add_node("generate", self._generate)

        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "grade_documents")
        graph.add_conditional_edges(
            "grade_documents",
            self._route_after_grade,
            {
                "generate": "generate",
                "rewrite_query": "rewrite_query",
                "web_search": "web_search",
            },
        )
        graph.add_edge("rewrite_query", "retrieve")
        graph.add_edge("web_search", "generate")
        graph.add_edge("generate", END)

        return graph.compile()

    def invoke(self, question: str) -> GraphState:
        return self.app.invoke(
            {
                "question": question,
                "search_query": question,
                "documents": [],
                "web_documents": [],
                "retrieval_grade": "",
                "generation": "",
                "messages": [],
                "loop_count": 0,
                "max_loops": self.max_loops,
                "used_web": False,
            }
        )

    def _retrieve(self, state: GraphState) -> dict:
        query = state["search_query"]
        docs = retrieve(self.store, query, k=4)
        return {"documents": docs}

    def _grade_documents(self, state: GraphState) -> dict:
        overall, relevant = grade_retrieval_batch(
            state["question"], state["documents"], model=self.model
        )
        return {
            "retrieval_grade": overall.score,
            "documents": relevant if relevant else state["documents"],
            "messages": [
                HumanMessage(
                    content=f"Retrieval grade: {overall.score} — {overall.reasoning}"
                )
            ],
        }

    def _route_after_grade(
        self, state: GraphState
    ) -> Literal["generate", "rewrite_query", "web_search"]:
        grade = state["retrieval_grade"]
        loops = state["loop_count"]
        max_loops = state["max_loops"]

        if grade == "correct":
            return "generate"

        if grade == "ambiguous" and loops < max_loops:
            return "rewrite_query"

        if grade == "incorrect" and loops < max_loops and not state["used_web"]:
            return "rewrite_query"

        if not state["used_web"]:
            return "web_search"

        return "generate"

    def _rewrite_query(self, state: GraphState) -> dict:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Rewrite the search query to improve retrieval for the user's question. "
                    "Use clearer keywords, synonyms, or a decomposed sub-question. "
                    "Output only the new query, no explanation.",
                ),
                (
                    "human",
                    "User question: {question}\nPrevious search query: {search_query}",
                ),
            ]
        )
        chain = prompt | self.llm
        new_query = chain.invoke(
            {
                "question": state["question"],
                "search_query": state["search_query"],
            }
        ).content.strip()
        return {
            "search_query": new_query,
            "loop_count": state["loop_count"] + 1,
            "messages": [HumanMessage(content=f"Reformulated query: {new_query}")],
        }

    def _web_search_node(self, state: GraphState) -> dict:
        query = state["search_query"]
        docs = web_search(query, max_results=5)
        return {
            "web_documents": docs,
            "used_web": True,
            "messages": [
                HumanMessage(content=f"Web fallback: fetched {len(docs)} result(s).")
            ],
        }

    def _generate(self, state: GraphState) -> dict:
        kb_docs = state["documents"] if state["retrieval_grade"] == "correct" else []
        if not kb_docs and state["documents"]:
            kb_docs = state["documents"]
        web_docs = state["web_documents"]
        context_docs = kb_docs + web_docs
        context = _format_docs(context_docs)

        messages = [
            SystemMessage(
                content=(
                    "Answer the user question using only the provided context. "
                    "If context is insufficient, say what is missing. "
                    "Cite sources by bracket number when possible."
                )
            ),
            HumanMessage(
                content=f"Question: {state['question']}\n\nContext:\n{context}"
            ),
        ]
        answer = self.llm.invoke(messages).content
        return {"generation": answer, "messages": [HumanMessage(content=answer)]}
