from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.documents import Document
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class GradeDocuments(BaseModel):
    """Structured output for document relevance grading."""

    binary_score: Literal["yes", "no"] = Field(
        description="Whether the document helps answer the question."
    )
    reasoning: str = Field(description="Brief justification.")


class GradeRetrieval(BaseModel):
    """Overall retrieval quality for the current query."""

    score: Literal["correct", "incorrect", "ambiguous"]
    reasoning: str


class GraphState(TypedDict):
    question: str
    search_query: str
    documents: list[Document]
    web_documents: list[Document]
    retrieval_grade: str
    generation: str
    messages: Annotated[list, add_messages]
    loop_count: int
    max_loops: int
    used_web: bool
