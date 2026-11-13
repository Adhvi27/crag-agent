from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.state import GradeDocuments, GradeRetrieval


def _llm(model: str | None = None) -> ChatOpenAI:
    return ChatOpenAI(model=model or "gpt-4o-mini", temperature=0)


def grade_document_relevance(
    question: str,
    document: Document,
    *,
    model: str | None = None,
) -> GradeDocuments:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You grade whether a retrieved document is relevant to answering the user question. "
                "Answer with binary_score 'yes' or 'no' and brief reasoning.",
            ),
            (
                "human",
                "Question: {question}\n\nDocument:\n{document}",
            ),
        ]
    )
    structured = _llm(model).with_structured_output(GradeDocuments)
    chain = prompt | structured
    return chain.invoke(
        {"question": question, "document": document.page_content[:4000]}
    )


def grade_retrieval_batch(
    question: str,
    documents: list[Document],
    *,
    model: str | None = None,
) -> tuple[GradeRetrieval, list[Document]]:
    if not documents:
        return (
            GradeRetrieval(
                score="incorrect",
                reasoning="No documents were retrieved.",
            ),
            [],
        )

    relevant: list[Document] = []
    for doc in documents:
        grade = grade_document_relevance(question, doc, model=model)
        if grade.binary_score == "yes":
            relevant.append(doc)

    n = len(documents)
    r = len(relevant)
    if r == 0:
        overall = GradeRetrieval(
            score="incorrect",
            reasoning=f"0/{n} retrieved documents were relevant.",
        )
    elif r == n:
        overall = GradeRetrieval(
            score="correct",
            reasoning=f"All {n} retrieved documents appear relevant.",
        )
    else:
        overall = GradeRetrieval(
            score="ambiguous",
            reasoning=f"{r}/{n} documents relevant; partial retrieval quality.",
        )
    return overall, relevant
