#!/usr/bin/env python3
"""CLI for Corrective RAG (CRAG) agent."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.graph import CRAGAgent

ROOT = Path(__file__).resolve().parent
DEFAULT_KB = ROOT / "data" / "knowledge_base.txt"


def main() -> int:
    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY in .env or environment.", file=sys.stderr)
        return 1

    parser = argparse.ArgumentParser(description="Corrective RAG agent (LangGraph)")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument(
        "--kb",
        type=Path,
        default=DEFAULT_KB,
        help="Path to knowledge base text file",
    )
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--max-loops", type=int, default=2)
    args = parser.parse_args()

    question = args.question
    if not question:
        question = input("Question: ").strip()
    if not question:
        print("No question provided.", file=sys.stderr)
        return 1

    agent = CRAGAgent(
        args.kb,
        model=args.model,
        persist_dir=ROOT / ".chroma",
        max_loops=args.max_loops,
    )
    result = agent.invoke(question)

    print("\n--- Trace ---")
    for msg in result.get("messages", []):
        print(f"• {msg.content[:500]}")

    print("\n--- Answer ---")
    print(result.get("generation", ""))
    print(f"\n(retrieval_grade={result.get('retrieval_grade')}, used_web={result.get('used_web')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
