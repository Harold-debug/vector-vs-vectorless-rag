"""LangGraph wrapper for the vectorless RAG pipeline."""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_aws import ChatBedrock
from langgraph.graph import END, StateGraph

from .pipeline import query
from .tree import PDFNode


class State(TypedDict, total=False):
    question: str
    result: dict[str, Any]


def compile_graph(
    tree: list[PDFNode],
    llm: ChatBedrock,
    *,
    expert_rules: str | None = None,
    k: int = 6,
):
    def run(state: State) -> State:
        return {
            "question": state["question"],
            "result": query(state["question"], tree, llm, expert_rules=expert_rules, k=k),
        }

    g = StateGraph(State)
    g.add_node("rag", run)
    g.set_entry_point("rag")
    g.add_edge("rag", END)
    return g.compile()
