"""LangGraph wrapper for the vector RAG pipeline."""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_community.vectorstores import FAISS
from langgraph.graph import END, StateGraph

from .pipeline import query


class State(TypedDict, total=False):
    question: str
    result: dict[str, Any]


def compile_graph(store: FAISS, llm: ChatBedrock, *, k: int = 5):
    def run(state: State) -> State:
        return {"question": state["question"], "result": query(state["question"], store, llm, k=k)}

    g = StateGraph(State)
    g.add_node("rag", run)
    g.set_entry_point("rag")
    g.add_edge("rag", END)
    return g.compile()
