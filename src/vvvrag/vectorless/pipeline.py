"""
Vectorless RAG pipeline (PageIndex-style):

  PDF → tree index → LLM reasons over tree → retrieve nodes → ChatBedrock answer
"""

from __future__ import annotations

from typing import Any

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage

from .router import find_nodes, route
from .tree import PDFNode, build_outline_tree, build_page_tree

ANSWER_SYSTEM = (
    "Answer the question using ONLY the provided context. "
    "Cite sources by section title and page range like (Section X, pp.1-3). "
    "If the context does not contain the answer, say so."
)


def build_index(pdf_path: str) -> list[PDFNode]:
    """Build the tree index. Uses PDF outline if available, else page-level."""
    tree = build_outline_tree(pdf_path)
    return tree if tree else build_page_tree(pdf_path)


def query(
    question: str,
    tree: list[PDFNode],
    llm: ChatBedrock,
    *,
    expert_rules: str | None = None,
    k: int = 6,
) -> dict[str, Any]:
    routing = route(question, tree, llm, expert_rules=expert_rules)
    node_ids = [str(x) for x in (routing.get("node_list") or [])][:k]
    nodes = find_nodes(tree, set(node_ids))

    if not nodes:
        return {"answer": "No relevant sections found.", "route": routing, "nodes": []}

    context = "\n\n---\n\n".join(
        f"[{n.title}, pp.{n.page_start}-{n.page_end}]\n{n.text}" for n in nodes
    )
    resp = llm.invoke([
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"Question:\n{question}\n\nContext:\n{context}"),
    ])
    answer = resp.content if isinstance(resp.content, str) else str(resp.content)

    return {
        "answer": answer.strip(),
        "route": routing,
        "nodes": [
            {"node_id": n.node_id, "title": n.title, "pages": f"{n.page_start}-{n.page_end}"}
            for n in nodes
        ],
    }
