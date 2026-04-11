"""
LLM tree routing — the core of vectorless retrieval.

Instead of cosine similarity, we send the compressed document tree to a
chat model which reasons over the structure and returns relevant node IDs.
Matches the approach from the PageIndex crash-course notebook.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage

from .tree import PDFNode

ROUTE_SYSTEM = (
    "You are routing a question to the most relevant document sections. "
    "You receive a document tree (like a Table of Contents with summaries). "
    "Return ONLY a JSON object: {\"thinking\": \"...\", \"node_list\": [\"0001\", ...]}"
)


def _compress(nodes: list[PDFNode], max_chars: int = 200) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for n in nodes:
        entry: dict[str, Any] = {
            "node_id": n.node_id,
            "title": n.title,
            "pages": f"{n.page_start}-{n.page_end}",
            "summary": n.text.replace("\n", " ")[:max_chars],
        }
        if n.children:
            entry["children"] = _compress(n.children, max_chars)
        out.append(entry)
    return out


def _parse_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        text = m.group(1).strip()
    return json.loads(text)


def find_nodes(nodes: list[PDFNode], ids: set[str]) -> list[PDFNode]:
    found: list[PDFNode] = []
    for n in nodes:
        if n.node_id in ids:
            found.append(n)
        if n.children:
            found.extend(find_nodes(n.children, ids))
    return found


def route(
    question: str,
    tree: list[PDFNode],
    llm: ChatBedrock,
    *,
    expert_rules: str | None = None,
) -> dict[str, Any]:
    user_payload = {
        "question": question,
        "document_tree": _compress(tree),
        "expert_rules": expert_rules or "",
        "instructions": "Pick the smallest set of nodes that contain the answer.",
    }
    user_msg = json.dumps(user_payload, ensure_ascii=False) + "\n\nReply with JSON only."
    resp = llm.invoke([SystemMessage(content=ROUTE_SYSTEM), HumanMessage(content=user_msg)])
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    return _parse_json(content)
