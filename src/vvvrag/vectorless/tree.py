"""
Build a hierarchical tree index from the PDF outline / table of contents.
Fallback: one node per page when no outline exists.

This replaces the vector DB — the LLM reasons over the tree to find
relevant sections instead of computing cosine similarity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader


@dataclass
class PDFNode:
    node_id: str
    title: str
    page_start: int
    page_end: int
    text: str
    children: list[PDFNode] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "title": self.title,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "text": self.text[:300],
            "children": [c.to_dict() for c in self.children],
        }


def _load_pages(pdf_path: str | Path) -> list[str]:
    """Use PyPDFLoader (same loader as vector path) to extract per-page text."""
    docs = PyPDFLoader(str(pdf_path)).load()
    return [d.page_content for d in docs]


def _load_outline(pdf_path: str | Path):
    """Best-effort parse of PDF bookmarks via pypdf."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    outline = getattr(reader, "outline", None) or getattr(reader, "outlines", None)
    if not outline:
        return [], reader
    return outline, reader


def build_page_tree(pdf_path: str | Path) -> list[PDFNode]:
    """Fallback: one node per page."""
    pages = _load_pages(pdf_path)
    return [
        PDFNode(node_id=f"{i:04d}", title=f"Page {i}", page_start=i, page_end=i, text=txt)
        for i, txt in enumerate(pages, start=1)
    ]


def build_outline_tree(pdf_path: str | Path) -> list[PDFNode]:
    """Build tree from PDF outline bookmarks. Returns [] if none found."""
    pages = _load_pages(pdf_path)
    outline_raw, reader = _load_outline(pdf_path)
    if not outline_raw:
        return []

    def walk(items) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for it in items:
            if isinstance(it, list):
                out.extend(walk(it))
                continue
            title = getattr(it, "title", None) or str(it)
            try:
                page_idx = int(reader.get_destination_page_number(it)) + 1
            except Exception:
                page_idx = None
            children: list[dict[str, Any]] = []
            if getattr(it, "children", None):
                try:
                    children = walk(list(it.children))
                except Exception:
                    pass
            out.append({"title": title, "page_index": page_idx, "children": children})
        return out

    try:
        flat = walk(outline_raw)
    except Exception:
        return []
    if not flat:
        return []

    counter = 0

    def mk_nodes(items: list[dict[str, Any]], default_end: int) -> list[PDFNode]:
        nonlocal counter
        nodes: list[PDFNode] = []
        starts = [it.get("page_index") for it in items]
        for idx, it in enumerate(items):
            counter += 1
            nid = f"{counter:04d}"
            title = str(it.get("title") or f"Section {counter}")
            start = it.get("page_index") or 1
            next_start = next((s for s in starts[idx + 1 :] if s), None)
            end = (next_start - 1) if (next_start and next_start > start) else default_end
            start, end = max(1, min(start, len(pages))), max(start, min(end, len(pages)))
            text = "\n\n".join(pages[start - 1 : end])
            children = mk_nodes(it.get("children") or [], end) if it.get("children") else []
            nodes.append(PDFNode(node_id=nid, title=title, page_start=start, page_end=end, text=text, children=children))
        return nodes

    return mk_nodes(flat, default_end=len(pages))
