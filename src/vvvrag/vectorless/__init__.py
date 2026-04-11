"""
Vectorless RAG (PageIndex-style):
  PDF → tree index → LLM tree routing → grounded answer.
  No embeddings, no vector DB.
"""

from .pipeline import build_index, query
from .tree import PDFNode

__all__ = ["PDFNode", "build_index", "query"]
