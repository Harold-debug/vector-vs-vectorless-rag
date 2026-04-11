"""
Vector RAG: PyPDFLoader → RecursiveCharacterTextSplitter → BedrockEmbeddings → FAISS
"""

from .pipeline import build_index, query

__all__ = ["build_index", "query"]
