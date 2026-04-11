"""
Vector RAG pipeline (same pattern as bluegen + RAG-Tutorials):

  PyPDFLoader → RecursiveCharacterTextSplitter → FAISS.from_documents → retriever → ChatBedrock
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

SYSTEM_PROMPT = (
    "Answer the question using ONLY the provided context. "
    "For every claim, cite the source page in parentheses like (p.3). "
    "If the context does not contain the answer, say so."
)


def load_and_split(
    pdf_path: str | Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(pages)


def build_index(
    pdf_path: str | Path,
    embeddings: BedrockEmbeddings,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    save_dir: str | Path | None = None,
) -> FAISS:
    docs = load_and_split(pdf_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    store = FAISS.from_documents(docs, embeddings)
    if save_dir:
        store.save_local(str(save_dir))
    return store


def query(
    question: str,
    store: FAISS,
    llm: ChatBedrock,
    *,
    k: int = 5,
) -> dict[str, Any]:
    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    docs = retriever.invoke(question)

    context = "\n\n---\n\n".join(
        f"[p.{d.metadata.get('page', '?')}] {d.page_content}" for d in docs
    )
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Question:\n{question}\n\nContext:\n{context}"),
    ])
    answer = response.content if isinstance(response.content, str) else str(response.content)

    return {
        "answer": answer.strip(),
        "sources": [
            {"page": d.metadata.get("page", "?"), "text": d.page_content[:200]}
            for d in docs
        ],
    }
