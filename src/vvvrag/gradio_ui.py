"""Gradio UI: upload a PDF, index both approaches, compare side-by-side."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import gradio as gr

from .core import get_chat_model, get_embeddings, now_ms
from .vector import build_index as vec_build, query as vec_query
from .vectorless import build_index as vless_build, query as vless_query

THEME = gr.themes.Soft()

HEADER = """
# Vector vs Vectorless RAG
**AWS Bedrock** &bull; Upload a PDF, index it with both approaches, then compare answers side-by-side.
"""

CSS = """
.vec-header  { background: #1a5276; color: white; padding: 12px 16px; border-radius: 8px 8px 0 0; margin-bottom: 0; }
.vless-header { background: #7d3c98; color: white; padding: 12px 16px; border-radius: 8px 8px 0 0; margin-bottom: 0; }
.result-box  { border: 1px solid #333; border-radius: 0 0 8px 8px; padding: 16px; min-height: 200px; }
"""


def _pdf_path(f: Any) -> str | None:
    if not f:
        return None
    return f if isinstance(f, str) else getattr(f, "name", str(f))


def index_document(file_obj: Any):
    path = _pdf_path(file_obj)
    if not path or not Path(path).is_file():
        yield "Upload a PDF first.", None
        return

    yield "**Indexing...** Building vector store (embeddings + FAISS) and document tree...", None

    try:
        t0 = now_ms()
        embeddings = get_embeddings()
        llm = get_chat_model()
        store = vec_build(path, embeddings)
        tree = vless_build(path)
        ms = now_ms() - t0
        state = {"store": store, "tree": tree, "llm": llm, "embeddings": embeddings}
        md = (
            f"**Indexed in {ms / 1000:.1f}s**\n\n"
            f"| | Detail |\n|---|---|\n"
            f"| **Vector** | {store.index.ntotal} chunks in FAISS |\n"
            f"| **Vectorless** | {len(tree)} top-level tree node(s) |"
        )
        yield md, state
    except Exception as e:
        yield f"**Error:** `{e}`\n```\n{traceback.format_exc()}\n```", None


def compare(question: str, expert_rules: str, state: dict | None, k: int):
    q = (question or "").strip()
    placeholder = "*Waiting...*"
    if not q:
        yield "*Enter a question.*", "*Enter a question.*"
        return
    if not state:
        yield "*Index a PDF first.*", "*Index a PDF first.*"
        return

    yield "**Running vector RAG...**", "**Waiting for vector RAG to finish...**"

    try:
        t1 = now_ms()
        vr = vec_query(q, state["store"], state["llm"], k=max(1, int(k)))
        dt1 = now_ms() - t1

        src = "\n".join(f"- p.{s['page']}: {s['text'][:120]}..." for s in vr["sources"][:6])
        left = (
            f"**{dt1 / 1000:.1f}s**\n\n"
            f"### Answer\n{vr['answer']}\n\n"
            f"### Retrieved chunks\n{src or '_(none)_'}"
        )
        yield left, "**Running vectorless RAG (LLM tree routing)...**"

        t2 = now_ms()
        vl = vless_query(q, state["tree"], state["llm"], expert_rules=expert_rules.strip() or None, k=max(3, int(k)))
        dt2 = now_ms() - t2

        nds = "\n".join(f"- {n['title']} ({n['pages']})" for n in vl["nodes"][:8])
        thinking = vl.get("route", {}).get("thinking", "")
        right = (
            f"**{dt2 / 1000:.1f}s**\n\n"
            f"### Answer\n{vl['answer']}\n\n"
            f"### Routed nodes\n{nds or '_(none)_'}"
        )
        if thinking:
            right += f"\n\n### LLM routing reasoning\n> {thinking}"

        yield left, right
    except Exception as e:
        err = f"**Error:** `{e}`\n```\n{traceback.format_exc()}\n```"
        yield err, err


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Vector vs Vectorless RAG", theme=THEME, css=CSS) as demo:
        gr.Markdown(HEADER)
        state = gr.State(None)

        with gr.Group():
            with gr.Row():
                pdf_in = gr.File(label="PDF document", file_types=[".pdf"], scale=3)
                index_btn = gr.Button("Index PDF", variant="primary", scale=1, size="lg")
            status = gr.Markdown("*No document indexed yet.*")

        with gr.Group():
            with gr.Row():
                question = gr.Textbox(label="Question", lines=2, placeholder="Ask something about the document...", scale=4)
                k = gr.Slider(1, 15, value=5, step=1, label="Top-k", scale=1)
            expert = gr.Textbox(label="Expert rules (vectorless only, optional)", lines=1, placeholder="e.g. Focus on financial data...")
            run_btn = gr.Button("Compare", variant="primary", size="lg")

        with gr.Row(equal_height=True):
            with gr.Column():
                gr.Markdown("### Vector RAG\n`PyPDFLoader → Splitter → FAISS → Bedrock`", elem_classes=["vec-header"])
                out_vec = gr.Markdown("*Results will appear here.*", elem_classes=["result-box"])
            with gr.Column():
                gr.Markdown("### Vectorless RAG\n`PDF → Tree Index → LLM Routing → Bedrock`", elem_classes=["vless-header"])
                out_vless = gr.Markdown("*Results will appear here.*", elem_classes=["result-box"])

        index_btn.click(index_document, [pdf_in], [status, state])
        run_btn.click(compare, [question, expert, state, k], [out_vec, out_vless])

    return demo


def launch_ui(*, server_name="127.0.0.1", server_port=7860, share=False):
    build_demo().launch(server_name=server_name, server_port=server_port, share=share)
