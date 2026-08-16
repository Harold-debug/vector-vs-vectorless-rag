"""Gradio UI: test each RAG approach individually + benchmark both side-by-side."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import gradio as gr

from .core import get_chat_model, get_embeddings, now_ms
from .vector import build_index as vec_build, query as vec_query
from .vectorless import build_index as vless_build, query as vless_query

THEME = gr.themes.Soft()

CSS = """
.vec-accent   { border-left: 4px solid #1a5276 !important; }
.vless-accent { border-left: 4px solid #7d3c98 !important; }
.bench-vec    { background: #1a5276; color: white; padding: 10px 14px; border-radius: 8px 8px 0 0; }
.bench-vless  { background: #7d3c98; color: white; padding: 10px 14px; border-radius: 8px 8px 0 0; }
.result-box   { border: 1px solid #444; border-radius: 0 0 8px 8px; padding: 16px; min-height: 180px; }
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pdf_path(f: Any) -> str | None:
    if not f:
        return None
    return f if isinstance(f, str) else getattr(f, "name", str(f))


def _fmt_vec(vr: dict, dt: int) -> str:
    src = "\n".join(f"- p.{s['page']}: {s['text'][:120]}..." for s in vr["sources"][:6])
    return (
        f"**{dt / 1000:.1f}s**\n\n"
        f"### Answer\n{vr['answer']}\n\n"
        f"### Retrieved chunks\n{src or '_(none)_'}"
    )


def _fmt_vless(vl: dict, dt: int) -> str:
    nds = "\n".join(f"- {n['title']} ({n['pages']})" for n in vl["nodes"][:8])
    thinking = vl.get("route", {}).get("thinking", "")
    md = (
        f"**{dt / 1000:.1f}s**\n\n"
        f"### Answer\n{vl['answer']}\n\n"
        f"### Routed nodes\n{nds or '_(none)_'}"
    )
    if thinking:
        md += f"\n\n### Routing reasoning\n> {thinking}"
    return md


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

def index_document(file_obj: Any):
    path = _pdf_path(file_obj)
    if not path or not Path(path).is_file():
        yield "Upload a PDF first.", None
        return

    yield "**Indexing...** Building FAISS vector store and document tree...", None
    try:
        t0 = now_ms()
        embeddings = get_embeddings()
        llm = get_chat_model()
        store = vec_build(path, embeddings)
        tree = vless_build(path)
        dt = now_ms() - t0
        state = {"store": store, "tree": tree, "llm": llm, "pdf": path}
        md = (
            f"**Indexed in {dt / 1000:.1f}s**\n\n"
            f"| Approach | Detail |\n|---|---|\n"
            f"| Vector | {store.index.ntotal} chunks in FAISS |\n"
            f"| Vectorless | {len(tree)} top-level tree node(s) |"
        )
        yield md, state
    except Exception as e:
        yield f"**Error:** `{e}`\n```\n{traceback.format_exc()}\n```", None


# ---------------------------------------------------------------------------
# Single-approach handlers
# ---------------------------------------------------------------------------

def run_vector(question: str, k: int, state: dict | None):
    q = (question or "").strip()
    if not q:
        yield "*Enter a question.*"
        return
    if not state:
        yield "*Index a PDF first (use the section above).*"
        return
    yield "**Running vector RAG...**"
    try:
        t = now_ms()
        vr = vec_query(q, state["store"], state["llm"], k=max(1, int(k)))
        yield _fmt_vec(vr, now_ms() - t)
    except Exception as e:
        yield f"**Error:** `{e}`\n```\n{traceback.format_exc()}\n```"


def run_vectorless(question: str, expert_rules: str, k: int, state: dict | None):
    q = (question or "").strip()
    if not q:
        yield "*Enter a question.*"
        return
    if not state:
        yield "*Index a PDF first (use the section above).*"
        return
    yield "**Running vectorless RAG (LLM tree routing)...**"
    try:
        t = now_ms()
        vl = vless_query(q, state["tree"], state["llm"], expert_rules=expert_rules.strip() or None, k=max(3, int(k)))
        yield _fmt_vless(vl, now_ms() - t)
    except Exception as e:
        yield f"**Error:** `{e}`\n```\n{traceback.format_exc()}\n```"


# ---------------------------------------------------------------------------
# Benchmark (both side-by-side)
# ---------------------------------------------------------------------------

def run_benchmark(question: str, expert_rules: str, k: int, state: dict | None):
    q = (question or "").strip()
    if not q:
        yield "*Enter a question.*", "*Enter a question.*"
        return
    if not state:
        yield "*Index a PDF first.*", "*Index a PDF first.*"
        return

    yield "**Running vector RAG...**", "**Waiting...**"
    try:
        t1 = now_ms()
        vr = vec_query(q, state["store"], state["llm"], k=max(1, int(k)))
        left = _fmt_vec(vr, now_ms() - t1)

        yield left, "**Running vectorless RAG...**"

        t2 = now_ms()
        vl = vless_query(q, state["tree"], state["llm"], expert_rules=expert_rules.strip() or None, k=max(3, int(k)))
        right = _fmt_vless(vl, now_ms() - t2)

        yield left, right
    except Exception as e:
        err = f"**Error:** `{e}`\n```\n{traceback.format_exc()}\n```"
        yield err, err


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Vector vs Vectorless RAG") as demo:
        gr.Markdown("# Vector vs Vectorless RAG\n**AWS Bedrock** · Upload a PDF, then test each approach or benchmark both.")
        state = gr.State(None)

        # ---- Shared: PDF upload + index ----
        with gr.Group():
            gr.Markdown("### 1. Load & index a PDF")
            with gr.Row():
                pdf_in = gr.File(label="PDF document", file_types=[".pdf"], scale=3)
                index_btn = gr.Button("Index PDF", variant="primary", scale=1, size="lg")
            status = gr.Markdown("*No document indexed yet.*")

        # ---- Tabs ----
        with gr.Tabs():
            # -- Tab 1: Vector RAG --
            with gr.Tab("Vector RAG"):
                gr.Markdown(
                    "### 2a. Vector RAG\n"
                    "`PyPDFLoader → RecursiveCharacterTextSplitter → BedrockEmbeddings → FAISS → ChatBedrock`"
                )
                with gr.Row():
                    vec_q = gr.Textbox(label="Question", lines=2, placeholder="Ask something...", scale=4)
                    vec_k = gr.Slider(1, 15, value=5, step=1, label="Top-k", scale=1)
                vec_btn = gr.Button("Run Vector RAG", variant="primary")
                vec_out = gr.Markdown("*Results will appear here.*", elem_classes=["vec-accent"])

            # -- Tab 2: Vectorless RAG --
            with gr.Tab("Vectorless RAG"):
                gr.Markdown(
                    "### 2b. Vectorless RAG (PageIndex-style)\n"
                    "`PyPDFLoader → Tree Index → LLM Tree Routing → ChatBedrock`\n\n"
                    "No embeddings, no vector DB. The LLM reasons over the document structure."
                )
                with gr.Row():
                    vless_q = gr.Textbox(label="Question", lines=2, placeholder="Ask something...", scale=4)
                    vless_k = gr.Slider(1, 15, value=5, step=1, label="Top-k nodes", scale=1)
                vless_expert = gr.Textbox(label="Expert rules (optional)", lines=1, placeholder="e.g. Focus on financial data...")
                vless_btn = gr.Button("Run Vectorless RAG", variant="primary")
                vless_out = gr.Markdown("*Results will appear here.*", elem_classes=["vless-accent"])

            # -- Tab 3: Benchmark --
            with gr.Tab("Benchmark"):
                gr.Markdown(
                    "### 3. Benchmark — side-by-side\n"
                    "Same question, same PDF. Compare speed, answer quality, and retrieval strategy."
                )
                with gr.Row():
                    bench_q = gr.Textbox(label="Question", lines=2, placeholder="Ask something...", scale=4)
                    bench_k = gr.Slider(1, 15, value=5, step=1, label="Top-k", scale=1)
                bench_expert = gr.Textbox(label="Expert rules (vectorless only)", lines=1, placeholder="Optional...")
                bench_btn = gr.Button("Compare both", variant="primary", size="lg")
                with gr.Row(equal_height=True):
                    with gr.Column():
                        gr.Markdown("### Vector RAG", elem_classes=["bench-vec"])
                        bench_vec_out = gr.Markdown("*Waiting...*", elem_classes=["result-box"])
                    with gr.Column():
                        gr.Markdown("### Vectorless RAG", elem_classes=["bench-vless"])
                        bench_vless_out = gr.Markdown("*Waiting...*", elem_classes=["result-box"])

        # ---- Events ----
        index_btn.click(index_document, [pdf_in], [status, state])
        vec_btn.click(run_vector, [vec_q, vec_k, state], [vec_out])
        vless_btn.click(run_vectorless, [vless_q, vless_expert, vless_k, state], [vless_out])
        bench_btn.click(run_benchmark, [bench_q, bench_expert, bench_k, state], [bench_vec_out, bench_vless_out])

    return demo


def launch_ui(*, server_name="127.0.0.1", server_port=7860, share=False):
    build_demo().launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        theme=THEME,
        css=CSS,
    )
