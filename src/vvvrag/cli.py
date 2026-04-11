from __future__ import annotations

from typing import Optional

import typer
from rich import print

from . import __version__

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command("version")
def version():
    """Print package version."""
    print(__version__)


@app.command("bench")
def bench(
    pdf: str = typer.Option(..., help="Path to a PDF file."),
    qas: str = typer.Option(..., help="Path to a JSONL QA file."),
    out: str = typer.Option("results/run.json", help="Output JSON path."),
    k: int = typer.Option(5, help="Top-k chunks/nodes."),
    expert_rules: Optional[str] = typer.Option(None, help="Expert routing rules (vectorless)."),
):
    """Run vector + vectorless RAG benchmark."""
    from .bench import run_benchmark

    payload = run_benchmark(pdf_path=pdf, qas_path=qas, out_path=out, k=k, expert_rules=expert_rules)
    print(f"[bold]Wrote[/bold] {out}")
    print(payload["summary"])


@app.command("ui")
def ui(
    host: str = typer.Option("127.0.0.1", help="Bind address."),
    port: int = typer.Option(7860, help="Port."),
    share: bool = typer.Option(False, help="Public Gradio link."),
):
    """Launch Gradio UI."""
    from .gradio_ui import launch_ui

    launch_ui(server_name=host, server_port=port, share=share)


def main():
    app()
