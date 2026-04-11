# Vector vs Vectorless RAG

Side-by-side comparison of **Vector RAG** vs **Vectorless RAG** (PageIndex-style) on the same PDF, using **AWS Bedrock**, **LangChain**, and **LangGraph**.

## Architecture

### Vector RAG (`vector/`)

```
PDF → PyPDFLoader → RecursiveCharacterTextSplitter → BedrockEmbeddings → FAISS → retriever → ChatBedrock
```

Standard chunking + embedding approach (same pattern as [bluegen](https://github.com/user/bluegen) and [RAG-Tutorials](https://github.com/krishnaik06/RAG-Tutorials)).

### Vectorless RAG (`vectorless/`)

```
PDF → tree index (outline / page-level) → LLM reasons over tree → retrieve nodes → ChatBedrock
```

No embeddings, no vector DB. The LLM acts like an expert scanning a Table of Contents. Based on the [PageIndex crash course](https://github.com/krishnaik06/RAG-Tutorials/blob/main/PageIndex_Vectorless_RAG_CrashCourse%20(1).ipynb).

## Quickstart

```bash
uv venv && source .venv/bin/activate && uv sync
```

### AWS credentials

Standard AWS credential chain (`~/.aws/credentials`, env vars, or IAM role).

### `.env`

```bash
AWS_REGION=us-east-1
BEDROCK_CHAT_MODEL_ID=anthropic.claude-3-5-haiku-20241022-v1:0
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0
```

### Gradio UI

```bash
uv run python -m vvvrag ui
# http://127.0.0.1:7860
```

### Benchmark

```bash
uv run python -m vvvrag bench --pdf data/pdfs/sample.pdf --qas data/qas/sample.jsonl
```

QA file format (`.jsonl`):

```json
{"id":"q1","question":"What is X?","answer":"X is ..."}
```

## Project structure

```
src/vvvrag/
├── core/              # Bedrock factories + utils
│   ├── bedrock.py     #   get_chat_model(), get_embeddings()
│   └── utils.py
├── vector/            # Vector RAG
│   ├── pipeline.py    #   PyPDFLoader → splitter → FAISS → ChatBedrock
│   └── graph.py       #   LangGraph wrapper
├── vectorless/        # Vectorless RAG
│   ├── tree.py        #   PDF outline → PDFNode tree
│   ├── router.py      #   LLM tree routing
│   ├── pipeline.py    #   tree → route → answer
│   └── graph.py       #   LangGraph wrapper
├── bench.py           # Benchmark harness
├── cli.py             # CLI: bench, ui, version
└── gradio_ui.py       # Gradio comparison UI
```
