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
cp .env.example .env   # first time — set AWS region / Bedrock model IDs
make sync
make ui
# http://127.0.0.1:7860
```

Or without Make:

```bash
uv sync --no-editable
uv run --no-editable python run.py ui
```

Use `run.py` (or `make ui`) instead of `python -m vvvrag`. Editable installs break under Documents/iCloud: the `.pth` gets marked hidden and Python 3.13 skips it → `No module named vvvrag`.

### AWS credentials

Use the standard AWS credential chain (profile, env vars, or IAM role). Set region and Bedrock model IDs in `.env`.

### `.env`

```bash
AWS_REGION=us-west-2
BEDROCK_CHAT_MODEL_ID=us.anthropic.claude-sonnet-4-5-20250929-v1:0
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0
UV_NO_EDITABLE=1
```

### Benchmark

```bash
make bench ARGS='--pdf data/pdfs/sample.pdf --qas data/qas/sample.jsonl'
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

