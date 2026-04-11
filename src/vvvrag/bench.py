"""Benchmark harness: run both approaches on the same PDF + QA set."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core import get_chat_model, get_embeddings, now_ms, sha1_text, write_json, disk_usage_bytes
from .vector import build_index as vec_build, query as vec_query
from .vectorless import build_index as vless_build, query as vless_query


@dataclass(frozen=True)
class QAPair:
    id: str
    question: str
    answer: str


def load_qas(path: str | Path) -> list[QAPair]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"QA file not found: {p.resolve()}. "
            'Create a .jsonl file: {"id":"q1","question":"...","answer":"..."}'
        )
    qas: list[QAPair] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        qas.append(QAPair(id=str(obj["id"]), question=str(obj["question"]), answer=str(obj["answer"])))
    return qas


def _normalize(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def run_benchmark(
    *,
    pdf_path: str,
    qas_path: str,
    out_path: str = "results/run.json",
    k: int = 5,
    expert_rules: str | None = None,
) -> dict[str, Any]:
    embeddings = get_embeddings()
    llm = get_chat_model()
    qas = load_qas(qas_path)
    run_id = sha1_text(f"{pdf_path}|{qas_path}|{time.time()}")[:12]

    # ---- Vector ----
    save_dir = Path("indexes") / f"vector_{run_id}"
    t0 = now_ms()
    store = vec_build(pdf_path, embeddings, save_dir=save_dir)
    t_vec_index = now_ms() - t0
    vec_bytes = disk_usage_bytes(save_dir)

    # ---- Vectorless ----
    t1 = now_ms()
    tree = vless_build(pdf_path)
    t_vless_index = now_ms() - t1

    # ---- Queries ----
    per_q: list[dict[str, Any]] = []
    t_vec_total = t_vless_total = em_vec = em_vless = 0

    for qa in qas:
        ts = now_ms()
        vr = vec_query(qa.question, store, llm, k=k)
        dt = now_ms() - ts
        t_vec_total += dt
        match_v = _normalize(vr["answer"]) == _normalize(qa.answer)
        em_vec += int(match_v)

        ts2 = now_ms()
        vl = vless_query(qa.question, tree, llm, expert_rules=expert_rules, k=max(3, k))
        dt2 = now_ms() - ts2
        t_vless_total += dt2
        match_vl = _normalize(vl["answer"]) == _normalize(qa.answer)
        em_vless += int(match_vl)

        per_q.append({
            "id": qa.id, "question": qa.question, "gold": qa.answer,
            "vector": {"ms": dt, "answer": vr["answer"], "sources": vr["sources"]},
            "vectorless": {"ms": dt2, "answer": vl["answer"], "nodes": vl["nodes"]},
        })

    n = max(1, len(qas))
    payload = {
        "run_id": run_id, "pdf": pdf_path, "qas": qas_path, "k": k,
        "summary": {
            "n_questions": len(qas),
            "vector": {"index_ms": t_vec_index, "avg_query_ms": t_vec_total / n, "exact_match": em_vec / n, "index_bytes": vec_bytes},
            "vectorless": {"index_ms": t_vless_index, "avg_query_ms": t_vless_total / n, "exact_match": em_vless / n},
        },
        "per_question": per_q,
    }
    write_json(out_path, payload)
    return payload
