import csv
import json
import time
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reduce noisy Chroma telemetry warnings in local benchmark runs.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import app as project_app

OUTPUT_DIR = BASE_DIR / "outputs"
QUESTIONS_PATH = BASE_DIR / "questions.csv"
RAW_RESULTS_PATH = OUTPUT_DIR / "raw_results.jsonl"
MANUAL_TEMPLATE_PATH = BASE_DIR / "manual_scores_template.csv"

METHODS = ("llm_only", "rag_only", "graph_rag")
NEO_RETRY = 3
NEO_RETRY_SLEEP_SEC = 1.2
_NEO_SSC_FALLBACK_TRIED = False


def load_questions(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = (row.get("question") or "").strip()
            if not q:
                continue
            rows.append(
                {
                    "id": (row.get("id") or "").strip(),
                    "category": (row.get("category") or "").strip(),
                    "question": q,
                    "reference_root_path": (row.get("reference_root_path") or "").strip(),
                }
            )
    return rows


def build_rag_context(question: str, top_k: int = 5) -> List[Dict[str, Any]]:
    return project_app.hybrid_kb_retrieve(question, focus_terms=[], top_k=top_k)


def build_kg_context(question: str, limit: int = 8) -> List[Dict[str, str]]:
    global _NEO_SSC_FALLBACK_TRIED
    for attempt in range(1, NEO_RETRY + 1):
        # Keep using neo4j+s only in this environment.
        # Recreate driver each retry to avoid stale routed connections.
        project_app.neo4j_service._close_driver()
        hits = project_app.neo4j_service.search_triplets(question, limit=limit)
        if hits:
            return hits
        err = str(project_app.neo4j_service.last_error or "")
        uri = str(project_app.neo4j_service.uri or "")
        if ("routing information" in err.lower()) and (not _NEO_SSC_FALLBACK_TRIED) and uri.startswith("neo4j+s://"):
            ssc_uri = "neo4j+ssc://" + uri[len("neo4j+s://") :]
            project_app.neo4j_service.update_config(
                ssc_uri,
                project_app.neo4j_service.user,
                project_app.neo4j_service.password,
                project_app.neo4j_service.database,
            )
            _NEO_SSC_FALLBACK_TRIED = True
            print(f"[Neo4j] fallback uri => {ssc_uri} (test-only)")
        if attempt < NEO_RETRY:
            time.sleep(NEO_RETRY_SLEEP_SEC * attempt)
    return []


def answer_llm_only(question: str, model_name: str = "") -> Tuple[str, Dict[str, Any]]:
    prompt = f"用户问题：{question}\n请直接回答，结论简洁可执行，控制在220字以内。"
    start = time.perf_counter()
    text = project_app.cloud_llm.chat(prompt, model=model_name, max_models_to_try=1, timeout_seconds_override=90)
    latency_ms = int((time.perf_counter() - start) * 1000)
    answer = project_app.ensure_complete_sentences((text or "").strip() or "模型未返回内容。")
    meta = {
        "latency_ms": latency_ms,
        "error": project_app.cloud_llm.last_error,
        "model": project_app.cloud_llm.last_model or model_name or project_app.cloud_llm.default_model,
        "kb_count": 0,
        "kg_count": 0,
    }
    return answer, meta


def answer_rag_only(question: str, model_name: str = "") -> Tuple[str, Dict[str, Any]]:
    kb_hits = build_rag_context(question, top_k=5)
    context = "\n".join([f"- {it.get('title', '文档片段')}: {str(it.get('text', ''))[:160]}" for it in kb_hits])
    prompt = (
        f"用户问题：{question}\n\n"
        f"【文档片段（RAG）】\n{context if context else '- 无'}\n\n"
        "请仅依据文档片段回答。若证据不足，明确写“证据不足”。控制在220字以内。"
    )
    start = time.perf_counter()
    text = project_app.cloud_llm.chat(prompt, model=model_name, max_models_to_try=1, timeout_seconds_override=120)
    latency_ms = int((time.perf_counter() - start) * 1000)
    answer = project_app.ensure_complete_sentences((text or "").strip() or "模型未返回内容。")
    meta = {
        "latency_ms": latency_ms,
        "error": project_app.cloud_llm.last_error,
        "model": project_app.cloud_llm.last_model or model_name or project_app.cloud_llm.default_model,
        "kb_count": len(kb_hits),
        "kg_count": 0,
    }
    return answer, meta


def answer_graph_rag(question: str, model_name: str = "") -> Tuple[str, Dict[str, Any]]:
    kb_hits = build_rag_context(question, top_k=5)
    kg_hits = build_kg_context(question, limit=8)
    kb_context = "\n".join([f"- {it.get('title', '文档片段')}: {str(it.get('text', ''))[:130]}" for it in kb_hits])
    kg_context = "\n".join([f"- ({t.get('head','')})-[{t.get('rel','')}]->({t.get('tail','')})" for t in kg_hits])
    prompt = (
        f"用户问题：{question}\n\n"
        f"【知识图谱三元组】\n{kg_context if kg_context else '- 无'}\n\n"
        f"【文档片段补充】\n{kb_context if kb_context else '- 无'}\n\n"
        "请先给出结论，再给出简短证据链（部件-现象-原因-措施）。若证据不足请明确指出。控制在260字以内。"
    )
    start = time.perf_counter()
    text = project_app.cloud_llm.chat(prompt, model=model_name, max_models_to_try=1, timeout_seconds_override=120)
    latency_ms = int((time.perf_counter() - start) * 1000)
    answer = project_app.ensure_complete_sentences((text or "").strip() or "模型未返回内容。")
    meta = {
        "latency_ms": latency_ms,
        "error": project_app.cloud_llm.last_error,
        "model": project_app.cloud_llm.last_model or model_name or project_app.cloud_llm.default_model,
        "kb_count": len(kb_hits),
        "kg_count": len(kg_hits),
    }
    return answer, meta


def run_one(method: str, question: str) -> Tuple[str, Dict[str, Any]]:
    if method == "llm_only":
        return answer_llm_only(question)
    if method == "rag_only":
        return answer_rag_only(question)
    if method == "graph_rag":
        return answer_graph_rag(question)
    raise ValueError(f"Unsupported method: {method}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_questions(QUESTIONS_PATH)
    all_rows: List[Dict[str, Any]] = []
    print(
        f"[Neo4j config] uri={project_app.neo4j_service.uri} "
        f"user={project_app.neo4j_service.user} db={project_app.neo4j_service.database}"
    )

    for q in questions:
        for method in METHODS:
            answer, meta = run_one(method, q["question"])
            row = {
                "question_id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "reference_root_path": q["reference_root_path"],
                "method": method,
                "answer": answer,
                "latency_ms": meta["latency_ms"],
                "model": meta["model"],
                "kb_count": meta["kb_count"],
                "kg_count": meta["kg_count"],
                "kg_connected": bool(meta["kg_count"] > 0 or not meta["error"]),
                "llm_error": meta["error"],
            }
            all_rows.append(row)
            print(f"[{method}] {q['id']} done, latency={meta['latency_ms']}ms")

    with RAW_RESULTS_PATH.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = [
        "question_id",
        "category",
        "question",
        "reference_root_path",
        "method",
        "answer",
        "latency_ms",
        "model",
        "kb_count",
        "kg_count",
        "kg_connected",
        "llm_error",
        "is_correct",
        "evidence_hit",
        "hallucination",
        "root_path_hit",
        "annotator",
        "notes",
    ]
    with MANUAL_TEMPLATE_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(
                {
                    **row,
                    "is_correct": "",
                    "evidence_hit": "",
                    "hallucination": "",
                    "root_path_hit": "",
                    "annotator": "",
                    "notes": "",
                }
            )

    print(f"\nraw results: {RAW_RESULTS_PATH}")
    print(f"manual score template: {MANUAL_TEMPLATE_PATH}")


if __name__ == "__main__":
    main()
