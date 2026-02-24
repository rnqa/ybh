#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from rag30 import OtusStyleRAG, normalize_text, stable_doc_id, tokenize

try:
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover
    CrossEncoder = None  # type: ignore[assignment]


LOGGER = logging.getLogger("retrieval_diagnostic")


def setup_logger(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def serialize_doc(doc: Any, rank: int, score: Optional[float] = None) -> Dict[str, Any]:
    md = doc.metadata or {}
    payload = {
        "rank": rank,
        "doc_id": stable_doc_id(doc),
        "score": score,
        "title": md.get("source_title", ""),
        "law_id": md.get("law_id", ""),
        "source_type": md.get("source_type", ""),
        "hierarchy": md.get("hierarchy_str", ""),
        "source_url": md.get("source_url", ""),
        "preview": (doc.page_content[:240] + "...") if len(doc.page_content) > 240 else doc.page_content,
    }
    return payload


def lexical_overlap_score(text: str, query_tokens: Set[str]) -> float:
    if not query_tokens:
        return 0.0
    tokens = tokenize(text)
    return len(tokens.intersection(query_tokens)) / max(1, len(query_tokens))


def detect_stage_failure(docs: List[Any], query_tokens: Set[str]) -> str:
    if not docs:
        return "retriever_empty"
    overlaps = []
    for d in docs[:5]:
        md = d.metadata or {}
        searchable = f"{md.get('source_title','')} {md.get('hierarchy_str','')} {d.page_content[:1200]}"
        overlaps.append(lexical_overlap_score(searchable, query_tokens))
    if max(overlaps, default=0.0) <= 0.0:
        return "semantic_or_filter_mismatch"
    if max(overlaps, default=0.0) < 0.2:
        return "weak_rerank_or_query_underfit"
    return "ok"


def probe_vector_similarity(rag: OtusStyleRAG, query: str, k: int = 20) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    results = rag.vector_store.similarity_search_with_score(query, k=k)
    for rank, (doc, distance) in enumerate(results, start=1):
        score = 1.0 / (1.0 + float(distance))
        out.append(serialize_doc(doc, rank=rank, score=round(score, 6)))
    return out


def probe_topk_scaling(rag: OtusStyleRAG, query: str, base_top_k: int) -> List[Dict[str, Any]]:
    checks = []
    for top_k in [base_top_k, base_top_k * 2, base_top_k * 3]:
        docs, debug = rag.retrieve(query, top_k=top_k)
        checks.append(
            {
                "top_k": top_k,
                "retrieved": len(docs),
                "debug": {
                    "law_hints": debug.law_hints,
                    "explicit_law_hints": debug.explicit_law_hints,
                    "inferred_law_hints": debug.inferred_law_hints,
                    "entity_tokens": debug.entity_tokens,
                    "law_docs": debug.law_docs,
                    "practice_docs": debug.practice_docs,
                    "lexical_law_docs": debug.lexical_law_docs,
                },
                "top_docs": [serialize_doc(d, i + 1) for i, d in enumerate(docs[:5])],
            }
        )
    return checks


def probe_multiquery_toggle(rag: OtusStyleRAG, query: str, top_k: int) -> List[Dict[str, Any]]:
    original = rag.use_llm_query_expansion
    checks = []
    for mode in [False, True]:
        rag.use_llm_query_expansion = mode
        docs, debug = rag.retrieve(query, top_k=top_k)
        checks.append(
            {
                "llm_query_expansion": mode,
                "retrieved": len(docs),
                "query_variants": debug.query_variants,
                "top_docs": [serialize_doc(d, i + 1) for i, d in enumerate(docs[:5])],
            }
        )
    rag.use_llm_query_expansion = original
    return checks


def build_reformulations(query: str) -> List[str]:
    q = normalize_text(query)
    variants = [
        query,
        f"правовая основа {query}",
        f"полномочия и структура {query}",
        f"{query} федеральный закон фкз",
    ]
    if "суд" in q:
        variants.append(f"{query} место в судебной системе рф")
    if "мвд" in q or "фсб" in q or "прокуратур" in q or "ск рф" in q:
        variants.append(f"{query} подотчетность порядок формирования")
    dedup = []
    for v in variants:
        if v not in dedup:
            dedup.append(v)
    return dedup[:6]


def probe_reformulations(rag: OtusStyleRAG, query: str, top_k: int) -> List[Dict[str, Any]]:
    out = []
    for variant in build_reformulations(query):
        docs, debug = rag.retrieve(variant, top_k=top_k)
        out.append(
            {
                "query": variant,
                "retrieved": len(docs),
                "debug_hints": {
                    "law_hints": debug.law_hints,
                    "entity_tokens": debug.entity_tokens,
                },
                "top_docs": [serialize_doc(d, i + 1) for i, d in enumerate(docs[:5])],
            }
        )
    return out


def probe_hybrid_search(rag: OtusStyleRAG, query: str, top_k: int) -> Dict[str, Any]:
    query_tokens = tokenize(query)
    vector_candidates = rag.vector_store.similarity_search_with_score(query, k=max(60, top_k * 10))
    scored: Dict[str, Tuple[float, Any]] = {}
    for rank, (doc, distance) in enumerate(vector_candidates, start=1):
        vec = 1.0 / (1.0 + float(distance))
        md = doc.metadata or {}
        searchable = f"{md.get('source_title','')} {md.get('hierarchy_str','')} {doc.page_content[:1800]}"
        lex = lexical_overlap_score(searchable, query_tokens)
        combined = 0.65 * vec + 0.35 * lex + (1.0 / (100.0 + rank))
        doc_id = stable_doc_id(doc)
        if doc_id not in scored or combined > scored[doc_id][0]:
            scored[doc_id] = (combined, doc)

    ranked = sorted(scored.values(), key=lambda x: x[0], reverse=True)[:top_k]
    return {
        "hybrid_top_docs": [serialize_doc(doc, i + 1, round(score, 6)) for i, (score, doc) in enumerate(ranked)],
        "candidate_count": len(scored),
    }


def probe_reranker(rag: OtusStyleRAG, query: str, top_k: int) -> Dict[str, Any]:
    candidates = rag.vector_store.similarity_search(query, k=max(40, top_k * 8))
    if not candidates:
        return {"enabled": False, "reason": "no_candidates", "top_docs": []}

    model = None
    if CrossEncoder is not None:
        try:
            model = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
        except Exception:
            model = None

    if model is not None:
        pairs = []
        for d in candidates:
            md = d.metadata or {}
            text = f"{md.get('source_title','')} {md.get('hierarchy_str','')} {d.page_content[:1200]}"
            pairs.append((query, text))
        scores = model.predict(pairs, batch_size=16, show_progress_bar=False)
        ranked = sorted(zip(scores, candidates), key=lambda x: float(x[0]), reverse=True)[:top_k]
        return {
            "enabled": True,
            "model": "BAAI/bge-reranker-v2-m3",
            "top_docs": [serialize_doc(doc, i + 1, round(float(score), 6)) for i, (score, doc) in enumerate(ranked)],
        }

    q_tokens = tokenize(query)
    scored = []
    for d in candidates:
        md = d.metadata or {}
        text = f"{md.get('source_title','')} {md.get('hierarchy_str','')} {d.page_content[:1800]}"
        score = lexical_overlap_score(text, q_tokens)
        scored.append((score, d))
    ranked = sorted(scored, key=lambda x: x[0], reverse=True)[:top_k]
    return {
        "enabled": False,
        "model": "lexical_fallback",
        "top_docs": [serialize_doc(doc, i + 1, round(float(score), 6)) for i, (score, doc) in enumerate(ranked)],
    }


def probe_chunk_size_sensitivity(rag: OtusStyleRAG, query: str, sizes: Sequence[int]) -> List[Dict[str, Any]]:
    q_tokens = tokenize(query)
    vector_docs = rag.vector_store.similarity_search(query, k=20)
    out = []
    for size in sizes:
        scores = []
        for d in vector_docs:
            md = d.metadata or {}
            text = f"{md.get('source_title','')} {md.get('hierarchy_str','')} {d.page_content[:size]}"
            scores.append(lexical_overlap_score(text, q_tokens))
        out.append(
            {
                "virtual_chunk_size_chars": size,
                "avg_overlap": round(sum(scores) / max(1, len(scores)), 6),
                "best_overlap": round(max(scores) if scores else 0.0, 6),
            }
        )
    return out


def diagnose_query(
    rag: OtusStyleRAG,
    query: str,
    top_k: int = 8,
    expected_tokens: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    docs, debug = rag.retrieve(query, top_k=top_k)
    q_tokens = tokenize(query)
    stage = detect_stage_failure(docs, q_tokens)

    baseline_docs = [serialize_doc(d, i + 1) for i, d in enumerate(docs[:10])]
    vector_probe = probe_vector_similarity(rag, query, k=max(20, top_k * 3))

    baseline_relevance = False
    if expected_tokens:
        for d in docs[:5]:
            md = d.metadata or {}
            searchable = f"{md.get('source_title','')} {md.get('hierarchy_str','')} {d.page_content[:1600]}"
            if expected_tokens.intersection(tokenize(searchable)):
                baseline_relevance = True
                break
    else:
        baseline_relevance = len(docs) > 0

    diagnostics = {
        "query": query,
        "baseline": {
            "stage_status": stage,
            "retrieved_docs": len(docs),
            "debug": {
                "law_hints": debug.law_hints,
                "explicit_law_hints": debug.explicit_law_hints,
                "inferred_law_hints": debug.inferred_law_hints,
                "entity_tokens": debug.entity_tokens,
                "query_variants": debug.query_variants,
                "law_docs": debug.law_docs,
                "practice_docs": debug.practice_docs,
                "lexical_law_docs": debug.lexical_law_docs,
            },
            "top_docs": baseline_docs,
            "relevance_pass": baseline_relevance,
        },
        "vector_similarity_probe": vector_probe[:15],
    }

    if not baseline_relevance or stage != "ok":
        diagnostics["probes"] = {
            "topk_scaling": probe_topk_scaling(rag, query, base_top_k=top_k),
            "multiquery_toggle": probe_multiquery_toggle(rag, query, top_k=top_k),
            "reformulations": probe_reformulations(rag, query, top_k=top_k),
            "hybrid_search": probe_hybrid_search(rag, query, top_k=top_k),
            "reranker": probe_reranker(rag, query, top_k=top_k),
            "chunk_size_sensitivity": probe_chunk_size_sensitivity(rag, query, sizes=[600, 1200, 2400, 4000]),
        }
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval diagnostics for rag30")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output", default="diagnostics/diagnostic_single_query.json")
    parser.add_argument("--index-dir", default="faiss_indexes")
    parser.add_argument("--index-name", default="law_db")
    parser.add_argument("--data-dir", default="NPA3001")
    parser.add_argument("--llm-query-expansion", action="store_true")
    parser.add_argument("--log-file", default="logs/retrieval_diagnostic.log")
    args = parser.parse_args()

    setup_logger(Path(args.log_file))
    rag = OtusStyleRAG(
        index_dir=Path(args.index_dir),
        index_name=args.index_name,
        data_dir=Path(args.data_dir),
        use_llm_query_expansion=args.llm_query_expansion,
    )
    report = diagnose_query(rag, query=args.query, top_k=args.top_k)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Diagnostic report saved: %s", out_path)


if __name__ == "__main__":
    main()
