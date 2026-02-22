#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

from rag30 import OtusStyleRAG, canonical_law_codes_from_metadata, normalize_law_code, normalize_text, tokenize


def _doc_matches_hint(doc: Any, expected_hints: List[str]) -> bool:
    if not expected_hints:
        return True
    md = doc.metadata or {}
    if canonical_law_codes_from_metadata(md).intersection(expected_hints):
        return True
    law_id = normalize_law_code(str(md.get("law_id", "")))
    title = normalize_text(str(md.get("source_title", "")))
    hierarchy = normalize_text(str(md.get("hierarchy_str", "")))
    blob = f"{law_id} {title} {hierarchy}"
    for hint in expected_hints:
        if hint in law_id:
            return True
        if hint == "TK" and ("трудов" in blob or " тк " in f" {blob} "):
            return True
        if hint == "GK" and ("гражданск" in blob or " гк " in f" {blob} "):
            return True
        if hint == "NK" and ("налог" in blob or " нк " in f" {blob} "):
            return True
        if hint == "UK" and ("уголовн" in blob or " ук " in f" {blob} "):
            return True
        if hint == "UPK" and ("уголовно" in blob and "процесс" in blob):
            return True
        if hint == "APK" and ("арбитраж" in blob and "процесс" in blob):
            return True
        if hint == "GPK" and ("гражданск" in blob and "процесс" in blob):
            return True
        if hint == "KOAP" and ("коап" in blob or "административ" in blob):
            return True
        if hint == "SK" and ("семейн" in blob or " ск " in f" {blob} "):
            return True
        if hint == "JK" and ("жилищ" in blob or " жк " in f" {blob} "):
            return True
        if hint == "ZK" and ("земель" in blob or " зк " in f" {blob} "):
            return True
    return False


def _doc_keyword_score(doc: Any, keywords: List[str]) -> float:
    if not keywords:
        return 0.0
    md = doc.metadata or {}
    searchable = (
        f"{md.get('source_title', '')} "
        f"{md.get('hierarchy_str', '')} "
        f"{doc.page_content[:1600]}"
    )
    doc_tokens = tokenize(searchable)
    hits = 0
    for kw in keywords:
        kw_toks = tokenize(kw)
        if not kw_toks:
            continue
        if kw_toks.intersection(doc_tokens):
            hits += 1
    return hits / max(1, len(keywords))


def evaluate(
    rag: OtusStyleRAG,
    questions: List[Dict[str, Any]],
    top_k: int,
    disable_llm_variants: bool,
) -> Dict[str, Any]:
    if disable_llm_variants:
        rag._generate_query_variants = (  # type: ignore[method-assign]
            lambda q, max_variants=4, law_hints=None: [q]
        )

    per_question: List[Dict[str, Any]] = []
    by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for item in questions:
        docs, debug = rag.retrieve(item["question"], top_k=top_k)
        expected_hints = item.get("expected_law_hints", [])
        expected_keywords = item.get("expected_keywords", [])

        top1 = docs[0] if docs else None
        top1_law_hit = _doc_matches_hint(top1, expected_hints) if top1 else False
        law_hits = sum(1 for d in docs if _doc_matches_hint(d, expected_hints))
        law_precision = law_hits / max(1, len(docs))

        keyword_scores = [_doc_keyword_score(d, expected_keywords) for d in docs]
        top1_keyword = keyword_scores[0] if keyword_scores else 0.0
        best_keyword = max(keyword_scores) if keyword_scores else 0.0

        row = {
            "id": item["id"],
            "domain": item["domain"],
            "type": item["type"],
            "question": item["question"],
            "top1_law_hit": top1_law_hit,
            "law_hits": law_hits,
            "law_precision_at_k": round(law_precision, 4),
            "top1_keyword_score": round(top1_keyword, 4),
            "best_keyword_score_at_k": round(best_keyword, 4),
            "retrieved_docs": len(docs),
            "debug": {
                "law_hints": debug.law_hints,
                "explicit_law_hints": debug.explicit_law_hints,
                "inferred_law_hints": debug.inferred_law_hints,
                "entity_tokens": debug.entity_tokens,
                "article_hints": debug.article_hints,
                "query_variants": debug.query_variants,
                "law_docs": debug.law_docs,
                "practice_docs": debug.practice_docs,
                "lexical_law_docs": debug.lexical_law_docs,
            },
            "top_sources": [
                {
                    "rank": i + 1,
                    "law_id": (d.metadata or {}).get("law_id", ""),
                    "source_type": (d.metadata or {}).get("source_type", ""),
                    "title": (d.metadata or {}).get("source_title", ""),
                    "hierarchy": (d.metadata or {}).get("hierarchy_str", ""),
                }
                for i, d in enumerate(docs[:5])
            ],
        }
        per_question.append(row)
        by_domain[item["domain"]].append(row)

    domain_summary = {}
    for domain, rows in by_domain.items():
        domain_summary[domain] = {
            "count": len(rows),
            "top1_law_hit_rate": round(mean(1.0 if r["top1_law_hit"] else 0.0 for r in rows), 4),
            "avg_law_precision_at_k": round(mean(r["law_precision_at_k"] for r in rows), 4),
            "avg_top1_keyword_score": round(mean(r["top1_keyword_score"] for r in rows), 4),
            "avg_best_keyword_score_at_k": round(mean(r["best_keyword_score_at_k"] for r in rows), 4),
            "avg_practice_docs_in_debug": round(mean(r["debug"]["practice_docs"] for r in rows), 2),
        }

    overall = {
        "count": len(per_question),
        "top1_law_hit_rate": round(mean(1.0 if r["top1_law_hit"] else 0.0 for r in per_question), 4),
        "avg_law_precision_at_k": round(mean(r["law_precision_at_k"] for r in per_question), 4),
        "avg_top1_keyword_score": round(mean(r["top1_keyword_score"] for r in per_question), 4),
        "avg_best_keyword_score_at_k": round(mean(r["best_keyword_score_at_k"] for r in per_question), 4),
        "disable_llm_variants": disable_llm_variants,
        "top_k": top_k,
    }

    return {
        "overall": overall,
        "by_domain": domain_summary,
        "per_question": per_question,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rag30 retrieval quality")
    parser.add_argument("--questions", default="evaluation_questions_legal_ru.json")
    parser.add_argument("--output", default="rag30_eval_report.json")
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument("--disable-llm-variants", action="store_true")
    parser.add_argument("--index-dir", default="faiss_indexes")
    parser.add_argument("--index-name", default="law_db")
    parser.add_argument("--data-dir", default="NPA3001")
    args = parser.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    rag = OtusStyleRAG(
        index_dir=Path(args.index_dir),
        index_name=args.index_name,
        data_dir=Path(args.data_dir),
    )
    report = evaluate(
        rag=rag,
        questions=questions,
        top_k=args.top_k,
        disable_llm_variants=args.disable_llm_variants,
    )
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Overall:", report["overall"])
    print("Report:", args.output)


if __name__ == "__main__":
    main()
