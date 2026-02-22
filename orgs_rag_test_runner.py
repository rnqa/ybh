#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Set, Tuple

from rag30 import OtusStyleRAG, normalize_text, tokenize
from retrieval_diagnostic import diagnose_query, serialize_doc
from retriever_enhanced import EnhancedLegalRetriever


LOGGER = logging.getLogger("orgs_rag_test_runner")


QUESTION_TEMPLATES = [
    "Дайте определение {entity}.",
    "Какова правовая основа деятельности {entity}? Укажите НПА.",
    "Какие полномочия у {entity}?",
    "Какова структура {entity}?",
    "Какое место {entity} занимает в системе органов государственной власти РФ?",
    "Каков порядок формирования {entity}?",
    "Кому подотчетен(а) {entity}?",
]


@dataclass
class EntitySpec:
    category: str
    name: str
    tokens: Tuple[str, ...]
    npa_markers: Tuple[str, ...]


ENTITY_SPECS: List[EntitySpec] = [
    EntitySpec("Суды РФ", "Верховный Суд РФ", ("верховн", "суд"), ("фкз о верховном суде", "судебной системе")),
    EntitySpec("Суды РФ", "Конституционный Суд РФ", ("конституцион", "суд"), ("фкз о конституционном суде",)),
    EntitySpec("Суды РФ", "Районный суд", ("район", "суд"), ("о судах общей юрисдикции", "гпк", "упк")),
    EntitySpec("Суды РФ", "Мировые судьи", ("миров", "суд"), ("о мировых судьях",)),
    EntitySpec("Суды РФ", "Арбитражные суды", ("арбитраж", "суд"), ("об арбитражных судах", "апк")),
    EntitySpec("Суды РФ", "Апелляционные суды общей юрисдикции", ("апелляцион", "юрисдикц"), ("о судах общей юрисдикции",)),
    EntitySpec("Суды РФ", "Кассационные суды общей юрисдикции", ("кассацион", "юрисдикц"), ("о судах общей юрисдикции",)),
    EntitySpec("Госорганы РФ", "МВД РФ", ("мвд", "внутрен", "дел"), ("о полиции", "мвд")),
    EntitySpec("Госорганы РФ", "Следственный комитет РФ", ("следствен", "комитет"), ("о ск рф", "следствен")),
    EntitySpec("Госорганы РФ", "ФСБ РФ", ("фсб", "безопасн"), ("о фсб",)),
    EntitySpec("Госорганы РФ", "СВР РФ", ("свр", "разведк"), ("о внешней разведке", "свр")),
    EntitySpec("Госорганы РФ", "Прокуратура РФ", ("прокуратур",), ("о прокуратуре",)),
    EntitySpec("Госорганы РФ", "ФСИН РФ", ("фсин", "исполнен", "наказан"), ("уик", "фсин")),
    EntitySpec("Госорганы РФ", "Росгвардия", ("росгвард", "гвард"), ("о войсках национальной гвардии", "росгвард")),
]


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


def check_relevance(docs: List[Any], expected_tokens: Set[str]) -> bool:
    if not docs:
        return False
    if not expected_tokens:
        return True
    for d in docs[:7]:
        md = d.metadata or {}
        text = f"{md.get('source_title','')} {md.get('hierarchy_str','')} {d.page_content[:1800]}"
        if expected_tokens.intersection(tokenize(text)):
            return True
    return False


def check_npa_references(answer: str) -> bool:
    a = normalize_text(answer)
    has_norm_term = any(
        marker in a
        for marker in ("ст.", "статья", "фз", "фкз", "кодекс", "конституц", "постановлен", "пленум")
    )
    has_source_citation = bool(re.search(r"\[s\d+\]", a))
    return has_norm_term and has_source_citation


def check_completeness(answer: str) -> bool:
    a = normalize_text(answer)
    has_definition = any(x in a for x in ("определен", "это", "представля", "понима"))
    has_powers = any(x in a for x in ("полномоч", "компетенц", "функц"))
    has_structure = any(x in a for x in ("структур", "состав", "систем"))
    return has_definition and has_powers and has_structure


def check_hallucinated_sources(answer: str, sources_count: int) -> bool:
    refs = re.findall(r"\[S(\d+)\]", answer, flags=re.IGNORECASE)
    if not refs:
        return False
    ref_nums = {int(x) for x in refs}
    return all(1 <= n <= sources_count for n in ref_nums)


def check_npa_presence_in_sources(sources: List[Dict[str, Any]], markers: Tuple[str, ...]) -> bool:
    if not sources:
        return False
    blob = normalize_text(" ".join(f"{s.get('title','')} {s.get('hierarchy','')} {s.get('source_type','')}" for s in sources))
    return any(m in blob for m in markers)


def run_case(
    rag: OtusStyleRAG,
    retriever: EnhancedLegalRetriever,
    question: str,
    entity: EntitySpec,
    top_k: int,
    run_answer: bool,
) -> Dict[str, Any]:
    docs, debug = retriever.search(question, top_k=top_k, use_hybrid=True, use_rerank=True, return_debug=True)
    sources = [
        {
            "id": i + 1,
            "title": (d.metadata or {}).get("source_title", ""),
            "law_id": (d.metadata or {}).get("law_id", ""),
            "source_type": (d.metadata or {}).get("source_type", ""),
            "hierarchy": (d.metadata or {}).get("hierarchy_str", ""),
            "source_url": (d.metadata or {}).get("source_url", ""),
        }
        for i, d in enumerate(docs)
    ]

    answer = ""
    if run_answer:
        answer = rag.answer(question, docs)

    expected_tokens = set(entity.tokens)
    relevance_ok = check_relevance(docs, expected_tokens)
    npa_links_ok = check_npa_references(answer) if run_answer else False
    completeness_ok = check_completeness(answer) if run_answer else False
    no_hallucinations = check_hallucinated_sources(answer, len(sources)) if run_answer else False
    npa_in_sources = check_npa_presence_in_sources(sources, entity.npa_markers)

    failed = not relevance_ok or (run_answer and (not npa_links_ok or not completeness_ok or not no_hallucinations))
    diagnostic = None
    if failed:
        diagnostic = diagnose_query(rag, query=question, top_k=top_k, expected_tokens=expected_tokens)

    return {
        "question": question,
        "entity": entity.name,
        "category": entity.category,
        "relevance_ok": relevance_ok,
        "npa_links_ok": npa_links_ok,
        "completeness_ok": completeness_ok,
        "no_hallucinations": no_hallucinations,
        "npa_in_sources": npa_in_sources,
        "pipeline_debug": debug,
        "top_sources": [serialize_doc(d, i + 1) for i, d in enumerate(docs[:10])],
        "answer": answer,
        "diagnostic": diagnostic,
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_entity: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        by_entity.setdefault(r["entity"], []).append(r)

    entities_summary = []
    missing_acts = []
    for entity, rows in by_entity.items():
        relevance_rate = mean(1.0 if x["relevance_ok"] else 0.0 for x in rows)
        completeness_rate = mean(1.0 if x["completeness_ok"] else 0.0 for x in rows)
        npa_rate = mean(1.0 if x["npa_links_ok"] else 0.0 for x in rows)
        hallucination_safe_rate = mean(1.0 if x["no_hallucinations"] else 0.0 for x in rows)
        has_npa_sources = any(x["npa_in_sources"] for x in rows)
        if not has_npa_sources:
            missing_acts.append(entity)
        entities_summary.append(
            {
                "entity": entity,
                "cases": len(rows),
                "relevance_rate": round(relevance_rate, 4),
                "completeness_rate": round(completeness_rate, 4),
                "npa_reference_rate": round(npa_rate, 4),
                "no_hallucination_rate": round(hallucination_safe_rate, 4),
                "likely_missing_normative_act": not has_npa_sources,
            }
        )

    worst_by_relevance = sorted(entities_summary, key=lambda x: x["relevance_rate"])[:5]
    return {
        "total_cases": len(results),
        "entities": entities_summary,
        "worst_entities_by_relevance": worst_by_relevance,
        "likely_missing_acts": missing_acts,
    }


def build_cases() -> List[Tuple[EntitySpec, str]]:
    out = []
    for spec in ENTITY_SPECS:
        for tmpl in QUESTION_TEMPLATES:
            out.append((spec, tmpl.format(entity=spec.name)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Автоматическое тестирование RAG по судам и госорганам РФ")
    parser.add_argument("--output", default="diagnostics/orgs_rag_test_report.json")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--with-answer", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--index-dir", default="faiss_indexes")
    parser.add_argument("--index-name", default="law_db")
    parser.add_argument("--data-dir", default="NPA3001")
    parser.add_argument("--llm-query-expansion", action="store_true")
    parser.add_argument("--log-file", default="logs/orgs_rag_test_runner.log")
    args = parser.parse_args()

    setup_logger(Path(args.log_file))
    rag = OtusStyleRAG(
        index_dir=Path(args.index_dir),
        index_name=args.index_name,
        data_dir=Path(args.data_dir),
        use_llm_query_expansion=args.llm_query_expansion,
    )
    retriever = EnhancedLegalRetriever(rag)

    cases = build_cases()
    if args.limit > 0:
        cases = cases[: args.limit]

    results: List[Dict[str, Any]] = []
    for idx, (entity, question) in enumerate(cases, start=1):
        LOGGER.info("[%s/%s] %s | %s", idx, len(cases), entity.name, question)
        case_result = run_case(
            rag=rag,
            retriever=retriever,
            question=question,
            entity=entity,
            top_k=args.top_k,
            run_answer=args.with_answer,
        )
        results.append(case_result)

    report = {
        "config": {
            "top_k": args.top_k,
            "with_answer": args.with_answer,
            "llm_query_expansion": args.llm_query_expansion,
            "cases": len(cases),
        },
        "summary": summarize(results),
        "results": results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Saved report: %s", out_path)


if __name__ == "__main__":
    main()

