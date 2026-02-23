#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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
    EntitySpec("Суды РФ", "Верховный Суд РФ", ("верховн", "суд"), ("фкз о верховном суде", "о судебной системе")),
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


def slugify(value: str, max_len: int = 80) -> str:
    value = value.strip().replace("/", " ").replace("\\", " ")
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^0-9A-Za-zА-Яа-яЁё_\-]+", "", value)
    value = value.strip("_-")
    return (value[:max_len] or "case").lower()


def build_cases() -> List[Tuple[EntitySpec, str]]:
    out: List[Tuple[EntitySpec, str]] = []
    for spec in ENTITY_SPECS:
        for tmpl in QUESTION_TEMPLATES:
            out.append((spec, tmpl.format(entity=spec.name)))
    return out


def build_final_query_for_llm(rag: OtusStyleRAG, question: str, docs: List[Any]) -> str:
    builder = getattr(rag, "build_answer_prompt", None)
    if callable(builder):
        return str(builder(question, docs))

    context_builder = getattr(rag, "_build_context", None)
    if callable(context_builder):
        context = context_builder(docs)
        return f"""
Работай только по переданному контексту.
Если в вопросе есть тест с вариантами — выбери только один правильный вариант и обоснуй нормой.
Всегда указывай ссылки на источники в формате [S1], [S2] и при возможности давай ссылку на consultant.ru.
Если контекста недостаточно — явно укажи это.

Структура ответа:
1) Краткий вывод.
2) Правовое обоснование с нормами и их логикой применения.
3) При необходимости — алгоритм действий/шаги.
4) Ограничения ответа (если есть).
5) Финальная фраза из системной инструкции — дословно.

Контекст:
{context}

Вопрос:
{question}
"""

    return question


def check_relevance(docs: List[Any], expected_tokens: Set[str]) -> bool:
    if not docs:
        return False
    if not expected_tokens:
        return True

    def _soft_overlap(exp: Set[str], got: Set[str]) -> float:
        if not exp or not got:
            return 0.0
        matched = 0
        for e in exp:
            if any(g == e or g.startswith(e) or e.startswith(g) for g in got):
                matched += 1
        return matched / max(1, len(exp))

    for d in docs[:7]:
        md = d.metadata or {}
        text = f"{md.get('source_title','')} {md.get('hierarchy_str','')} {d.page_content[:1800]}"
        doc_tokens = tokenize(text)
        if expected_tokens.intersection(doc_tokens):
            return True
        if _soft_overlap(expected_tokens, doc_tokens) >= 0.5:
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


def save_case_json(case_payload: Dict[str, Any], output_root: Path, case_index: int) -> Path:
    category = str(case_payload.get("category", "unknown"))
    entity = str(case_payload.get("entity", "unknown"))
    question = str(case_payload.get("question", ""))
    q_hash = hashlib.sha1(question.encode("utf-8", errors="ignore")).hexdigest()[:10]

    case_dir = output_root / slugify(category) / slugify(entity)
    case_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{case_index:04d}_{q_hash}.json"
    case_path = case_dir / file_name
    case_path.write_text(json.dumps(case_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return case_path


def run_case(
    rag: OtusStyleRAG,
    retriever: EnhancedLegalRetriever,
    question: str,
    entity: EntitySpec,
    top_k: int,
    run_answer: bool,
    case_index: int = 0,
    per_question_dir: Optional[Path] = None,
    diagnose_all: bool = False,
) -> Dict[str, Any]:
    docs, debug = retriever.search(question, top_k=top_k, use_hybrid=True, use_rerank=True, return_debug=True)

    all_found_documents = [serialize_doc(d, i + 1) for i, d in enumerate(docs)]
    top_sources = all_found_documents[:10]

    llm_invoked = False
    final_query_sent_to_llm = ""
    answer = ""

    if run_answer:
        if docs:
            final_query_sent_to_llm = build_final_query_for_llm(rag, question, docs)
            llm_invoked = True
        answer = rag.answer(question, docs)
        # Если rag.answer() обновил финальный prompt — берем фактически отправленный.
        last_prompt = getattr(rag, "last_llm_query", "")
        if isinstance(last_prompt, str) and last_prompt.strip():
            final_query_sent_to_llm = last_prompt

    expected_tokens = set(entity.tokens)
    relevance_ok = check_relevance(docs, expected_tokens)
    npa_links_ok = check_npa_references(answer) if run_answer else False
    completeness_ok = check_completeness(answer) if run_answer else False
    no_hallucinations = check_hallucinated_sources(answer, len(top_sources)) if run_answer else False
    npa_in_sources = check_npa_presence_in_sources(top_sources, entity.npa_markers)

    quality_checks = {
        "relevance_ok": relevance_ok,
        "npa_links_ok": npa_links_ok,
        "completeness_ok": completeness_ok,
        "no_hallucinations": no_hallucinations,
        "npa_in_sources": npa_in_sources,
        "expected_tokens": sorted(expected_tokens),
    }

    failed = not relevance_ok or (run_answer and (not npa_links_ok or not completeness_ok or not no_hallucinations))
    should_diagnose = diagnose_all or failed
    diagnostic = diagnose_query(rag, query=question, top_k=top_k, expected_tokens=expected_tokens) if should_diagnose else None

    pipeline_debug = {
        "top_k": top_k,
        "use_hybrid": True,
        "use_rerank": True,
        "pipeline": debug.get("pipeline", ""),
        "baseline": debug.get("baseline_debug", {}),
        "hybrid": {
            "baseline_count": debug.get("baseline_count", 0),
            "vector_count": debug.get("vector_count", 0),
            "fused_count": debug.get("fused_count", 0),
        },
        "rerank": {
            "enabled": "rerank" in str(debug.get("pipeline", "")).lower(),
            "final_count": debug.get("final_count", len(docs)),
        },
        "raw": debug,
    }

    result = {
        "question": question,
        "entity": entity.name,
        "category": entity.category,
        "top_sources": top_sources,
        "all_found_documents": all_found_documents,
        "final_query_sent_to_llm": final_query_sent_to_llm,
        "answer": answer,
        "pipeline_debug": pipeline_debug,
        "quality_checks": quality_checks,
        "relevance_ok": relevance_ok,
        "npa_links_ok": npa_links_ok,
        "completeness_ok": completeness_ok,
        "no_hallucinations": no_hallucinations,
        "npa_in_sources": npa_in_sources,
        "llm_invocation": {
            "run_answer": run_answer,
            "invoked": llm_invoked,
            "system_prompt": getattr(rag, "system_prompt", ""),
        },
        "diagnostic": diagnostic,
    }

    if per_question_dir is not None:
        case_path = save_case_json(result, output_root=per_question_dir, case_index=case_index)
        result["case_log_path"] = str(case_path)
        LOGGER.info("Saved case log: %s", case_path)

    return result


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_entity: Dict[str, List[Dict[str, Any]]] = {}
    for row in results:
        by_entity.setdefault(str(row["entity"]), []).append(row)

    entities_summary = []
    missing_acts = []
    for entity, rows in by_entity.items():
        relevance_rate = mean(1.0 if x.get("relevance_ok") else 0.0 for x in rows)
        completeness_rate = mean(1.0 if x.get("completeness_ok") else 0.0 for x in rows)
        npa_rate = mean(1.0 if x.get("npa_links_ok") else 0.0 for x in rows)
        hallucination_safe_rate = mean(1.0 if x.get("no_hallucinations") else 0.0 for x in rows)
        has_npa_sources = any(x.get("npa_in_sources") for x in rows)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Автоматическое тестирование RAG по судам и госорганам РФ")
    parser.add_argument("--output", default="diagnostics/orgs_rag_test_report.json")
    parser.add_argument("--per-question-dir", default="diagnostics/questions")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--with-answer", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--index-dir", default="faiss_indexes")
    parser.add_argument("--index-name", default="law_db")
    parser.add_argument("--data-dir", default="NPA3001")
    parser.add_argument("--llm-query-expansion", action="store_true")
    parser.add_argument("--diagnose-all", action="store_true")
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

    per_question_dir = Path(args.per_question_dir)
    per_question_dir.mkdir(parents=True, exist_ok=True)

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
            case_index=idx,
            per_question_dir=per_question_dir,
            diagnose_all=args.diagnose_all,
        )

        LOGGER.info(
            "Retrieved docs=%s | relevance_ok=%s | completeness_ok=%s | npa_links_ok=%s",
            len(case_result.get("all_found_documents", [])),
            case_result.get("relevance_ok"),
            case_result.get("completeness_ok"),
            case_result.get("npa_links_ok"),
        )

        for src in case_result.get("all_found_documents", []):
            LOGGER.info(
                "doc#%s | title=%s | law_id=%s | hierarchy=%s",
                src.get("rank"),
                src.get("title"),
                src.get("law_id"),
                src.get("hierarchy"),
            )

        results.append(case_result)

    report = {
        "config": {
            "top_k": args.top_k,
            "with_answer": args.with_answer,
            "llm_query_expansion": args.llm_query_expansion,
            "diagnose_all": args.diagnose_all,
            "cases": len(cases),
            "per_question_dir": str(per_question_dir),
        },
        "summary": summarize(results),
        "results": results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Saved aggregate report: %s", out_path)


if __name__ == "__main__":
    main()
