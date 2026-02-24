#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from rag30 import OtusStyleRAG, normalize_text, tokenize
from retrieval_diagnostic import diagnose_query, serialize_doc


LOGGER = logging.getLogger("full_rag_test_runner")


QUESTION_TEMPLATES = [
    "Дайте определение {entity}.",
    "Какова правовая основа деятельности {entity}? Укажите НПА.",
    "Какие полномочия у {entity}?",
    "Какова структура {entity}?",
    "Какое место {entity} занимает в системе органов государственной власти РФ?",
    "Каков порядок формирования {entity}?",
    "Кому подотчетен(а) {entity}?",
]


@dataclass(frozen=True)
class EntitySpec:
    category: str
    name: str
    tokens: Tuple[str, ...]
    npa_markers: Tuple[str, ...]


@dataclass(frozen=True)
class BranchSpec:
    name: str
    tokens: Tuple[str, ...]
    npa_markers: Tuple[str, ...]
    theory_question: str
    case_question: str


ENTITY_SPECS: List[EntitySpec] = [
    EntitySpec("Суды РФ", "Верховный Суд РФ", ("верховн", "суд"), ("фкз о верховном суде", "о судебной системе")),
    EntitySpec("Суды РФ", "Конституционный Суд РФ", ("конституцион", "суд"), ("фкз о конституционном суде",)),
    EntitySpec("Суды РФ", "Районный суд", ("район", "суд"), ("о судах общей юрисдикции", "гпк", "упк")),
    EntitySpec("Суды РФ", "Мировые судьи", ("миров", "суд"), ("о мировых судьях",)),
    EntitySpec("Суды РФ", "Арбитражные суды", ("арбитраж", "суд"), ("об арбитражных судах", "апк")),
    EntitySpec(
        "Суды РФ",
        "Апелляционные суды общей юрисдикции",
        ("апелляцион", "юрисдикц"),
        ("о судах общей юрисдикции",),
    ),
    EntitySpec(
        "Суды РФ",
        "Кассационные суды общей юрисдикции",
        ("кассацион", "юрисдикц"),
        ("о судах общей юрисдикции",),
    ),
    EntitySpec("Госорганы РФ", "МВД РФ", ("мвд", "внутрен", "дел"), ("о полиции", "мвд")),
    EntitySpec("Госорганы РФ", "Следственный комитет РФ", ("следствен", "комитет"), ("о следственном комитете", "ск рф")),
    EntitySpec("Госорганы РФ", "ФСБ РФ", ("фсб", "безопасн"), ("о фсб", "фсб")),
    EntitySpec("Госорганы РФ", "СВР РФ", ("свр", "разведк"), ("о внешней разведке", "свр")),
    EntitySpec("Госорганы РФ", "Прокуратура РФ", ("прокуратур",), ("о прокуратуре",)),
    EntitySpec(
        "Госорганы РФ",
        "ФСИН РФ",
        ("фсин", "исполн", "наказан"),
        ("уголовно-исполнительной системе", "уик", "фсин"),
    ),
    EntitySpec(
        "Госорганы РФ",
        "Росгвардия",
        ("росгвард", "гвард"),
        ("о войсках национальной гвардии", "росгвард"),
    ),
]


BRANCH_SPECS: List[BranchSpec] = [
    BranchSpec(
        name="Уголовное право",
        tokens=("уголовн", "преступ"),
        npa_markers=("ук рф",),
        theory_question=(
            "Что такое уголовное право и какие основные источники его регулируют? "
            "Ответьте с опорой на НПА."
        ),
        case_question=(
            "Гражданин тайно похитил телефон из магазина. Дайте юридическую квалификацию и "
            "укажите применимые нормы НПА."
        ),
    ),
    BranchSpec(
        name="Гражданское право",
        tokens=("граждан", "договор", "обязат"),
        npa_markers=("гк рф",),
        theory_question=(
            "Дайте понятие гражданского права и назовите ключевые источники регулирования. "
            "Ответьте по НПА."
        ),
        case_question=(
            "Организация не исполнила договор поставки и требует неустойку. "
            "Какие нормы НПА применимы?"
        ),
    ),
    BranchSpec(
        name="Административное право",
        tokens=("административ", "правонаруш"),
        npa_markers=("коап",),
        theory_question=(
            "Что такое административное право и на каких НПА оно основано?"
        ),
        case_question=(
            "Водитель превысил скорость и был привлечен к ответственности. "
            "Какие нормы НПА применимы?"
        ),
    ),
    BranchSpec(
        name="Трудовое право",
        tokens=("труд", "работник", "работодатель"),
        npa_markers=("тк рф",),
        theory_question=(
            "Определите предмет трудового права и укажите основные источники по НПА."
        ),
        case_question=(
            "Работнику задержали заработную плату на месяц. "
            "Какие нормы НПА регулируют ответственность работодателя?"
        ),
    ),
    BranchSpec(
        name="Семейное право",
        tokens=("семейн", "брак", "алимент"),
        npa_markers=("ск рф",),
        theory_question=(
            "Что регулирует семейное право и какие НПА являются ключевыми?"
        ),
        case_question=(
            "Родитель уклоняется от уплаты алиментов. "
            "Какие нормы НПА применимы?"
        ),
    ),
    BranchSpec(
        name="Жилищное право",
        tokens=("жилищ", "квартира", "собственник"),
        npa_markers=("жк рф",),
        theory_question=(
            "Определите жилищное право и укажите основные НПА, которые его регулируют."
        ),
        case_question=(
            "Наниматель систематически нарушает правила пользования жилым помещением. "
            "Какие нормы НПА применимы?"
        ),
    ),
    BranchSpec(
        name="Налоговое право",
        tokens=("налог", "сбор", "декларац"),
        npa_markers=("нк рф",),
        theory_question=(
            "Что такое налоговое право и какие НПА лежат в его основе?"
        ),
        case_question=(
            "Организация просрочила уплату налога. Какие нормы НПА применимы?"
        ),
    ),
    BranchSpec(
        name="Арбитражный процесс",
        tokens=("арбитраж", "процесс", "иск"),
        npa_markers=("апк рф",),
        theory_question=(
            "Определите арбитражный процесс и укажите основные НПА регулирования."
        ),
        case_question=(
            "Компания хочет обжаловать решение арбитражного суда. "
            "Какие нормы НПА регулируют порядок обжалования?"
        ),
    ),
    BranchSpec(
        name="Финансовое право",
        tokens=("финансов", "бюджет", "казна"),
        npa_markers=("бк рф",),
        theory_question=(
            "Что регулирует финансовое право и какие НПА являются базовыми?"
        ),
        case_question=(
            "Государственное учреждение использовало бюджетные средства не по назначению. "
            "Какие нормы НПА применимы?"
        ),
    ),
    BranchSpec(
        name="Конституционное право",
        tokens=("конституцион", "права", "свобод"),
        npa_markers=("конституц",),
        theory_question=(
            "Дайте понятие конституционного права и укажите ключевые НПА."
        ),
        case_question=(
            "Гражданин считает, что нарушено его конституционное право на свободу слова. "
            "Какие нормы НПА применимы?"
        ),
    ),
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


def build_org_cases() -> List[Tuple[EntitySpec, str]]:
    out: List[Tuple[EntitySpec, str]] = []
    for spec in ENTITY_SPECS:
        for tmpl in QUESTION_TEMPLATES:
            out.append((spec, tmpl.format(entity=spec.name)))
    return out


def build_branch_cases() -> List[Tuple[BranchSpec, str, str]]:
    out: List[Tuple[BranchSpec, str, str]] = []
    for spec in BRANCH_SPECS:
        out.append((spec, "theory", spec.theory_question))
        out.append((spec, "case", spec.case_question))
    return out


def is_textbook_doc(md: Dict[str, Any]) -> bool:
    blob = normalize_text(f"{md.get('source_type','')} {md.get('source_title','')} {md.get('law_id','')}")
    return "учеб" in blob or blob.endswith("_uch")


def filter_docs_by_mode(docs: Sequence[Any], mode: str) -> List[Any]:
    if mode == "textbook":
        return [d for d in docs if is_textbook_doc(d.metadata or {})]
    if mode == "npa":
        return [d for d in docs if not is_textbook_doc(d.metadata or {})]
    return list(docs)


def build_retrieval_query(question: str, mode: str) -> str:
    if mode == "textbook":
        return f"{question} учебник"
    if mode == "npa":
        return f"{question} НПА"
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
        text = f"{md.get('source_title','')} {md.get('hierarchy_str','')} {d.page_content[:1600]}"
        doc_tokens = tokenize(text)
        if expected_tokens.intersection(doc_tokens):
            return True
        if _soft_overlap(expected_tokens, doc_tokens) >= 0.5:
            return True
    return False


def check_source_mode(sources: List[Dict[str, Any]], mode: str) -> bool:
    if not sources:
        return False
    if mode == "textbook":
        return any(is_textbook_doc({"source_type": s.get("source_type", ""), "source_title": s.get("title", ""), "law_id": s.get("law_id", "")}) for s in sources)
    if mode == "npa":
        return any(not is_textbook_doc({"source_type": s.get("source_type", ""), "source_title": s.get("title", ""), "law_id": s.get("law_id", "")}) for s in sources)
    return True


def check_npa_references(answer: str) -> bool:
    a = normalize_text(answer)
    has_norm_term = any(
        marker in a
        for marker in ("ст.", "статья", "фз", "фкз", "кодекс", "конституц", "постановлен", "пленум")
    )
    has_source_citation = bool(re.search(r"\[s\d+\]", a))
    return has_norm_term and has_source_citation


def check_hallucinated_sources(answer: str, sources_count: int) -> bool:
    refs = re.findall(r"\[S(\d+)\]", answer, flags=re.IGNORECASE)
    if not refs:
        return False
    ref_nums = {int(x) for x in refs}
    return all(1 <= n <= sources_count for n in ref_nums)


def save_case_json(case_payload: Dict[str, Any], output_root: Path, case_index: int) -> Path:
    category = str(case_payload.get("category", "unknown"))
    entity = str(case_payload.get("entity", "unknown"))
    mode = str(case_payload.get("source_mode", "unknown"))
    q_hash = hashlib.sha1(case_payload["question"].encode("utf-8", errors="ignore")).hexdigest()[:10]

    case_dir = output_root / slugify(mode) / slugify(category) / slugify(entity)
    case_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{case_index:04d}_{q_hash}.json"
    case_path = case_dir / file_name
    case_path.write_text(json.dumps(case_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return case_path


def safe_diagnose_query(
    rag: OtusStyleRAG,
    query: str,
    top_k: int,
    expected_tokens: Set[str],
) -> Optional[Dict[str, Any]]:
    try:
        return diagnose_query(rag, query=query, top_k=top_k, expected_tokens=expected_tokens)
    except Exception as exc:  # pragma: no cover
        return {"error": f"diagnose_query_failed: {exc}"}


def run_case(
    rag: OtusStyleRAG,
    question: str,
    retrieval_query: str,
    category: str,
    entity: str,
    expected_tokens: Set[str],
    npa_markers: Tuple[str, ...],
    mode: str,
    top_k: int,
    run_answer: bool,
    case_index: int,
    per_question_dir: Path,
    diagnose_on_fail: bool,
) -> Dict[str, Any]:
    docs_all, debug = rag.retrieve(retrieval_query, top_k=top_k)
    docs = filter_docs_by_mode(docs_all, mode)

    answer = ""
    if run_answer:
        answer = rag.answer(question, docs)

    sources = [serialize_doc(doc, rank=i) for i, doc in enumerate(docs, start=1)]
    sources_all = [serialize_doc(doc, rank=i) for i, doc in enumerate(docs_all, start=1)]

    relevance_ok = check_relevance(docs, expected_tokens)
    source_mode_ok = check_source_mode(sources, mode)
    npa_refs_ok = check_npa_references(answer) if answer else False
    hallucination_ok = check_hallucinated_sources(answer, len(sources)) if answer else False

    diagnostic = None
    if diagnose_on_fail and (not docs or not relevance_ok or not source_mode_ok):
        diagnostic = safe_diagnose_query(
            rag, query=retrieval_query, top_k=top_k, expected_tokens=expected_tokens
        )

    payload = {
        "question": question,
        "retrieval_query": retrieval_query,
        "category": category,
        "entity": entity,
        "source_mode": mode,
        "answer": answer,
        "sources": sources,
        "sources_all": sources_all,
        "debug": {
            "query_variants": debug.query_variants,
            "law_hints": debug.law_hints,
            "article_hints": debug.article_hints,
            "law_docs": debug.law_docs,
            "practice_docs": debug.practice_docs,
            "lexical_law_docs": debug.lexical_law_docs,
            "final_docs": debug.final_docs,
        },
        "checks": {
            "relevance_ok": relevance_ok,
            "source_mode_ok": source_mode_ok,
            "npa_refs_ok": npa_refs_ok,
            "hallucination_ok": hallucination_ok,
        },
        "expected_tokens": sorted(expected_tokens),
        "npa_markers": list(npa_markers),
        "diagnostic": diagnostic,
    }
    save_case_json(payload, per_question_dir, case_index)
    return payload


def summarize(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(cases)
    if not total:
        return {}

    def rate(key: str) -> float:
        return round(100 * sum(1 for c in cases if c["checks"].get(key)) / total, 2)

    by_category: Dict[str, List[Dict[str, Any]]] = {}
    for c in cases:
        by_category.setdefault(c["category"], []).append(c)

    category_stats = {
        k: {
            "count": len(v),
            "relevance_ok_rate": rate_for(v, "relevance_ok"),
            "source_mode_ok_rate": rate_for(v, "source_mode_ok"),
            "npa_refs_ok_rate": rate_for(v, "npa_refs_ok"),
            "hallucination_ok_rate": rate_for(v, "hallucination_ok"),
        }
        for k, v in by_category.items()
    }

    return {
        "total": total,
        "relevance_ok_rate": rate("relevance_ok"),
        "source_mode_ok_rate": rate("source_mode_ok"),
        "npa_refs_ok_rate": rate("npa_refs_ok"),
        "hallucination_ok_rate": rate("hallucination_ok"),
        "category_stats": category_stats,
    }


def rate_for(cases: List[Dict[str, Any]], key: str) -> float:
    if not cases:
        return 0.0
    return round(100 * sum(1 for c in cases if c["checks"].get(key)) / len(cases), 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Полный тест RAG: суды/ведомства + отрасли права")
    parser.add_argument("--index-dir", default="faiss_indexes")
    parser.add_argument("--index-name", default="law_db")
    parser.add_argument("--data-dir", default="NPA3001")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output", default="diagnostics/full_rag_test_report.json")
    parser.add_argument("--per-question-dir", default="diagnostics/full_test")
    parser.add_argument("--log-file", default="logs/full_rag_test_runner.log")
    parser.add_argument("--no-answer", action="store_true")
    parser.add_argument("--diagnose-on-fail", action="store_true")
    parser.add_argument("--start-case", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    setup_logger(Path(args.log_file))

    per_question_dir = Path(args.per_question_dir)
    per_question_dir.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        all_cases: List[Dict[str, Any]] = []
        for path in per_question_dir.rglob("*.json"):
            try:
                all_cases.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        report = {
            "total_cases": len(all_cases),
            "duration_sec": 0.0,
            "summary": summarize(all_cases),
            "cases": all_cases,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("Saved report to %s", output_path)
        LOGGER.info("Total cases: %s", len(all_cases))
        return

    rag = OtusStyleRAG(
        index_dir=Path(args.index_dir),
        index_name=args.index_name,
        data_dir=Path(args.data_dir),
    )

    cases: List[Dict[str, Any]] = []
    case_index = 0
    start_ts = time.time()

    org_cases = build_org_cases()
    for spec, question in org_cases:
        expected_tokens = {normalize_text(t) for t in spec.tokens}
        for mode in ("npa", "textbook"):
            case_index += 1
            if case_index < args.start_case:
                continue
            if args.max_cases and len(cases) >= args.max_cases:
                break
            retrieval_query = build_retrieval_query(question, mode)
            cases.append(
                run_case(
                    rag=rag,
                    question=question,
                    retrieval_query=retrieval_query,
                    category=spec.category,
                    entity=spec.name,
                    expected_tokens=expected_tokens,
                    npa_markers=spec.npa_markers,
                    mode=mode,
                    top_k=args.top_k,
                    run_answer=not args.no_answer,
                    case_index=case_index,
                    per_question_dir=per_question_dir,
                    diagnose_on_fail=args.diagnose_on_fail,
                )
            )
        if args.max_cases and len(cases) >= args.max_cases:
            break

    branch_cases = build_branch_cases()
    for spec, q_type, question in branch_cases:
        expected_tokens = {normalize_text(t) for t in spec.tokens}
        for mode in ("npa", "textbook"):
            case_index += 1
            if case_index < args.start_case:
                continue
            if args.max_cases and len(cases) >= args.max_cases:
                break
            retrieval_query = build_retrieval_query(question, mode)
            cases.append(
                run_case(
                    rag=rag,
                    question=question,
                    retrieval_query=retrieval_query,
                    category="Отрасли права",
                    entity=f"{spec.name} ({q_type})",
                    expected_tokens=expected_tokens,
                    npa_markers=spec.npa_markers,
                    mode=mode,
                    top_k=args.top_k,
                    run_answer=not args.no_answer,
                    case_index=case_index,
                    per_question_dir=per_question_dir,
                    diagnose_on_fail=args.diagnose_on_fail,
                )
            )
        if args.max_cases and len(cases) >= args.max_cases:
            break

    duration = round(time.time() - start_ts, 2)
    report = {
        "total_cases": len(cases),
        "duration_sec": duration,
        "summary": summarize(cases),
        "cases": cases,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    LOGGER.info("Saved report to %s", output_path)
    LOGGER.info("Total cases: %s", len(cases))
    LOGGER.info("Duration: %.2fs", duration)


if __name__ == "__main__":
    main()
