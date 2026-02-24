#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import heapq
import importlib.util
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from dotenv import load_dotenv

IMPORT_ERRORS: Dict[str, Exception] = {}

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.messages import HumanMessage, SystemMessage
except Exception as exc:  # pragma: no cover
    IMPORT_ERRORS["langchain-core"] = exc
    Document = object  # type: ignore[assignment]
    Embeddings = object  # type: ignore[assignment]
    HumanMessage = object  # type: ignore[assignment]
    SystemMessage = object  # type: ignore[assignment]

try:
    from langchain_community.vectorstores import FAISS
except Exception as exc:  # pragma: no cover
    IMPORT_ERRORS["langchain-community"] = exc
    FAISS = None  # type: ignore[assignment]

try:
    from langchain_openai import ChatOpenAI
except Exception as exc:  # pragma: no cover
    IMPORT_ERRORS["langchain-openai"] = exc
    ChatOpenAI = None  # type: ignore[assignment]

try:
    from aiogram import Bot, Dispatcher, types
    from aiogram.filters import CommandStart
except Exception as exc:  # pragma: no cover
    IMPORT_ERRORS["aiogram"] = exc
    Bot = None  # type: ignore[assignment]
    Dispatcher = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]
    CommandStart = None  # type: ignore[assignment]

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except Exception as exc:  # pragma: no cover
    IMPORT_ERRORS["sentence-transformers"] = exc
    SentenceTransformer = None  # type: ignore[assignment]
    CrossEncoder = None  # type: ignore[assignment]

try:
    from rank_bm25 import BM25Okapi
except Exception as exc:  # pragma: no cover
    IMPORT_ERRORS["rank-bm25"] = exc
    BM25Okapi = None  # type: ignore[assignment]


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("rag30.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("rag30")


# Конфигурация retrieval (по статье OTUS/Habr).
LAW_WEIGHT = 0.70
PRACTICE_WEIGHT = 0.30
RRF_K = 60
VECTOR_FETCH_K = 120
BM25_FETCH_K = 120
RERANK_CANDIDATES = 80
RERANK_TOP_K = 8
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

LAW_K = 20
LAW_FETCH_K = 80
LAW_LAMBDA = 0.20

PRACTICE_K = 8
PRACTICE_FETCH_K = 40
PRACTICE_LAMBDA = 0.80

TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]{3,}")
ARTICLE_RE = re.compile(r"(?:ст\.?|статья)\s*(\d+(?:\.\d+)?)", flags=re.IGNORECASE)

LAW_SOURCE_MARKERS = (
    "кодекс",
    "федеральный закон",
    "федеральный конституционный закон",
    "конституция",
)
PRACTICE_SOURCE_MARKERS = (
    "постановление пленума",
    "обзор",
    "судебной практик",
    "определение",
    "решение",
    "приговор",
)

LAW_HINT_PATTERNS: Dict[str, List[str]] = {
    "TK": [r"(?<!\w)тк(?!\w)", r"трудов"],
    "GK": [r"(?<!\w)гк(?!\w)", r"гражданск"],
    "NK": [r"(?<!\w)нк(?!\w)", r"налог"],
    "UK": [r"(?<!\w)ук\s*рф(?!\w)", r"уголовн"],
    "UPK": [r"(?<!\w)упк(?!\w)", r"уголов.*процесс"],
    "GPK": [r"(?<!\w)гпк(?!\w)", r"граждан.*процесс"],
    "APK": [r"(?<!\w)апк(?!\w)", r"арбитраж.*процесс"],
    "KOAP": [r"(?<!\w)коап(?!\w)", r"административн"],
    "SK": [r"(?<!\w)ск(?!\w)", r"семейн"],
    "JK": [r"(?<!\w)жк(?!\w)", r"жилищ"],
    "ZK": [r"(?<!\w)зк(?!\w)", r"земельн"],
    "BK": [r"(?<!\w)бк(?!\w)", r"бюджет"],
}

LAW_HINT_TOKENS: Dict[str, Tuple[str, ...]] = {
    "TK": ("трудов", "трудовой", "тк"),
    "GK": ("граждан", "гк"),
    "NK": ("налог", "нк"),
    "UK": ("уголов", "ук"),
    "UPK": ("уголовно", "процесс", "упк"),
    "GPK": ("гражданско", "процесс", "гпк"),
    "APK": ("арбитраж", "процесс", "апк"),
    "KOAP": ("административ", "коап"),
}

TEXTBOOK_MARKERS = (
    "учебник",
    "учебное пособие",
    "курс лекций",
    "учеб",
)

EDU_INTENT_MARKERS = (
    "объясни",
    "простыми словами",
    "для экзамена",
    "для семинара",
    "конспект",
    "теория",
    "учебник",
    "учебное пособие",
)

NORMATIVE_INTENT_MARKERS = (
    "статья",
    "ст.",
    "кодекс",
    "фз",
    "фкз",
    "конституц",
    "норма",
    "пункт",
    "часть",
    "постановлен",
    "пленум",
)

LAW_HINT_TITLE_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "TK": (r"трудов(ой|ого)?\s+кодекс", r"\bтк\b"),
    "GK": (r"гражданск(ий|ого)?\s+кодекс", r"\bгк\b"),
    "NK": (r"налогов(ый|ого)?\s+кодекс", r"\bнк\b"),
    "UK": (r"уголовн(ый|ого)?\s+кодекс", r"\bук\b"),
    "UPK": (r"уголовно[-\s]?процессуальн", r"\bупк\b"),
    "GPK": (r"гражданско[-\s]?процессуальн", r"\bгпк\b"),
    "APK": (r"арбитражно[-\s]?процессуальн", r"\bапк\b"),
    "KOAP": (r"административн(ых|ого)?\s+правонаруш", r"\bкоап\b"),
    "SK": (r"семейн(ый|ого)?\s+кодекс", r"\bск\b"),
    "JK": (r"жилищн(ый|ого)?\s+кодекс", r"\bжк\b"),
    "ZK": (r"земельн(ый|ого)?\s+кодекс", r"\bзк\b"),
    "BK": (r"бюджетн(ый|ого)?\s+кодекс", r"\bбк\b"),
}

LAW_CANON_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "TK": (r"трудов(ый|ого)?\s+кодекс", r"(?<!\w)тк(?!\w)"),
    "GK": (r"гражданск(ий|ого)?\s+кодекс", r"(?<!\w)гк(?!\w)"),
    "NK": (r"налогов(ый|ого)?\s+кодекс", r"(?<!\w)нк(?!\w)"),
    "UK": (r"уголовн(ый|ого)?\s+кодекс", r"(?<!\w)ук(?!\w)"),
    "UPK": (r"уголовно[-\s]?процессуальн", r"(?<!\w)упк(?!\w)"),
    "GPK": (r"гражданско[-\s]?процессуальн", r"(?<!\w)гпк(?!\w)"),
    "APK": (r"арбитражно[-\s]?процессуальн", r"(?<!\w)апк(?!\w)"),
    "KOAP": (r"административн(ых|ого)?\s+правонаруш", r"(?<!\w)коап(?!\w)"),
    "SK": (r"семейн(ый|ого)?\s+кодекс", r"(?<!\w)ск(?!\w)"),
    "JK": (r"жилищн(ый|ого)?\s+кодекс", r"(?<!\w)жк(?!\w)"),
    "ZK": (r"земельн(ый|ого)?\s+кодекс", r"(?<!\w)зк(?!\w)"),
    "BK": (r"бюджетн(ый|ого)?\s+кодекс", r"(?<!\w)бк(?!\w)"),
}

PRACTICE_INTENT_MARKERS = (
    "судебн", "практик", "пленум", "позици", "дело", "определен", "постановлен", "кассац", "апелляц",
)

QUERY_HINT_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "TK": ("работник", "работодател", "трудов", "увольнен", "зарплат", "отпуск", "сверхуроч", "испытательн"),
    "GK": ("неустойк", "обязательств", "договор", "сделк", "иск", "давност", "аренд", "арендатор", "субаренд", "убытк", "собственник"),
    "UK": ("преступлен", "наказан", "краж", "взятк", "убийств", "соучаст", "рецидив"),
    "UPK": ("подозреваем", "обвиняем", "следователь", "допрос", "обыск", "приговор", "доказательств"),
    "APK": ("арбитраж", "иск", "ответчик", "истец", "обеспечител", "миров", "апелляц"),
    "GPK": ("гражданск", "судопроизводств", "судебн", "заявлен", "истец", "ответчик"),
    "KOAP": ("административ", "правонаруш", "штраф", "протокол", "задержан", "постановлен"),
    "SK": ("брак", "супруг", "алим", "родитель", "ребен", "отцовств", "развод"),
    "JK": ("жилищ", "мкд", "квартир", "жильц", "коммунальн", "перепланировк", "выселен", "соцнайм"),
    "NK": ("налог", "ндс", "ндфл", "декларац", "инспекц", "усн", "камеральн", "выездн"),
    "ZK": ("земель", "участк", "сервитут", "егрн", "границ", "целев"),
}

ORG_ENTITY_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "Верховный Суд РФ": ("верховн", "суд"),
    "Конституционный Суд РФ": ("конституцион", "суд"),
    "Районный суд": ("район", "суд"),
    "Мировые судьи": ("миров", "суд"),
    "Арбитражные суды": ("арбитраж", "суд"),
    "Апелляционные суды общей юрисдикции": ("апелляцион", "общ", "юрисдикц"),
    "Кассационные суды общей юрисдикции": ("кассацион", "общ", "юрисдикц"),
    "МВД РФ": ("мвд", "внутрен", "дел"),
    "Следственный комитет РФ": ("следствен", "комитет"),
    "ФСБ РФ": ("фсб", "безопасн"),
    "СВР РФ": ("свр", "внешн", "разведк"),
    "Прокуратура РФ": ("прокуратур",),
    "ФСИН РФ": ("фсин", "исполнен", "наказан"),
    "Росгвардия": ("росгвард", "национальн", "гвард"),
}

ENTITY_LAW_MARKERS: Dict[str, Tuple[str, ...]] = {
    "Верховный Суд РФ": ("верховном суде", "о судебной системе"),
    "Конституционный Суд РФ": ("конституционном суде", "конституционного суда"),
    "Районный суд": ("о судах общей юрисдикции",),
    "Мировые судьи": ("о мировых судьях",),
    "Арбитражные суды": ("об арбитражных судах",),
    "Апелляционные суды общей юрисдикции": ("о судах общей юрисдикции", "апелляционного суда общей юрисдикции"),
    "Кассационные суды общей юрисдикции": ("о судах общей юрисдикции", "кассационного суда общей юрисдикции"),
    "МВД РФ": ("о полиции", "внутренних дел"),
    "Следственный комитет РФ": ("о следственном комитете", "ск рф"),
    "ФСБ РФ": ("о федеральной службе безопасности", "фсб"),
    "СВР РФ": ("о внешней разведке", "свр"),
    "Прокуратура РФ": ("о прокуратуре",),
    "ФСИН РФ": ("уголовно-исполнительной системе", "исполнения наказаний", "фсин"),
    "Росгвардия": ("национальной гвардии", "росгвард"),
}

ORG_PROFILE_INTENT_MARKERS = (
    "правов", "основ", "полномоч", "структур", "подотчет", "подотчетн",
    "формирован", "место в системе", "определен",
)

GENERIC_QUERY_TOKENS = {
    "вопрос", "основан", "согласн", "поряд", "договор", "прав", "обязан", "ответствен",
    "стат", "пункт", "российск", "федерац", "кодекс", "работник", "работодател",
    "правов", "основ", "деятельн", "структур", "полномоч", "орган", "государствен", "власт",
}

RU_STOPWORDS = {
    "это", "как", "для", "или", "при", "его", "ее", "они", "она", "оно", "все", "всех", "если",
    "когда", "также", "согласно", "который", "которые", "которых", "быть", "есть", "надо", "нужно",
    "ваш", "ваша", "ваши", "какие", "какой", "какая", "какое", "такое", "где", "кто", "чем", "про",
    "после", "перед", "между", "этой", "этого", "этот", "эта", "эти", "что", "чего", "чему", "кому",
    "каких", "какому", "какой", "ли", "же", "по", "из", "на", "от", "до", "со", "под", "над", "об",
    "а", "и", "но", "не", "да", "у", "о", "к", "в", "с", "так",
}

RU_SUFFIXES = (
    "иями", "ями", "ами", "иях", "ев", "ов", "ие", "ые", "ое", "ей", "ий", "ый", "ой", "ем", "им", "ым", "ом",
    "его", "ого", "ему", "ому", "их", "ых", "ую", "юю", "ая", "яя", "ою", "ею", "ия", "ья", "ию", "ью",
    "иям", "ьям", "ием", "ьем", "ах", "ях", "ам", "ям", "ом", "ем", "а", "я", "ы", "и", "е", "у", "ю", "о",
)

STUDENT_SYSTEM_PROMPT = """
Ты — строгий юридический ассистент по праву РФ.

Ключевое правило:
- Отвечай только на основе переданного контекста. Никаких предположений и внешних знаний.

Требования к источникам:
- Если в контексте нет нормы или факта, прямо скажи: "Информации недостаточно."
- Не ссылайся на нормы, статьи и источники, которых нет в контексте.
- Для каждого правового тезиса указывай ссылку на источник в формате [S1], [S2].
- Если в контексте есть ссылка на источник (например, consultant.ru), можешь упомянуть ее. Иначе не добавляй ссылки.

Структура ответа:
1) Краткий вывод.
2) Правовое обоснование с логикой применения норм.
3) При необходимости — алгоритм действий/шаги.
4) Ограничения ответа (если есть).
5) Финальная фраза:
   "Этот ответ основан на действующем законодательстве РФ и предназначен для учебных целей. Для уточнения текста нормы — проверьте её на consultant.ru или проконсультируйтесь с преподавателем."
""".strip()


def load_prompt_template(prompt_path: Path = Path("prompts/legal_student_prompt_v2.txt")) -> str:
    try:
        if prompt_path.exists():
            text = prompt_path.read_text(encoding="utf-8").strip()
            if text:
                return text
    except Exception as exc:
        logger.warning("Failed to load prompt file %s: %s", prompt_path, exc)
    return STUDENT_SYSTEM_PROMPT


def check_runtime_dependencies() -> None:
    required_specs = {
        "langchain-core": "langchain_core",
        "langchain-community": "langchain_community",
        "langchain-openai": "langchain_openai",
        "sentence-transformers": "sentence_transformers",
        "rank-bm25": "rank_bm25",
        "faiss-cpu": "faiss",
    }
    missing = [pkg for pkg, mod in required_specs.items() if importlib.util.find_spec(mod) is None]

    if not missing and not IMPORT_ERRORS:
        return

    print("\n[Ошибка зависимостей] Отсутствуют пакеты или есть ошибки импорта:")
    for pkg in missing:
        print(f"  - {pkg}")
    for pkg, err in IMPORT_ERRORS.items():
        print(f"  - {pkg}: {err}")

    py = sys.executable
    print("\nУстановите зависимости в этот же интерпретатор:")
    print(f"  \"{py}\" -m pip install --upgrade pip")
    print(
        f"  \"{py}\" -m pip install "
        "langchain-core langchain-community langchain-openai sentence-transformers rank-bm25 faiss-cpu"
    )
    raise SystemExit(1)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().replace("ё", "е")).strip()


def normalize_law_code(raw: str) -> str:
    if not raw:
        return ""
    text = raw.upper().replace(" ", "")
    repl = str.maketrans(
        {
            "А": "A",
            "В": "B",
            "Е": "E",
            "К": "K",
            "М": "M",
            "Н": "H",
            "О": "O",
            "Р": "P",
            "С": "C",
            "Т": "T",
            "У": "Y",
            "Х": "X",
        }
    )
    return text.translate(repl)


def canonical_law_codes_from_text(text: str) -> Set[str]:
    normalized = normalize_text(text)
    found: Set[str] = set()
    for code, patterns in LAW_CANON_PATTERNS.items():
        if any(re.search(p, normalized) for p in patterns):
            found.add(code)
    return found


def canonical_law_codes_from_metadata(metadata: Dict[str, Any]) -> Set[str]:
    md = metadata or {}
    raw = (
        f"{md.get('law_id', '')} "
        f"{md.get('source_title', '')} "
        f"{md.get('hierarchy_str', '')} "
        f"{md.get('source_type', '')}"
    )
    return canonical_law_codes_from_text(raw)


def extract_law_hints(query: str) -> Set[str]:
    q = normalize_text(query)
    hints: Set[str] = set()
    for code, patterns in LAW_HINT_PATTERNS.items():
        if any(re.search(p, q) for p in patterns):
            hints.add(code)
    return hints


def infer_law_hints(query: str) -> Set[str]:
    q_tokens = tokenize(query)
    if not q_tokens:
        return set()

    def token_match_score(keyword: str) -> int:
        score = 0
        for t in q_tokens:
            if t == keyword:
                score = max(score, 3)
            elif t.startswith(keyword) or keyword.startswith(t):
                score = max(score, 2)
            elif keyword in t:
                score = max(score, 1)
        return score

    scored: List[Tuple[int, str]] = []
    for code, keywords in QUERY_HINT_KEYWORDS.items():
        score = sum(token_match_score(kw) for kw in keywords)
        if score > 0:
            scored.append((score, code))

    if not scored:
        return set()

    scored.sort(reverse=True)
    top_score = scored[0][0]
    second_score = scored[1][0] if len(scored) > 1 else -1

    # Не уверены в отрасли: не форсируем жесткий фильтр.
    if top_score < 3:
        return set()
    if second_score >= 0 and (top_score - second_score) <= 1:
        return set()

    winners = [code for score, code in scored if score == top_score]
    return set(winners[:1])


def extract_entity_tokens(query: str) -> Set[str]:
    q_tokens = tokenize(query)
    entities: Set[str] = set()
    for entity in extract_entity_profiles(query):
        entities.update(ORG_ENTITY_KEYWORDS.get(entity, ()))
    return entities


def extract_entity_profiles(query: str) -> Set[str]:
    q_tokens = tokenize(query)
    if not q_tokens:
        return set()

    scored: List[Tuple[float, int, float, str]] = []
    for entity_name, keywords in ORG_ENTITY_KEYWORDS.items():
        matched = sum(1 for kw in keywords if any(t.startswith(kw) or kw.startswith(t) for t in q_tokens))
        if matched <= 0:
            continue
        ratio = matched / max(1, len(keywords))
        # full_match_priority, matched_count, ratio, entity_name
        scored.append((1.0 if matched == len(keywords) else 0.0, matched, ratio, entity_name))

    if not scored:
        return set()

    full = {entity for is_full, _, _, entity in scored if is_full >= 1.0}
    if full:
        return full

    best_ratio = max(ratio for _, _, ratio, _ in scored)
    threshold = max(0.67, best_ratio - 0.10)
    picked = {
        entity
        for _, matched, ratio, entity in scored
        if ratio >= threshold and matched >= 2
    }
    if picked:
        return picked
    # fallback: best single match entity for very short/abbreviation queries
    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return {scored[0][3]}


def collect_entity_law_markers(entity_profiles: Set[str]) -> Set[str]:
    markers: Set[str] = set()
    for entity in entity_profiles:
        markers.update(ENTITY_LAW_MARKERS.get(entity, ()))
    return markers


def is_org_profile_intent(query: str) -> bool:
    q = normalize_text(query)
    return any(marker in q for marker in ORG_PROFILE_INTENT_MARKERS)


def extract_article_hints(query: str) -> Set[str]:
    return {m.group(1) for m in ARTICLE_RE.finditer(query)}


def has_federal_law_intent(query: str) -> bool:
    q = normalize_text(query)
    return any(
        marker in q
        for marker in (
            "фз",
            "фкз",
            "федеральный закон",
            "федерального закона",
            "федеральный конституционный закон",
            "федерального конституционного закона",
        )
    )


def is_textbook_metadata(metadata: Dict[str, Any]) -> bool:
    md = metadata or {}
    source_type = normalize_text(str(md.get("source_type", "")))
    source_title = normalize_text(str(md.get("source_title", "")))
    law_id = normalize_text(str(md.get("law_id", "")))
    if any(marker in source_type for marker in TEXTBOOK_MARKERS):
        return True
    if any(marker in source_title for marker in TEXTBOOK_MARKERS):
        return True
    if "uch" in law_id:
        return True
    return False


def detect_source_policy(
    question: str,
    explicit_law_hints: Set[str],
    article_hints: Set[str],
) -> str:
    q = normalize_text(question)
    is_edu = any(marker in q for marker in EDU_INTENT_MARKERS)
    is_norm = any(marker in q for marker in NORMATIVE_INTENT_MARKERS)
    if explicit_law_hints or article_hints or is_norm:
        return "exclude_textbooks"
    if is_edu:
        return "prefer_textbooks"
    return "allow_all"

def normalize_token(token: str) -> str:
    t = token.lower().replace("ё", "е")
    if len(t) <= 3 or t.isdigit():
        return t
    for suffix in RU_SUFFIXES:
        if len(t) > len(suffix) + 2 and t.endswith(suffix):
            return t[: -len(suffix)]
    return t


def tokenize(text: str) -> Set[str]:
    tokens: Set[str] = set()
    for raw in TOKEN_RE.findall(normalize_text(text)):
        lowered = raw.lower().replace("ё", "е")
        if lowered in RU_STOPWORDS:
            continue
        token = normalize_token(lowered)
        if token in RU_STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def tokenize_bm25(text: str) -> List[str]:
    tokens: List[str] = []
    for raw in TOKEN_RE.findall(normalize_text(text)):
        lowered = raw.lower().replace("ё", "е")
        if lowered in RU_STOPWORDS:
            continue
        token = normalize_token(lowered)
        if token in RU_STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def is_informative_text(text: str) -> bool:
    if not text:
        return False
    stripped = text.strip()
    if len(stripped) < 40:
        return False
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", stripped)
    return len(letters) >= 20


def stable_doc_id(doc: Document) -> str:
    md = doc.metadata or {}
    for key in ("chunk_id", "id", "source_url"):
        value = md.get(key)
        if value:
            return str(value)
    base = f"{md.get('source_title','')}|{md.get('hierarchy_str','')}|{doc.page_content[:300]}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def is_practice_metadata(metadata: Dict[str, Any]) -> bool:
    data = normalize_text(
        f"{metadata.get('source_type', '')} {metadata.get('source_title', '')} "
        f"{metadata.get('law_id', '')} {metadata.get('hierarchy_str', '')}"
    )
    return any(marker in data for marker in PRACTICE_SOURCE_MARKERS)


def is_law_metadata(metadata: Dict[str, Any]) -> bool:
    data = normalize_text(
        f"{metadata.get('source_type', '')} {metadata.get('source_title', '')} "
        f"{metadata.get('law_id', '')} {metadata.get('hierarchy_str', '')}"
    )
    if any(marker in data for marker in LAW_SOURCE_MARKERS):
        return True
    return not is_practice_metadata(metadata)


class LaBSEEmbeddings(Embeddings):
    def __init__(self, model_name: str = "cointegrated/LaBSE-en-ru"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cuda")
        logger.info("Embeddings model loaded: %s", model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0].tolist()

    def embedding_dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())


@dataclass
class RetrievalDebug:
    query_variants: List[str]
    law_hints: List[str]
    explicit_law_hints: List[str]
    inferred_law_hints: List[str]
    entity_tokens: List[str]
    article_hints: List[str]
    law_docs: int
    practice_docs: int
    lexical_law_docs: int
    final_docs: int
    vector_docs: int = 0
    bm25_docs: int = 0
    fused_docs: int = 0
    rerank_model: str = ""
    source_policy: str = ""
    textbook_docs: int = 0


class OtusStyleRAG:
    def __init__(
        self,
        index_dir: Path = Path("faiss_indexes"),
        index_name: str = "law_db",
        data_dir: Path = Path("NPA3001"),
        embedding_model: str = "cointegrated/LaBSE-en-ru",
        llm_model: str = "gpt-4o-mini",
        use_llm_query_expansion: bool = False,
        prompt_path: Path = Path("prompts/legal_student_prompt_v2.txt"),
    ):
        load_dotenv("config.env")
        load_dotenv(".env")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY не найден в config.env/.env")

        self.embeddings = LaBSEEmbeddings(embedding_model)
        self.vector_store = FAISS.load_local(
            folder_path=str(index_dir),
            embeddings=self.embeddings,
            index_name=index_name,
            allow_dangerous_deserialization=True,
        )

        index_dim = int(getattr(self.vector_store.index, "d"))
        emb_dim = int(self.embeddings.embedding_dimension())
        if index_dim != emb_dim:
            raise RuntimeError(
                f"Несовпадение размерности: FAISS={index_dim}, embeddings={emb_dim}. "
                f"Нужно пересобрать индекс под {embedding_model}."
            )

        patched = self._patch_docstore_from_jsonl(data_dir)
        self.docs: List[Document] = list(self.vector_store.docstore._dict.values())
        self.law_buckets = self._build_law_buckets(self.docs)
        law_cnt = sum(1 for d in self.docs if is_law_metadata(d.metadata or {}))
        practice_cnt = sum(1 for d in self.docs if is_practice_metadata(d.metadata or {}))
        logger.info(
            "FAISS loaded: docs=%s, patched=%s, law_docs=%s, practice_docs=%s",
            len(self.docs),
            patched,
            law_cnt,
            practice_cnt,
        )

        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=0.1,
            timeout=90,
            max_retries=2,
            api_key=api_key,
        )
        self.use_llm_query_expansion = use_llm_query_expansion
        self.system_prompt = load_prompt_template(prompt_path)
        self.last_llm_query = ""
        self.last_retrieval: Dict[str, Any] = {}
        self.history_dir = Path("rag_history")
        self.history_path = self.history_dir / "local.jsonl"
        self.last_llm_usage: Dict[str, Any] = {}

        self._bm25: Optional[Any] = None
        self._bm25_docs: List[Document] = []
        self._bm25_doc_ids: List[str] = []
        self._bm25_ready = False
        self._reranker: Optional[Any] = None
        self._reranker_attempted = False

        self._build_bm25_index()

    @staticmethod
    def _build_law_buckets(docs: Iterable[Document]) -> Dict[str, List[Document]]:
        buckets: Dict[str, List[Document]] = {}
        for doc in docs:
            codes = canonical_law_codes_from_metadata(doc.metadata or {})
            if not codes:
                continue
            for code in codes:
                buckets.setdefault(code, []).append(doc)
        return buckets

    @staticmethod
    def _load_clean_chunk_map(data_dir: Path) -> Dict[str, Document]:
        if not data_dir.exists():
            logger.warning("Data dir not found: %s", data_dir)
            return {}

        chunk_map: Dict[str, Document] = {}
        jsonl_files = list(data_dir.glob("*.jsonl"))
        if not jsonl_files:
            logger.warning("No JSONL files in %s", data_dir)
            return {}

        for fp in jsonl_files:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue

                    chunk_id = item.get("chunk_id")
                    text = (item.get("text") or item.get("content") or "").strip()
                    if not chunk_id or not is_informative_text(text):
                        continue
                    if chunk_id in chunk_map:
                        continue

                    md = {
                        "id": item.get("id", ""),
                        "source_title": item.get("source_title", ""),
                        "hierarchy_str": item.get("hierarchy_str", ""),
                        "law_id": item.get("law_id", ""),
                        "source_type": item.get("source_type", ""),
                        "source_url": item.get("source_url", ""),
                        "chunk_id": chunk_id,
                    }
                    chunk_map[chunk_id] = Document(page_content=text, metadata=md)

        logger.info("Loaded clean chunk map: %s", len(chunk_map))
        return chunk_map

    def _patch_docstore_from_jsonl(self, data_dir: Path) -> int:
        clean = self._load_clean_chunk_map(data_dir)
        if not clean:
            return 0

        replaced = 0
        for key, old_doc in self.vector_store.docstore._dict.items():
            md = dict(old_doc.metadata or {})
            chunk_id = md.get("chunk_id")
            clean_doc = clean.get(chunk_id)
            if not clean_doc:
                continue
            merged_md = dict(md)
            merged_md.update(clean_doc.metadata or {})
            self.vector_store.docstore._dict[key] = Document(page_content=clean_doc.page_content, metadata=merged_md)
            replaced += 1
        return replaced

    def _build_bm25_index(self) -> None:
        if BM25Okapi is None:
            logger.warning("BM25 unavailable: rank-bm25 not installed.")
            return

        docs: List[Document] = []
        doc_ids: List[str] = []
        corpus_tokens: List[List[str]] = []
        for doc in self.docs:
            if not is_informative_text(doc.page_content):
                continue
            md = doc.metadata or {}
            text = (
                f"{md.get('source_title','')} {md.get('hierarchy_str','')} "
                f"{md.get('law_id','')} {doc.page_content}"
            )
            tokens = tokenize_bm25(text)
            if not tokens:
                continue
            docs.append(doc)
            doc_ids.append(stable_doc_id(doc))
            corpus_tokens.append(tokens)

        if not corpus_tokens:
            logger.warning("BM25 index not built: empty corpus.")
            return

        self._bm25 = BM25Okapi(corpus_tokens)
        self._bm25_docs = docs
        self._bm25_doc_ids = doc_ids
        self._bm25_ready = True
        logger.info("BM25 index ready: docs=%s", len(self._bm25_docs))

    @staticmethod
    def _doc_text_for_rerank(doc: Document) -> str:
        md = doc.metadata or {}
        return f"{md.get('source_title','')} {md.get('hierarchy_str','')} {doc.page_content[:1800]}"

    def _vector_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
        except Exception as exc:
            logger.warning("Vector search failed: %s", exc)
            return []

        out: List[Tuple[Document, float]] = []
        for doc, distance in results:
            if not is_informative_text(doc.page_content):
                continue
            score = 1.0 / (1.0 + float(distance))
            out.append((doc, score))
        return out

    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[Document, float]]:
        if not self._bm25_ready or self._bm25 is None:
            return []
        tokens = tokenize_bm25(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        if scores is None or len(scores) == 0:
            return []
        top_indices = heapq.nlargest(top_k, range(len(scores)), key=scores.__getitem__)
        out: List[Tuple[Document, float]] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0.0:
                continue
            out.append((self._bm25_docs[idx], score))
        return out

    def _get_reranker(self) -> Optional[Any]:
        if self._reranker_attempted:
            return self._reranker
        self._reranker_attempted = True
        if CrossEncoder is None:
            return None
        try:
            self._reranker = CrossEncoder(RERANK_MODEL, max_length=512, device="cuda")
            logger.info("Reranker enabled: %s", RERANK_MODEL)
        except Exception as exc:
            logger.warning("Reranker unavailable, fallback lexical: %s", exc)
            self._reranker = None
        return self._reranker

    def _rerank_candidates(self, query: str, candidates: List[Document]) -> Tuple[List[Document], Dict[str, float], str]:
        if not candidates:
            return [], {}, "none"

        model = self._get_reranker()
        score_map: Dict[str, float] = {}
        if model is not None:
            try:
                pairs = [(query, self._doc_text_for_rerank(doc)) for doc in candidates]
                scores = model.predict(pairs, batch_size=16, show_progress_bar=False)
                for doc, score in zip(candidates, scores):
                    score_map[stable_doc_id(doc)] = float(score)
                ranked = sorted(
                    candidates,
                    key=lambda d: score_map.get(stable_doc_id(d), 0.0),
                    reverse=True,
                )
                return ranked, score_map, RERANK_MODEL
            except Exception as exc:
                logger.warning("Reranker failed, fallback lexical: %s", exc)

        q_tokens = tokenize(query)
        for doc in candidates:
            score_map[stable_doc_id(doc)] = len(tokenize(self._doc_text_for_rerank(doc)).intersection(q_tokens))
        ranked = sorted(candidates, key=lambda d: score_map.get(stable_doc_id(d), 0.0), reverse=True)
        return ranked, score_map, "lexical_fallback"

    @staticmethod
    def _rrf_merge(
        lists_of_docs: List[List[Document]],
        weights: Optional[List[float]] = None,
    ) -> Tuple[List[Document], Dict[str, float]]:
        scores: Dict[str, float] = {}
        docs: Dict[str, Document] = {}

        if weights is None:
            weights = [1.0] * len(lists_of_docs)

        for docs_list, weight in zip(lists_of_docs, weights):
            for rank, doc in enumerate(docs_list, start=1):
                doc_id = stable_doc_id(doc)
                scores[doc_id] = scores.get(doc_id, 0.0) + weight / (RRF_K + rank)
                docs[doc_id] = doc

        ranked_ids = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
        return [docs[i] for i in ranked_ids], scores

    @staticmethod
    def _apply_candidate_boost(
        docs: List[Document],
        rrf_scores: Dict[str, float],
        law_hints: Set[str],
        explicit_law_hints: Set[str],
        entity_law_markers: Set[str],
        article_hints: Set[str],
        org_profile_intent: bool,
    ) -> List[Document]:
        if not docs:
            return docs
        boosted: List[Tuple[float, int, Document]] = []
        for idx, doc in enumerate(docs):
            md = doc.metadata or {}
            source_type = normalize_text(str(md.get("source_type", "")))
            meta_blob = normalize_text(
                f"{md.get('law_id', '')} {md.get('source_title', '')} {md.get('hierarchy_str', '')}"
            )
            header = normalize_text(f"{md.get('source_title', '')} {md.get('hierarchy_str', '')}")
            doc_id = stable_doc_id(doc)
            bonus = 0.0

            if law_hints:
                codes = canonical_law_codes_from_metadata(md)
                if codes.intersection(law_hints):
                    bonus += 0.25
                elif explicit_law_hints:
                    bonus -= 0.20

            if entity_law_markers and any(marker in meta_blob for marker in entity_law_markers):
                bonus += 0.30

            if article_hints and any(article in header for article in article_hints):
                bonus += 0.20

            if org_profile_intent:
                if "федеральный конституционный закон" in source_type:
                    bonus += 0.25
                elif "федеральный закон" in source_type:
                    bonus += 0.20
                elif "конституция" in source_type:
                    bonus += 0.15
                elif "кодекс" in source_type:
                    bonus -= 0.15

            base = rrf_scores.get(doc_id, 0.0)
            boosted.append((base + bonus, -idx, doc))

        boosted.sort(reverse=True)
        return [doc for _, _, doc in boosted]

    def _make_metadata_filter(
        self,
        channel: str,
        law_hints: Set[str],
        query_tokens: Optional[Set[str]] = None,
        allow_federal_supplement: bool = False,
        entity_law_markers: Optional[Set[str]] = None,
        org_profile_intent: bool = False,
    ) -> Callable[[Dict[str, Any]], bool]:
        def _filter(metadata: Dict[str, Any]) -> bool:
            md = metadata or {}
            if channel == "law":
                if not is_law_metadata(md):
                    return False
            else:
                if not is_practice_metadata(md):
                    return False

            source_type = normalize_text(str(md.get("source_type", "")))
            title = normalize_text(str(md.get("source_title", "")))
            hierarchy = normalize_text(str(md.get("hierarchy_str", "")))
            law_id = normalize_law_code(str(md.get("law_id", "")))
            combined = f"{law_id} {title} {hierarchy} {source_type}"
            marker_hit = bool(entity_law_markers and any(marker in combined for marker in entity_law_markers))

            if org_profile_intent and channel == "law":
                # Для вопросов по госорганам повышаем шанс профильных ФЗ/ФКЗ.
                if marker_hit:
                    return True
                if "кодекс" in source_type:
                    return False
                if query_tokens:
                    metadata_tokens = tokenize(f"{md.get('law_id', '')} {md.get('source_title', '')} {md.get('hierarchy_str', '')}")
                    if query_tokens.intersection(metadata_tokens):
                        return True

            if not law_hints:
                return True

            canonical_codes = canonical_law_codes_from_metadata(md)
            if canonical_codes.intersection(law_hints):
                return True

            for hint in law_hints:
                if hint in law_id:
                    return True
                for pattern in LAW_HINT_TITLE_PATTERNS.get(hint, ()):
                    if re.search(pattern, combined):
                        return True

            if allow_federal_supplement and channel == "law":
                if "федеральный закон" in source_type or "федеральный конституционный закон" in source_type:
                    if marker_hit:
                        return True
                    if not query_tokens:
                        return True
                    metadata_tokens = tokenize(
                        f"{md.get('law_id', '')} {md.get('source_title', '')} {md.get('hierarchy_str', '')}"
                    )
                    if query_tokens.intersection(metadata_tokens):
                        return True
            return False

        return _filter

    def _is_practice_intent(self, question: str) -> bool:
        q = normalize_text(question)
        return any(marker in q for marker in PRACTICE_INTENT_MARKERS)

    def _generate_rule_based_variants(self, question: str, law_hints: Set[str], max_variants: int = 4) -> List[str]:
        variants = [question]
        q_norm = normalize_text(question)
        q_tokens = tokenize(question)
        entity_tokens = extract_entity_tokens(question)
        entity_profiles = extract_entity_profiles(question)
        entity_markers = collect_entity_law_markers(entity_profiles)

        if law_hints:
            code_phrase = {
                "TK": "Трудовой кодекс РФ",
                "GK": "Гражданский кодекс РФ",
                "NK": "Налоговый кодекс РФ",
                "UK": "Уголовный кодекс РФ",
                "UPK": "УПК РФ",
                "GPK": "ГПК РФ",
                "APK": "АПК РФ",
                "KOAP": "КоАП РФ",
                "SK": "Семейный кодекс РФ",
                "JK": "Жилищный кодекс РФ",
                "ZK": "Земельный кодекс РФ",
            }
            for hint in sorted(law_hints):
                phrase = code_phrase.get(hint, f"{hint} РФ")
                variants.append(f"{question} ({phrase})")

        focus = [t for t in q_tokens if len(t) >= 5 and t not in GENERIC_QUERY_TOKENS and t not in RU_STOPWORDS]
        if focus:
            variants.append(" ".join(focus[:6]))
        if entity_tokens:
            variants.append(" ".join(sorted(entity_tokens)) + " правовая основа полномочия структура")
        if entity_profiles:
            for profile in sorted(entity_profiles):
                variants.append(f"{profile} правовая основа полномочия структура федеральный закон фкз")
        if entity_markers:
            variants.append(" ".join(sorted(entity_markers)[:6]))

        if "инициативе работодателя" in q_norm:
            variants.append("расторжение трудового договора по инициативе работодателя статья 81 тк рф")
        if "неустойк" in q_norm:
            variants.append("понятие неустойки статья 330 гк рф штраф пени")

        merged: List[str] = []
        for v in variants:
            clean = v.strip()
            if clean and clean not in merged:
                merged.append(clean)
        return merged[:max_variants]

    def _generate_query_variants(
        self,
        question: str,
        max_variants: int = 4,
        law_hints: Optional[Set[str]] = None,
    ) -> List[str]:
        law_hints = set(law_hints or extract_law_hints(question))
        base_variants = self._generate_rule_based_variants(question, law_hints=law_hints, max_variants=max_variants)

        if not self.use_llm_query_expansion:
            return base_variants

        prompt = f"""
Сгенерируй до {max_variants} поисковых формулировок для юридического запроса.
Требования:
- Сохрани исходный юридический смысл.
- Используй термины, полезные для поиска по НПА и судебной практике.
- Не добавляй вымышленных фактов.
- Верни только JSON-массив строк, без пояснений.

Запрос:
{question}
"""
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = str(response.content).strip()
            variants: List[str] = []
            cleaned = re.sub(r"```(?:json)?", "", content, flags=re.IGNORECASE).replace("```", "").strip()

            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, list):
                    variants = [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                bracket_match = re.search(r"\[[\s\S]*\]", cleaned)
                if bracket_match:
                    candidate_json = bracket_match.group(0)
                    try:
                        parsed = json.loads(candidate_json)
                        if isinstance(parsed, list):
                            variants = [str(x).strip() for x in parsed if str(x).strip()]
                    except Exception:
                        pass

                if not variants:
                    quoted = re.findall(r"\"([^\"]+)\"", cleaned)
                    if quoted:
                        variants = [q.strip() for q in quoted if q.strip()]

                if not variants:
                    for line in cleaned.splitlines():
                        stripped = line.strip().strip(",")
                        if not stripped or stripped in {"[", "]"}:
                            continue
                        if stripped.lower() == "json":
                            continue
                        candidate = re.sub(r"^\s*\d+[\).\-:]\s*", "", stripped).strip(" -•\t\"")
                        if candidate:
                            variants.append(candidate)

            filtered_variants: List[str] = []
            for variant in variants:
                v = variant.strip().strip("\"")
                if len(v) < 8:
                    continue
                if v.lower() in {"json", "[", "]"}:
                    continue
                filtered_variants.append(v)

            merged = list(base_variants)
            for v in filtered_variants:
                if v not in merged:
                    merged.append(v)
            return merged[:max_variants]
        except Exception as exc:
            logger.warning("Multi-query generation failed: %s", exc)
            return base_variants

    def _mmr_search(
        self,
        query: str,
        channel: str,
        law_hints: Set[str],
        query_tokens: Optional[Set[str]] = None,
        allow_federal_supplement: bool = False,
        entity_law_markers: Optional[Set[str]] = None,
        org_profile_intent: bool = False,
    ) -> List[Document]:
        if channel == "law":
            k, fetch_k, lambda_mult = LAW_K, LAW_FETCH_K, LAW_LAMBDA
        else:
            k, fetch_k, lambda_mult = PRACTICE_K, PRACTICE_FETCH_K, PRACTICE_LAMBDA

        filter_fn = self._make_metadata_filter(
            channel=channel,
            law_hints=law_hints,
            query_tokens=query_tokens,
            allow_federal_supplement=allow_federal_supplement,
            entity_law_markers=entity_law_markers,
            org_profile_intent=org_profile_intent,
        )

        try:
            docs = self.vector_store.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filter=filter_fn,
            )
        except Exception as exc:
            logger.warning("MMR search failed for channel=%s: %s", channel, exc)
            return []

        out: List[Document] = []
        for doc in docs:
            if not is_informative_text(doc.page_content):
                continue
            out.append(doc)
        return out

    def _law_guided_lexical_search(
        self,
        query_tokens: Set[str],
        focus_tokens: Set[str],
        law_hints: Set[str],
        per_law: int = 80,
    ) -> List[Document]:
        if not law_hints or not query_tokens:
            return []

        scored: List[Tuple[float, Document]] = []
        for hint in law_hints:
            docs = self.law_buckets.get(hint, [])
            if not docs:
                continue

            local: List[Tuple[float, Document]] = []
            for doc in docs:
                if not is_informative_text(doc.page_content):
                    continue

                md = doc.metadata or {}
                searchable = (
                    f"{md.get('source_title', '')} "
                    f"{md.get('hierarchy_str', '')} "
                    f"{doc.page_content[:1400]}"
                )
                tokens = tokenize(searchable)
                overlap = query_tokens.intersection(tokens)
                if not overlap:
                    continue

                focus_overlap = focus_tokens.intersection(tokens) if focus_tokens else set()
                score = len(overlap) / max(1, len(query_tokens))
                if focus_overlap:
                    score += 0.75 * (len(focus_overlap) / max(1, len(focus_tokens)))
                local.append((score, doc))

            local.sort(key=lambda x: x[0], reverse=True)
            scored.extend(local[:per_law])

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored]

    def retrieve(
        self,
        question: str,
        top_k: int = 8,
        use_hybrid: bool = True,
        use_rerank: bool = True,
    ) -> Tuple[List[Document], RetrievalDebug]:
        explicit_law_hints = extract_law_hints(question)
        entity_profiles = extract_entity_profiles(question)
        org_profile_intent = bool(entity_profiles) and is_org_profile_intent(question)
        inferred_law_hints = infer_law_hints(question) if not explicit_law_hints else set()
        if org_profile_intent and not explicit_law_hints:
            inferred_law_hints = set()
        law_hints = set(explicit_law_hints or inferred_law_hints)

        article_hints = extract_article_hints(question)
        variants = self._generate_query_variants(question, max_variants=4, law_hints=law_hints)
        entity_tokens = extract_entity_tokens(question)
        entity_law_markers = collect_entity_law_markers(entity_profiles)
        source_policy = detect_source_policy(question, explicit_law_hints, article_hints)

        vector_runs: List[List[Document]] = []
        bm25_runs: List[List[Document]] = []

        vector_doc_ids: Set[str] = set()
        bm25_doc_ids: Set[str] = set()

        for variant in variants:
            vec = self._vector_search(variant, k=max(VECTOR_FETCH_K, top_k * 8))
            vec_docs = [doc for doc, _ in vec]
            vector_runs.append(vec_docs)
            vector_doc_ids.update(stable_doc_id(d) for d in vec_docs)

            if use_hybrid:
                bm25 = self._bm25_search(variant, top_k=max(BM25_FETCH_K, top_k * 8))
                bm25_docs = [doc for doc, _ in bm25]
                bm25_runs.append(bm25_docs)
                bm25_doc_ids.update(stable_doc_id(d) for d in bm25_docs)

        fusion_lists = vector_runs + bm25_runs if use_hybrid else vector_runs
        fused_docs, rrf_scores = self._rrf_merge(fusion_lists) if fusion_lists else ([], {})
        fused_docs = self._apply_candidate_boost(
            fused_docs,
            rrf_scores,
            law_hints=law_hints,
            explicit_law_hints=explicit_law_hints,
            entity_law_markers=entity_law_markers,
            article_hints=article_hints,
            org_profile_intent=org_profile_intent,
        )

        # Строгий фильтр по кодексам/нормам (если явные подсказки).
        if explicit_law_hints:
            strict_matches = [
                d for d in fused_docs
                if canonical_law_codes_from_metadata(d.metadata or {}).intersection(explicit_law_hints)
            ]
            if len(strict_matches) >= max(3, min(10, top_k)):
                fused_docs = strict_matches

        # Если указан номер статьи — усиливаем точность.
        if article_hints:
            article_matches = []
            for d in fused_docs:
                header = normalize_text(
                    f"{(d.metadata or {}).get('hierarchy_str', '')} {(d.metadata or {}).get('source_title', '')}"
                )
                if any(article in header for article in article_hints):
                    article_matches.append(d)
            if len(article_matches) >= max(2, min(8, top_k)):
                fused_docs = article_matches

        # Политика по учебникам.
        textbook_docs = [d for d in fused_docs if is_textbook_metadata(d.metadata or {})]
        if source_policy == "exclude_textbooks":
            non_textbooks = [d for d in fused_docs if not is_textbook_metadata(d.metadata or {})]
            fused_docs = non_textbooks
        elif source_policy == "prefer_textbooks":
            if len(textbook_docs) >= max(3, min(8, top_k)):
                fused_docs = textbook_docs

        candidate_docs = fused_docs[: max(RERANK_CANDIDATES, top_k)]
        if use_rerank:
            reranked_docs, rerank_scores, rerank_model = self._rerank_candidates(question, candidate_docs)
        else:
            reranked_docs = candidate_docs
            rerank_scores = {stable_doc_id(d): rrf_scores.get(stable_doc_id(d), 0.0) for d in candidate_docs}
            rerank_model = "rrf_only"

        final_k = min(top_k, RERANK_TOP_K) if use_rerank else top_k
        final_docs = reranked_docs[:final_k]

        law_docs = sum(1 for d in fused_docs if is_law_metadata(d.metadata or {}))
        practice_docs = sum(1 for d in fused_docs if is_practice_metadata(d.metadata or {}))
        textbook_count = sum(1 for d in fused_docs if is_textbook_metadata(d.metadata or {}))

        debug = RetrievalDebug(
            query_variants=variants,
            law_hints=sorted(law_hints),
            explicit_law_hints=sorted(explicit_law_hints),
            inferred_law_hints=sorted(inferred_law_hints),
            entity_tokens=sorted(entity_tokens),
            article_hints=sorted(article_hints),
            law_docs=law_docs,
            practice_docs=practice_docs,
            lexical_law_docs=len(bm25_doc_ids),
            final_docs=len(final_docs),
            vector_docs=len(vector_doc_ids),
            bm25_docs=len(bm25_doc_ids),
            fused_docs=len(fused_docs),
            rerank_model=rerank_model,
            source_policy=source_policy,
            textbook_docs=textbook_count,
        )

        self.last_retrieval = {
            "query_variants": variants,
            "vector_hits": len(vector_doc_ids),
            "bm25_hits": len(bm25_doc_ids),
            "fused_hits": len(fused_docs),
            "candidate_count": len(candidate_docs),
            "rrf_scores": rrf_scores,
            "rerank_scores": rerank_scores,
            "reranker_model": rerank_model,
            "use_hybrid": use_hybrid,
            "use_rerank": use_rerank,
            "final_docs": final_docs,
            "candidate_docs": candidate_docs,
            "final_k": final_k,
            "source_policy": source_policy,
            "textbook_docs": textbook_count,
        }

        return final_docs, debug

    @staticmethod
    def _build_context(docs: List[Document]) -> str:
        parts: List[str] = []
        for i, doc in enumerate(docs, start=1):
            md = doc.metadata or {}
            title = md.get("source_title", "Неизвестный источник")
            hierarchy = md.get("hierarchy_str", "")
            law_id = md.get("law_id", "")
            source_type = md.get("source_type", "")
            article = ""
            match = ARTICLE_RE.search(str(hierarchy))
            if match:
                article = match.group(1)
            if law_id:
                title = f"{title} ({law_id})"
            header = (
                f"[S{i}] [Источник: {title}; "
                f"Статья: {article or 'не указана'}; "
                f"Тип документа: {source_type or 'не указан'}; "
                f"Иерархия: {hierarchy or 'не указана'}]"
            )
            parts.append(f"{header}\n{doc.page_content}")
        return "\n\n".join(parts)

    def build_answer_prompt(self, question: str, docs: List[Document]) -> str:
        return f"""
Отвечай строго по контексту.
Если данных недостаточно — напиши "Информации недостаточно." и не додумывай.
Для каждого правового тезиса ставь ссылку [S#] в конце предложения.
Если в вопросе есть тест с вариантами — выбери только один правильный вариант и обоснуй нормой.
Не добавляй ссылки на источники, которых нет в контексте.

Структура ответа:
1) Краткий вывод.
2) Правовое обоснование с нормами и их логикой применения.
3) При необходимости — алгоритм действий/шаги.
4) Ограничения ответа (если есть).
5) Финальная фраза из системной инструкции — дословно.

Контекст:
{self._build_context(docs)}

Вопрос:
{question}
"""

    def answer(self, question: str, docs: List[Document]) -> str:
        if not docs:
            self.last_llm_query = ""
            self.last_llm_usage = {}
            return "Информации недостаточно."

        prompt = self.build_answer_prompt(question, docs)
        self.last_llm_query = prompt
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt),
        ]
        response = self.llm.invoke(messages)
        self.last_llm_usage = self._extract_llm_usage(response)
        return str(response.content)

    @staticmethod
    def _extract_llm_usage(response: Any) -> Dict[str, Any]:
        usage: Dict[str, Any] = {}
        if response is None:
            return usage
        meta = getattr(response, "response_metadata", None)
        if isinstance(meta, dict):
            if isinstance(meta.get("token_usage"), dict):
                usage = dict(meta["token_usage"])
            elif isinstance(meta.get("usage"), dict):
                usage = dict(meta["usage"])
        usage_meta = getattr(response, "usage_metadata", None)
        if isinstance(usage_meta, dict) and usage_meta:
            usage = dict(usage_meta)
        return usage

    @staticmethod
    def _sanitize_user_id(value: Optional[str]) -> str:
        if not value:
            return "local"
        safe = re.sub(r"[^0-9A-Za-z_-]+", "_", str(value)).strip("_")
        return safe or "local"

    def _append_history_entry(self, payload: Dict[str, Any], user_id: Optional[str]) -> None:
        try:
            self.history_dir.mkdir(parents=True, exist_ok=True)
            user_key = self._sanitize_user_id(user_id)
            history_path = self.history_dir / f"{user_key}.jsonl"
            with history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("Failed to write history: %s", exc)

    def _build_history_entry(
        self,
        question: str,
        answer: str,
        docs: List[Document],
        user_id: Optional[str],
    ) -> Dict[str, Any]:
        retrieval = self.last_retrieval or {}
        candidate_docs = retrieval.get("candidate_docs", docs) or []
        rrf_scores = retrieval.get("rrf_scores", {}) or {}
        rerank_scores = retrieval.get("rerank_scores", {}) or {}

        found_documents = []
        for rank, doc in enumerate(candidate_docs, start=1):
            doc_id = stable_doc_id(doc)
            found_documents.append(
                {
                    "rank": rank,
                    "doc_id": doc_id,
                    "rerank_score": rerank_scores.get(doc_id),
                    "rrf_score": rrf_scores.get(doc_id),
                    "metadata": doc.metadata or {},
                    "text": doc.page_content,
                }
            )

        pipeline_debug = {
            "use_hybrid": retrieval.get("use_hybrid", True),
            "use_rerank": retrieval.get("use_rerank", True),
            "vector_hits": retrieval.get("vector_hits", 0),
            "bm25_hits": retrieval.get("bm25_hits", 0),
            "fused_hits": retrieval.get("fused_hits", 0),
            "candidate_count": retrieval.get("candidate_count", 0),
            "reranker_model": retrieval.get("reranker_model", ""),
            "final_k": retrieval.get("final_k", len(docs)),
        }

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": self._sanitize_user_id(user_id),
            "question": question,
            "query_variants": retrieval.get("query_variants", []),
            "found_documents": found_documents,
            "final_query_sent_to_llm": self.last_llm_query,
            "answer": answer,
            "llm_usage": self.last_llm_usage,
            "pipeline_debug": pipeline_debug,
        }

    def ask(self, question: str, top_k: int = 8, user_id: Optional[str] = None) -> Dict[str, Any]:
        docs, debug = self.retrieve(question, top_k=top_k)
        answer = self.answer(question, docs)
        self._append_history_entry(self._build_history_entry(question, answer, docs, user_id), user_id)
        sources = []
        for i, doc in enumerate(docs, start=1):
            md = doc.metadata or {}
            sources.append(
                {
                    "id": i,
                    "title": md.get("source_title", ""),
                    "law_id": md.get("law_id", ""),
                    "source_type": md.get("source_type", ""),
                    "hierarchy": md.get("hierarchy_str", ""),
                    "preview": (doc.page_content[:220] + "...") if len(doc.page_content) > 220 else doc.page_content,
                }
            )
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "debug": {
                "query_variants": debug.query_variants,
                "law_hints": debug.law_hints,
                "explicit_law_hints": debug.explicit_law_hints,
                "inferred_law_hints": debug.inferred_law_hints,
                "entity_tokens": debug.entity_tokens,
                "article_hints": debug.article_hints,
                "law_docs": debug.law_docs,
                "practice_docs": debug.practice_docs,
                "lexical_law_docs": debug.lexical_law_docs,
                "final_docs": debug.final_docs,
                "vector_docs": debug.vector_docs,
                "bm25_docs": debug.bm25_docs,
                "fused_docs": debug.fused_docs,
                "rerank_model": debug.rerank_model,
                "source_policy": debug.source_policy,
                "textbook_docs": debug.textbook_docs,
            },
        }


def print_result(result: Dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("ВОПРОС:")
    print(result["question"])
    print("=" * 72)
    print("DEBUG:", result["debug"])
    print("\nОТВЕТ:\n")
    print(result["answer"])
    print("\nИСТОЧНИКИ:")
    for src in result["sources"]:
        print(f"- [{src['id']}] {src['title']} ({src['law_id']})")
        if src["source_type"]:
            print(f"  Тип: {src['source_type']}")
        if src["hierarchy"]:
            print(f"  {src['hierarchy']}")


def _split_message(text: str, limit: int = 4000) -> List[str]:
    if len(text) <= limit:
        return [text]
    parts: List[str] = []
    current = ""
    for line in text.splitlines():
        if len(current) + len(line) + 1 <= limit:
            current += line + "\n"
        else:
            parts.append(current.strip())
            current = line + "\n"
    if current.strip():
        parts.append(current.strip())
    return parts


async def run_telegram_bot(rag: OtusStyleRAG) -> None:
    if Bot is None or Dispatcher is None or types is None:
        raise RuntimeError("aiogram не установлен. Установите: pip install aiogram")

    load_dotenv("config.env")
    load_dotenv(".env")
    token = os.getenv("TELEGRAM_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_TOKEN не найден в config.env/.env")

    bot = Bot(token=token)
    dp = Dispatcher()

    @dp.message(CommandStart())
    async def start_cmd(message: types.Message) -> None:
        await message.answer(
            "Привет! Я юридический RAG-ассистент по праву РФ.\n"
            "Задайте вопрос, я отвечу строго по базе документов."
        )

    @dp.message()
    async def handle_question(message: types.Message) -> None:
        user_question = (message.text or "").strip()
        if not user_question:
            await message.answer("Пожалуйста, задайте вопрос.")
            return

        await message.answer("Ищу в базе и формирую ответ...")
        try:
            user_id = str(message.from_user.id) if message.from_user else "unknown"
            result = await asyncio.to_thread(rag.ask, user_question, user_id=user_id)
            answer = str(result.get("answer", "")).strip()
        except Exception as exc:
            logger.exception("Ошибка при обработке вопроса: %s", exc)
            await message.answer("Произошла ошибка при обработке запроса.")
            return

        for idx, part in enumerate(_split_message(answer), start=1):
            if idx == 1:
                await message.answer(part)
            else:
                await message.answer(f"(продолжение {idx})\n{part}")

    await dp.start_polling(bot)


def interactive_cli(rag: OtusStyleRAG) -> None:
    examples = [
        "Какие основания для увольнения работника по инициативе работодателя согласно ТК РФ?",
        "Что такое неустойка и как она рассчитывается по ГК РФ?",
        "Какой порядок обжалования судебного акта в арбитражном процессе по АПК РФ?",
        "Как применяется срок исковой давности по ГК РФ?",
    ]

    while True:
        print("\n" + "=" * 72)
        print("1. Задать свой вопрос")
        print("2. Использовать пример")
        print("3. Выход")
        choice = input("Выберите (1-3): ").strip()

        if choice == "1":
            q = input("Введите юридический вопрос: ").strip()
            if q:
                print_result(rag.ask(q))
        elif choice == "2":
            for i, ex in enumerate(examples, start=1):
                print(f"{i}. {ex}")
            raw = input(f"Выберите (1-{len(examples)}): ").strip()
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(examples):
                    print_result(rag.ask(examples[idx]))
                else:
                    print("Некорректный номер")
            except ValueError:
                print("Некорректный ввод")
        elif choice == "3":
            break
        else:
            print("Некорректный выбор")


def main() -> None:
    check_runtime_dependencies()

    parser = argparse.ArgumentParser(description="RAG по архитектуре статьи OTUS (multi-query + weighted ensemble RRF)")
    parser.add_argument("--index-dir", default="faiss_indexes")
    parser.add_argument("--index-name", default="law_db")
    parser.add_argument("--data-dir", default="NPA3001")
    parser.add_argument("--prompt-path", default="prompts/legal_student_prompt_v2.txt")
    parser.add_argument("--query", default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--llm-query-expansion", action="store_true")
    parser.add_argument("--tg-bot", action="store_true")
    args = parser.parse_args()

    rag = OtusStyleRAG(
        index_dir=Path(args.index_dir),
        index_name=args.index_name,
        data_dir=Path(args.data_dir),
        use_llm_query_expansion=args.llm_query_expansion,
        prompt_path=Path(args.prompt_path),
    )

    if args.tg_bot:
        asyncio.run(run_telegram_bot(rag))
    elif args.query:
        print_result(rag.ask(args.query, top_k=args.top_k))
    else:
        interactive_cli(rag)


if __name__ == "__main__":
    main()
