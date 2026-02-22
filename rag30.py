#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import re
import sys
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
    from sentence_transformers import SentenceTransformer
except Exception as exc:  # pragma: no cover
    IMPORT_ERRORS["sentence-transformers"] = exc
    SentenceTransformer = None  # type: ignore[assignment]


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

GENERIC_QUERY_TOKENS = {
    "вопрос", "основан", "согласн", "поряд", "договор", "прав", "обязан", "ответствен",
    "стат", "пункт", "российск", "федерац", "кодекс", "работник", "работодател",
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
Ты — цифровой юридический ассистент для студентов юридических вузов России.

Роль:
- Помогай в обучении: объясняй нормы права, разбирай правовые задачи, решай тестовые вопросы,
  подсказывай структуру правовых алгоритмов, помогай готовиться к семинарам и экзаменам.
- Ты не заменяешь преподавателя, а объясняешь право понятно и структурно.

Принципы:
1. Источник информации:
- Основывай ответ исключительно на действующем законодательстве РФ:
  Конституция РФ, кодексы, федеральные законы, подзаконные акты, постановления Пленума ВС РФ.
- Указывай точные ссылки на статьи и по возможности давай ссылку на consultant.ru.
- Если применяется несколько норм — объясняй их взаимосвязь и логику.

2. Стиль:
- Доброжелательно, понятно, педагогично.
- Юридические термины объясняй простыми словами.
- Без излишней формализации.

3. Форматирование по типу запроса:
- Разъяснение нормы: кратко + объяснение простыми словами + ссылка на норму.
- Правовой вопрос: прямой ответ + статья/часть/пункт + краткое обоснование.
- Тест с вариантами: выбери только один правильный вариант, затем обоснование с нормой.
- Разбор задачи: квалификация/правовая оценка + соответствующие нормы.
- Алгоритм действий: пошаговый список с правовыми основаниями.

Ограничения:
- Не помогай нарушать правила учебного процесса.
- Не искажай смысл норм и не давай субъективных трактовок.
- Не советуй ничего, что нарушает законы РФ.
- Если вопрос не о праве РФ — прямо сообщи, что специализация только по праву РФ.

Важно:
- Не ссылайся на источники, которых нет в переданном контексте.
- Если данных в контексте недостаточно, прямо укажи это.
- В конце ответа обязательно добавляй фразу:
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
        "langchain-core langchain-community langchain-openai sentence-transformers faiss-cpu"
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
    for keywords in ORG_ENTITY_KEYWORDS.values():
        if all(any(t.startswith(kw) or kw.startswith(t) for t in q_tokens) for kw in keywords):
            entities.update(keywords)
            continue
        partial = sum(1 for kw in keywords if any(t.startswith(kw) or kw.startswith(t) for t in q_tokens))
        if partial >= max(1, len(keywords) - 1):
            entities.update(keywords)
    return entities


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


def tokenize(text: str) -> Set[str]:
    def normalize_token(token: str) -> str:
        t = token.lower().replace("ё", "е")
        if len(t) <= 3 or t.isdigit():
            return t
        for suffix in RU_SUFFIXES:
            if len(t) > len(suffix) + 2 and t.endswith(suffix):
                return t[: -len(suffix)]
        return t

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
        self.model = SentenceTransformer(model_name)
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

    def _make_metadata_filter(
        self,
        channel: str,
        law_hints: Set[str],
        query_tokens: Optional[Set[str]] = None,
        allow_federal_supplement: bool = False,
    ) -> Callable[[Dict[str, Any]], bool]:
        def _filter(metadata: Dict[str, Any]) -> bool:
            md = metadata or {}
            if channel == "law":
                if not is_law_metadata(md):
                    return False
            else:
                if not is_practice_metadata(md):
                    return False

            if not law_hints:
                return True

            canonical_codes = canonical_law_codes_from_metadata(md)
            if canonical_codes.intersection(law_hints):
                return True

            law_id = normalize_law_code(str(md.get("law_id", "")))
            title = normalize_text(str(md.get("source_title", "")))
            hierarchy = normalize_text(str(md.get("hierarchy_str", "")))
            combined = f"{law_id} {title} {hierarchy}"

            for hint in law_hints:
                if hint in law_id:
                    return True
                for pattern in LAW_HINT_TITLE_PATTERNS.get(hint, ()):
                    if re.search(pattern, combined):
                        return True

            if allow_federal_supplement and channel == "law":
                source_type = normalize_text(str(md.get("source_type", "")))
                if "федеральный закон" in source_type or "федеральный конституционный закон" in source_type:
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

    @staticmethod
    def _rrf_merge(lists_of_docs: List[List[Document]], weight: float = 1.0) -> List[Document]:
        scores: Dict[str, float] = {}
        docs: Dict[str, Document] = {}

        for docs_list in lists_of_docs:
            for rank, doc in enumerate(docs_list, start=1):
                doc_id = stable_doc_id(doc)
                scores[doc_id] = scores.get(doc_id, 0.0) + weight / (RRF_K + rank)
                docs[doc_id] = doc

        ranked_ids = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
        return [docs[i] for i in ranked_ids]

    def _weighted_channel_fusion(
        self,
        law_docs: List[Document],
        practice_docs: List[Document],
        law_weight: float,
        practice_weight: float,
    ) -> List[Document]:
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for rank, doc in enumerate(law_docs, start=1):
            doc_id = stable_doc_id(doc)
            scores[doc_id] = scores.get(doc_id, 0.0) + law_weight / (RRF_K + rank)
            doc_map[doc_id] = doc

        for rank, doc in enumerate(practice_docs, start=1):
            doc_id = stable_doc_id(doc)
            scores[doc_id] = scores.get(doc_id, 0.0) + practice_weight / (RRF_K + rank)
            doc_map[doc_id] = doc

        ranked = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
        return [doc_map[i] for i in ranked]

    def retrieve(self, question: str, top_k: int = 8) -> Tuple[List[Document], RetrievalDebug]:
        explicit_law_hints = extract_law_hints(question)
        inferred_law_hints = infer_law_hints(question) if not explicit_law_hints else set()
        law_hints = set(explicit_law_hints or inferred_law_hints)  # for scoring/debug

        # Жесткий фильтр по кодексу используем только если кодекс явно указан в вопросе.
        retrieval_law_hints = set(explicit_law_hints)
        article_hints = extract_article_hints(question)
        variants = self._generate_query_variants(question, max_variants=4, law_hints=law_hints)
        query_tokens = tokenize(question)
        entity_tokens = extract_entity_tokens(question)
        focus_tokens = {t for t in query_tokens if len(t) >= 5 and t not in GENERIC_QUERY_TOKENS and t not in RU_STOPWORDS}
        practice_intent = self._is_practice_intent(question)
        federal_law_intent = has_federal_law_intent(question)
        allow_federal_supplement = bool(explicit_law_hints) or federal_law_intent

        law_runs: List[List[Document]] = []
        practice_runs: List[List[Document]] = []

        for variant in variants:
            law_runs.append(
                self._mmr_search(
                    variant,
                    channel="law",
                    law_hints=retrieval_law_hints,
                    query_tokens=query_tokens,
                    allow_federal_supplement=allow_federal_supplement,
                )
            )
            if practice_intent:
                practice_runs.append(
                    self._mmr_search(
                        variant,
                        channel="practice",
                        law_hints=retrieval_law_hints,
                        query_tokens=query_tokens,
                        allow_federal_supplement=allow_federal_supplement,
                    )
                )

        lexical_law = self._law_guided_lexical_search(
            query_tokens=query_tokens,
            focus_tokens=focus_tokens,
            law_hints=law_hints,
            per_law=max(50, top_k * 10),
        )
        if lexical_law:
            law_runs.append(lexical_law)

        law_ranked = self._rrf_merge(law_runs, weight=1.0)
        practice_ranked = self._rrf_merge(practice_runs, weight=1.0)
        if practice_intent:
            law_weight = LAW_WEIGHT
            practice_weight = PRACTICE_WEIGHT
        else:
            law_weight = 0.92
            practice_weight = 0.08

        fused = self._weighted_channel_fusion(
            law_ranked,
            practice_ranked,
            law_weight=law_weight,
            practice_weight=practice_weight,
        )

        # Повышаем статьи и пункты, которые ближе к теме запроса.
        rescored: List[Tuple[float, int, Document]] = []
        for idx, doc in enumerate(fused):
            md = doc.metadata or {}
            header = normalize_text(f"{md.get('source_title', '')} {md.get('hierarchy_str', '')}")
            body = normalize_text(doc.page_content[:1200])
            header_tokens = tokenize(header)
            body_tokens = tokenize(body)
            score = 0.0

            if "статья" in header:
                score += 0.9
            if "пункт" in header:
                score += 0.4
            if "раздел" in header or "глава" in header or "§" in header:
                score -= 0.35

            if focus_tokens:
                h = len(focus_tokens.intersection(header_tokens))
                b = len(focus_tokens.intersection(body_tokens))
                score += 1.0 * (h / max(1, len(focus_tokens)))
                score += 0.5 * (b / max(1, len(focus_tokens)))
            if entity_tokens:
                eh = len(entity_tokens.intersection(header_tokens))
                eb = len(entity_tokens.intersection(body_tokens))
                score += 1.2 * (eh / max(1, len(entity_tokens)))
                score += 0.6 * (eb / max(1, len(entity_tokens)))

            if law_hints:
                doc_codes = canonical_law_codes_from_metadata(md)
                if doc_codes.intersection(law_hints):
                    score += 0.8
                else:
                    score -= 1.2

            rescored.append((score, -idx, doc))

        rescored.sort(reverse=True)
        fused = [doc for _, _, doc in rescored]

        # Финальный фильтр по номеру статьи (если указан в вопросе).
        if article_hints:
            preferred: List[Document] = []
            rest: List[Document] = []
            for d in fused:
                header = normalize_text(f"{(d.metadata or {}).get('hierarchy_str', '')} {(d.metadata or {}).get('source_title', '')}")
                if any(article in header for article in article_hints):
                    preferred.append(d)
                else:
                    rest.append(d)
            fused = preferred + rest

        if focus_tokens:
            filtered: List[Document] = []
            for d in fused:
                searchable = (
                    f"{(d.metadata or {}).get('source_title', '')} "
                    f"{(d.metadata or {}).get('hierarchy_str', '')} "
                    f"{d.page_content[:1600]}"
                )
                if focus_tokens.intersection(tokenize(searchable)):
                    filtered.append(d)
            if filtered:
                fused = filtered

        if entity_tokens:
            filtered: List[Document] = []
            for d in fused:
                searchable = (
                    f"{(d.metadata or {}).get('source_title', '')} "
                    f"{(d.metadata or {}).get('hierarchy_str', '')} "
                    f"{d.page_content[:1800]}"
                )
                doc_tokens = tokenize(searchable)
                if entity_tokens.intersection(doc_tokens):
                    filtered.append(d)
            if filtered:
                fused = filtered

        question_norm = normalize_text(question)
        if "инициативе работодателя" in question_norm:
            rescored: List[Tuple[float, int, Document]] = []
            for idx, doc in enumerate(fused):
                md = doc.metadata or {}
                text = normalize_text(
                    f"{md.get('source_title', '')} {md.get('hierarchy_str', '')} {doc.page_content[:1400]}"
                )
                score = 0.0
                if "инициативе работодателя" in text:
                    score += 2.0
                if "инициативе работника" in text:
                    score -= 2.0
                if "статья 81" in text:
                    score += 1.0
                rescored.append((score, -idx, doc))
            rescored.sort(reverse=True)
            fused = [doc for _, _, doc in rescored]

        final_docs = fused[:top_k]
        debug = RetrievalDebug(
            query_variants=variants,
            law_hints=sorted(law_hints),
            explicit_law_hints=sorted(explicit_law_hints),
            inferred_law_hints=sorted(inferred_law_hints),
            entity_tokens=sorted(entity_tokens),
            article_hints=sorted(article_hints),
            law_docs=len(law_ranked),
            practice_docs=len(practice_ranked),
            lexical_law_docs=len(lexical_law),
            final_docs=len(final_docs),
        )
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
            header = f"[S{i}] {title}"
            if law_id:
                header += f" ({law_id})"
            if source_type:
                header += f" | {source_type}"
            if hierarchy:
                header += f" | {hierarchy}"
            parts.append(f"{header}\n{doc.page_content}")
        return "\n\n".join(parts)

    def answer(self, question: str, docs: List[Document]) -> str:
        if not docs:
            return (
                "В базе не найдено достаточно релевантных фрагментов. "
                "Уточните вопрос и, по возможности, укажите кодекс/статью."
            )

        prompt = f"""
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
{self._build_context(docs)}

Вопрос:
{question}
"""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt),
        ]
        response = self.llm.invoke(messages)
        return str(response.content)

    def ask(self, question: str, top_k: int = 8) -> Dict[str, Any]:
        docs, debug = self.retrieve(question, top_k=top_k)
        answer = self.answer(question, docs)
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
    args = parser.parse_args()

    rag = OtusStyleRAG(
        index_dir=Path(args.index_dir),
        index_name=args.index_name,
        data_dir=Path(args.data_dir),
        use_llm_query_expansion=args.llm_query_expansion,
        prompt_path=Path(args.prompt_path),
    )

    if args.query:
        print_result(rag.ask(args.query, top_k=args.top_k))
    else:
        interactive_cli(rag)


if __name__ == "__main__":
    main()
