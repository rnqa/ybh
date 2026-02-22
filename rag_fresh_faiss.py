#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

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
    from sentence_transformers import CrossEncoder, SentenceTransformer
except Exception as exc:  # pragma: no cover
    IMPORT_ERRORS["sentence-transformers"] = exc
    CrossEncoder = None  # type: ignore[assignment]
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
        logging.FileHandler("rag_fresh.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("rag_fresh")


TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]{3,}")
ARTICLE_RE = re.compile(r"(?:ст\.?|статья)\s*(\d+(?:\.\d+)?)", flags=re.IGNORECASE)


RU_STOPWORDS = {
    "это", "как", "для", "или", "при", "его", "ее", "они", "она", "оно", "все", "всех", "если",
    "когда", "также", "согласно", "который", "которые", "которых", "быть", "есть", "надо", "нужно",
    "ваш", "ваша", "ваши", "какие", "какой", "какая", "какое", "такое", "где", "кто", "чем", "про",
    "после", "перед", "между", "этой", "этого", "этот", "эта", "эти", "что", "чего", "чему", "кому",
    "каких", "какому", "какой", "ли", "же", "по", "из", "на", "от", "до", "со", "под", "над", "об",
    "а", "и", "но", "не", "да", "у", "о", "к", "в", "с", "так",
}

GENERIC_QUERY_TOKENS = {
    "вопрос", "основан", "согласн", "порядк", "работник", "работодател", "договор", "инициатив",
    "российск", "федерац", "кодекс", "стат", "пункт", "прав", "обязан", "ответствен",
}

FOCUS_EXPANSIONS: Dict[str, Set[str]] = {
    "увольнен": {"расторжен", "прекращен"},
    "неустойк": {"штраф", "пен", "санкц"},
    "дтп": {"авари", "дорожн", "транспорт"},
    "налог": {"сбор", "страхов", "взнос"},
    "наслед": {"завещан", "наследован"},
}

DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


RU_SUFFIXES = (
    "иями", "ями", "ами", "иями", "иях", "иях", "ями", "ев", "ов", "ие", "ые", "ое", "ей", "ий",
    "ый", "ой", "ем", "им", "ым", "ом", "его", "ого", "ему", "ому", "их", "ых", "ую", "юю", "ая",
    "яя", "ою", "ею", "ия", "ья", "ию", "ью", "иям", "ьям", "ием", "ьем", "иях", "ьях", "ах", "ях",
    "ам", "ям", "ом", "ем", "а", "я", "ы", "и", "е", "у", "ю", "о",
)


LAW_HINT_PATTERNS: Dict[str, List[str]] = {
    "TK": [r"(?<!\w)тк(?!\w)", r"трудов"],
    "GK": [r"(?<!\w)гк(?!\w)", r"гражданск"],
    "NK": [r"(?<!\w)нк(?!\w)", r"налог"],
    "UK": [r"(?<!\w)ук(?!\w)", r"уголовн"],
    "UPK": [r"(?<!\w)упк(?!\w)", r"уголовно[- ]процесс"],
    "GPK": [r"(?<!\w)гпк(?!\w)", r"гражданско[- ]процесс"],
    "APK": [r"(?<!\w)апк(?!\w)", r"арбитражно[- ]процесс"],
    "KOAP": [r"(?<!\w)коап(?!\w)", r"административн"],
    "BK": [r"(?<!\w)бк(?!\w)", r"бюджет"],
    "JK": [r"(?<!\w)жк(?!\w)", r"жилищ"],
    "SK": [r"(?<!\w)ск(?!\w)", r"семейн"],
    "ZK": [r"(?<!\w)зк(?!\w)", r"земельн"],
}


def check_runtime_dependencies() -> None:
    if not IMPORT_ERRORS:
        return

    print("\n[Dependency error] Missing runtime packages:")
    for pkg, err in IMPORT_ERRORS.items():
        print(f"  - {pkg}: {err}")

    py = sys.executable
    print("\nInstall into the same interpreter and retry:")
    print(f"  \"{py}\" -m pip install --upgrade pip")
    print(
        f"  \"{py}\" -m pip install "
        "langchain-core langchain-community langchain-openai "
        "sentence-transformers faiss-cpu"
    )
    raise SystemExit(1)


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


def is_informative_text(text: str) -> bool:
    if not text:
        return False
    stripped = text.strip()
    if len(stripped) < 40:
        return False
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", stripped)
    return len(letters) >= 20


def normalize_token(token: str) -> str:
    t = token.lower().replace("ё", "е")
    if t.isdigit() or len(t) <= 3:
        return t
    for suffix in RU_SUFFIXES:
        if len(t) > len(suffix) + 2 and t.endswith(suffix):
            return t[: -len(suffix)]
    return t


def tokenize(text: str) -> List[str]:
    out: List[str] = []
    for raw in TOKEN_RE.findall(text):
        lowered = raw.lower().replace("ё", "е")
        if lowered in RU_STOPWORDS:
            continue
        token = normalize_token(lowered)
        if token in RU_STOPWORDS:
            continue
        out.append(token)
    return out


def build_focus_tokens(query_tokens: Set[str]) -> Tuple[Set[str], Set[str]]:
    raw_focus = {t for t in query_tokens if len(t) >= 5 and t not in GENERIC_QUERY_TOKENS}
    expanded = set(raw_focus)
    for token in raw_focus:
        expanded.update(FOCUS_EXPANSIONS.get(token, set()))
    return raw_focus, expanded


def extract_law_hints(query: str) -> Set[str]:
    q = query.lower().replace("ё", "е")
    found: Set[str] = set()
    for code, patterns in LAW_HINT_PATTERNS.items():
        if any(re.search(p, q) for p in patterns):
            found.add(code)
    return found


def extract_article_hints(query: str) -> Set[str]:
    return {m.group(1) for m in ARTICLE_RE.finditer(query)}

def normalize_text_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("ё", "е")).strip()


def stable_doc_id(doc: Document) -> str:
    metadata = doc.metadata or {}
    for key in ("chunk_id", "id"):
        value = metadata.get(key)
        if value:
            return str(value)
    base = f"{metadata.get('source_title', '')}|{metadata.get('hierarchy_str', '')}|{doc.page_content[:300]}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def clone_doc(doc: Document) -> Document:
    return Document(page_content=doc.page_content, metadata=dict(doc.metadata or {}))


@dataclass
class ScoredDoc:
    doc: Document
    doc_id: str
    score: float
    source: str
    rank: int


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


class FreshFaissRAG:
    def __init__(
        self,
        index_dir: Path = Path("faiss_indexes"),
        index_name: str = "law_db",
        model_name: str = "cointegrated/LaBSE-en-ru",
        data_dir: Path = Path("NPA3001"),
        reranker_model: Optional[str] = DEFAULT_RERANKER_MODEL,
    ):
        load_dotenv("config.env")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found in config.env")

        self.embeddings = LaBSEEmbeddings(model_name)
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
                f"Embedding dimension mismatch: FAISS index has {index_dim}, "
                f"but model {model_name} has {emb_dim}. Rebuild index with the same embedding model."
            )

        patched = self._patch_docstore_from_jsonl(data_dir)
        self.docs: List[Document] = list(self.vector_store.docstore._dict.values())
        self.law_buckets = self._build_law_buckets(self.docs)
        logger.info("Loaded docs from FAISS docstore: %s (patched from JSONL: %s)", len(self.docs), patched)

        self.reranker_model = reranker_model
        self.cross_encoder = None
        if reranker_model and CrossEncoder is not None:
            try:
                self.cross_encoder = CrossEncoder(reranker_model, max_length=512)
                logger.info("Cross-encoder reranker loaded: %s", reranker_model)
            except Exception as exc:
                logger.warning("Cross-encoder reranker disabled (%s): %s", reranker_model, exc)

        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.1,
            timeout=90,
            max_retries=2,
            api_key=api_key,
        )

    @staticmethod
    def _load_clean_chunk_map(data_dir: Path) -> Dict[str, Document]:
        if not data_dir.exists():
            logger.warning("Data dir not found: %s", data_dir)
            return {}

        chunk_map: Dict[str, Document] = {}
        files = list(data_dir.glob("*.jsonl"))
        if not files:
            logger.warning("No JSONL files in %s", data_dir)
            return {}

        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue

                    text = (item.get("text") or item.get("content") or "").strip()
                    chunk_id = item.get("chunk_id")
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

        logger.info("Loaded clean chunk map from %s: %s unique chunk_id", data_dir, len(chunk_map))
        return chunk_map

    def _patch_docstore_from_jsonl(self, data_dir: Path) -> int:
        clean_map = self._load_clean_chunk_map(data_dir)
        if not clean_map:
            return 0

        replaced = 0
        for key, old_doc in self.vector_store.docstore._dict.items():
            metadata = dict(old_doc.metadata or {})
            chunk_id = metadata.get("chunk_id")
            clean_doc = clean_map.get(chunk_id)
            if not clean_doc:
                continue

            merged_meta = dict(metadata)
            merged_meta.update(clean_doc.metadata or {})
            self.vector_store.docstore._dict[key] = Document(
                page_content=clean_doc.page_content,
                metadata=merged_meta,
            )
            replaced += 1
        return replaced

    @staticmethod
    def _build_law_buckets(docs: Iterable[Document]) -> Dict[str, List[Document]]:
        buckets: Dict[str, List[Document]] = {}
        for doc in docs:
            law_id = normalize_law_code((doc.metadata or {}).get("law_id", ""))
            if not law_id:
                continue
            buckets.setdefault(law_id, []).append(doc)
        return buckets

    @staticmethod
    def _doc_haystack(doc: Document, limit: int = 1600) -> str:
        md = doc.metadata or {}
        return (
            f"{md.get('source_title', '')} "
            f"{md.get('hierarchy_str', '')} "
            f"{doc.page_content[:limit]}"
        )

    def _semantic_search(self, query: str, k: int = 140) -> List[ScoredDoc]:
        out: List[ScoredDoc] = []
        results = self.vector_store.similarity_search_with_score(query, k=k)
        for rank, (doc, distance) in enumerate(results, start=1):
            if not is_informative_text(doc.page_content):
                continue
            score = 1.0 / (1.0 + float(distance))
            d = clone_doc(doc)
            d.metadata["semantic_distance"] = float(distance)
            d.metadata["semantic_score"] = score
            out.append(ScoredDoc(d, stable_doc_id(d), score, "semantic", rank))
        return out

    def _header_focus_search(
        self,
        focus_tokens_expanded: Set[str],
        law_hints: Set[str],
        article_hints: Set[str],
        per_law: int = 80,
    ) -> List[ScoredDoc]:
        if not law_hints:
            return []

        out: List[ScoredDoc] = []
        for law in law_hints:
            docs = self.law_buckets.get(law, [])
            if not docs:
                continue

            local: List[Tuple[float, Document]] = []
            for doc in docs:
                if not is_informative_text(doc.page_content):
                    continue

                md = doc.metadata or {}
                header_text = normalize_text_for_match(
                    f"{md.get('source_title', '')} {md.get('hierarchy_str', '')}"
                )
                header_tokens = set(tokenize(header_text))
                focus_overlap = focus_tokens_expanded.intersection(header_tokens) if focus_tokens_expanded else set()
                article_hit = bool(article_hints and any(art in header_text for art in article_hints))

                if not focus_overlap and not article_hit:
                    continue

                score = 0.0
                if focus_overlap:
                    score += 1.2 * min(1.0, len(focus_overlap) / max(1, len(focus_tokens_expanded)))
                if article_hit:
                    score += 1.0
                local.append((score, doc))

            local.sort(key=lambda x: x[0], reverse=True)
            for rank, (score, doc) in enumerate(local[:per_law], start=1):
                d = clone_doc(doc)
                d.metadata["header_focus_score"] = float(score)
                out.append(ScoredDoc(d, stable_doc_id(d), float(score), "header_focus", rank))
        return out

    def _law_guided_lexical_search(
        self,
        query_tokens: Set[str],
        focus_tokens_expanded: Set[str],
        law_hints: Set[str],
        article_hints: Set[str],
        per_law: int = 100,
    ) -> List[ScoredDoc]:
        if not law_hints or not query_tokens:
            return []

        out: List[ScoredDoc] = []
        for law in law_hints:
            docs = self.law_buckets.get(law, [])
            if not docs:
                continue

            local: List[Tuple[float, Document]] = []
            for doc in docs:
                if not is_informative_text(doc.page_content):
                    continue

                haystack = self._doc_haystack(doc).lower()
                tokens = set(tokenize(haystack))
                overlap = query_tokens.intersection(tokens)
                focus_overlap = focus_tokens_expanded.intersection(tokens) if focus_tokens_expanded else set()

                md = doc.metadata or {}
                header_text = f"{md.get('source_title', '')} {md.get('hierarchy_str', '')}".lower()
                header_tokens = set(tokenize(header_text))
                header_overlap = query_tokens.intersection(header_tokens)

                article_hit = 0.0
                if article_hints and any(art in header_text for art in article_hints):
                    article_hit = 0.5

                if not overlap and not header_overlap and not article_hit:
                    continue

                score = len(overlap) / max(1, len(query_tokens))
                score += 0.35 * (len(header_overlap) / max(1, len(query_tokens)))
                score += article_hit
                if focus_overlap:
                    score += 0.45 * min(1.0, len(focus_overlap) / max(1, len(focus_tokens_expanded)))
                local.append((score, doc))

            local.sort(key=lambda x: x[0], reverse=True)
            for rank, (score, doc) in enumerate(local[:per_law], start=1):
                d = clone_doc(doc)
                d.metadata["lexical_score"] = float(score)
                out.append(ScoredDoc(d, stable_doc_id(d), float(score), "law_lexical", rank))

        return out

    @staticmethod
    def _rrf_fuse(channels: List[List[ScoredDoc]], law_hints: Set[str], top_n: int) -> List[Tuple[Document, float]]:
        k_rrf = 60.0
        fused: Dict[str, Tuple[Document, float]] = {}

        for channel in channels:
            for item in channel:
                law_id = normalize_law_code((item.doc.metadata or {}).get("law_id", ""))
                law_boost = 0.18 if law_hints and law_id in law_hints else 0.0

                add = 1.0 / (k_rrf + item.rank)
                add += law_boost
                if item.source == "law_lexical":
                    add += 0.35 * float(item.score)
                elif item.source == "semantic":
                    add += 0.18 * float(item.score)
                elif item.source == "header_focus":
                    add += 0.55 * float(item.score)

                if item.doc_id not in fused:
                    fused[item.doc_id] = (item.doc, add)
                else:
                    prev_doc, prev_score = fused[item.doc_id]
                    fused[item.doc_id] = (prev_doc, prev_score + add)

        ranked = sorted(fused.values(), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]

    def _post_rerank_scored(
        self,
        ranked: List[Tuple[Document, float]],
        query_text: str,
        query_tokens: Set[str],
        focus_tokens_raw: Set[str],
        focus_tokens_expanded: Set[str],
        law_hints: Set[str],
        article_hints: Set[str],
    ) -> List[Tuple[float, Document]]:
        rescored: List[Tuple[float, Document]] = []
        employer_initiative_query = (
            "инициативе работодателя" in query_text
            or ("инициатив" in query_tokens and "работодател" in query_tokens)
        )
        require_focus_in_header = bool(focus_tokens_raw and not article_hints)

        for idx, (doc, rrf_score) in enumerate(ranked, start=1):
            if not is_informative_text(doc.page_content):
                continue

            md = doc.metadata or {}
            law_id = normalize_law_code(md.get("law_id", ""))
            law_match = bool(law_hints and law_id in law_hints)

            if law_hints and not law_match:
                continue

            haystack = normalize_text_for_match(self._doc_haystack(doc))
            tokens = set(tokenize(haystack))
            overlap = len(query_tokens.intersection(tokens))
            overlap_ratio = overlap / max(1, len(query_tokens))
            focus_overlap = len(focus_tokens_expanded.intersection(tokens)) if focus_tokens_expanded else 0

            header_text = normalize_text_for_match(f"{md.get('source_title', '')} {md.get('hierarchy_str', '')}")
            header_tokens = set(tokenize(header_text))
            header_overlap = len(query_tokens.intersection(header_tokens))
            header_ratio = header_overlap / max(1, len(query_tokens))
            header_focus_overlap = len(focus_tokens_expanded.intersection(header_tokens)) if focus_tokens_expanded else 0

            article_hit = 0.0
            if article_hints and any(art in header_text for art in article_hints):
                article_hit = 1.0

            if overlap == 0 and header_overlap == 0 and article_hit == 0:
                continue
            if focus_tokens_raw and focus_overlap == 0:
                continue
            if require_focus_in_header and header_focus_overlap == 0 and focus_overlap < 2:
                continue

            rank_prior = 1.0 / (1.0 + idx)
            score = 0.35 * rank_prior
            score += 0.25 * rrf_score
            score += 0.35 * overlap_ratio
            score += 0.20 * header_ratio
            score += 0.30 * article_hit
            if focus_overlap:
                score += 0.40 * min(1.0, focus_overlap / max(1, len(focus_tokens_expanded)))
            if header_focus_overlap:
                score += 0.55 * min(1.0, header_focus_overlap / max(1, len(focus_tokens_expanded)))
            if law_match:
                score += 0.30
            if employer_initiative_query:
                if "инициативе работодателя" in haystack:
                    score += 0.35
                if "инициативе работника" in haystack:
                    score -= 0.45
            if score <= 0:
                continue

            rescored.append((score, doc))

        rescored.sort(key=lambda x: x[0], reverse=True)
        return rescored

    def _cross_encoder_rerank(
        self,
        query: str,
        scored_docs: List[Tuple[float, Document]],
        limit: int = 32,
    ) -> List[Tuple[float, Document]]:
        if not scored_docs:
            return []
        if self.cross_encoder is None:
            return scored_docs

        shortlist = scored_docs[:limit]
        pairs: List[Tuple[str, str]] = []
        for _, doc in shortlist:
            md = doc.metadata or {}
            candidate_text = (
                f"{md.get('source_title', '')}. "
                f"{md.get('hierarchy_str', '')}. "
                f"{doc.page_content[:1200]}"
            )
            pairs.append((query, candidate_text))

        try:
            ce_scores = self.cross_encoder.predict(pairs, batch_size=16, show_progress_bar=False)
        except Exception as exc:
            logger.warning("Cross-encoder rerank failed: %s", exc)
            return scored_docs

        reranked: List[Tuple[float, Document]] = []
        for (base_score, doc), ce in zip(shortlist, ce_scores):
            # Base score keeps stability; CE score dominates local ordering.
            combined = 0.35 * float(base_score) + 0.65 * float(ce)
            reranked.append((combined, doc))
        reranked.sort(key=lambda x: x[0], reverse=True)

        if len(scored_docs) > limit:
            reranked.extend(scored_docs[limit:])
        return reranked

    def retrieve(self, query: str, top_k: int = 8) -> Tuple[List[Document], Dict[str, object]]:
        query_text = normalize_text_for_match(query)
        query_tokens = set(tokenize(query))
        focus_tokens_raw, focus_tokens_expanded = build_focus_tokens(query_tokens)
        law_hints = extract_law_hints(query)
        article_hints = extract_article_hints(query)

        semantic = self._semantic_search(query, k=max(120, top_k * 20))
        law_lexical = self._law_guided_lexical_search(
            query_tokens=query_tokens,
            focus_tokens_expanded=focus_tokens_expanded,
            law_hints=law_hints,
            article_hints=article_hints,
            per_law=max(80, top_k * 12),
        )
        header_focus = self._header_focus_search(
            focus_tokens_expanded=focus_tokens_expanded,
            law_hints=law_hints,
            article_hints=article_hints,
            per_law=max(50, top_k * 8),
        )

        fused = self._rrf_fuse([semantic, law_lexical, header_focus], law_hints=law_hints, top_n=max(80, top_k * 10))
        scored_docs = self._post_rerank_scored(
            ranked=fused,
            query_text=query_text,
            query_tokens=query_tokens,
            focus_tokens_raw=focus_tokens_raw,
            focus_tokens_expanded=focus_tokens_expanded,
            law_hints=law_hints,
            article_hints=article_hints,
        )
        scored_docs = self._cross_encoder_rerank(query=query, scored_docs=scored_docs, limit=max(24, top_k * 4))
        final_docs = [doc for _, doc in scored_docs[:top_k]]

        if not final_docs:
            # Last-resort fallback: keep first semantic docs that at least mention hint law code.
            for item in semantic:
                law_id = normalize_law_code((item.doc.metadata or {}).get("law_id", ""))
                if law_hints and law_id not in law_hints:
                    continue
                final_docs.append(item.doc)
                if len(final_docs) >= top_k:
                    break

        info = {
            "law_hints": sorted(law_hints),
            "article_hints": sorted(article_hints),
            "query_tokens": sorted(list(query_tokens))[:14],
            "focus_tokens": sorted(list(focus_tokens_raw)),
            "semantic_candidates": len(semantic),
            "law_lexical_candidates": len(law_lexical),
            "header_focus_candidates": len(header_focus),
            "fused_candidates": len(fused),
            "cross_encoder_enabled": bool(self.cross_encoder is not None),
            "final_docs": len(final_docs),
        }
        return final_docs, info

    @staticmethod
    def _build_context(docs: List[Document]) -> str:
        parts: List[str] = []
        for i, doc in enumerate(docs, start=1):
            md = doc.metadata or {}
            title = md.get("source_title", "Unknown source")
            law_id = md.get("law_id", "")
            hierarchy = md.get("hierarchy_str", "")
            header = f"[S{i}] {title}"
            if law_id:
                header += f" ({law_id})"
            if hierarchy:
                header += f" | {hierarchy}"
            parts.append(f"{header}\n{doc.page_content}")
        return "\n\n".join(parts)

    def answer(self, query: str, docs: List[Document]) -> str:
        if not docs:
            return (
                "Не найдено релевантных фрагментов в FAISS базе. "
                "Уточните вопрос и укажите кодекс (например: ТК РФ, ГК РФ, НК РФ)."
            )

        context = self._build_context(docs)
        prompt = f"""
Ты юридический ассистент по законодательству РФ.
Отвечай только по контексту ниже. Не придумывай нормы.
Если контекста недостаточно, прямо скажи об этом.

Правила ответа:
1. Краткий вывод по сути вопроса.
2. Нормы/основания с точными ссылками на источники [S1], [S2], ...
3. Если в источниках есть пробелы, явно укажи ограничения.

Контекст:
{context}

Вопрос:
{query}
"""
        messages = [
            SystemMessage(content="Ты точный юридический ассистент."),
            HumanMessage(content=prompt),
        ]
        response = self.llm.invoke(messages)
        return str(response.content)

    def ask(self, query: str, top_k: int = 8) -> Dict[str, object]:
        docs, debug = self.retrieve(query, top_k=top_k)
        answer = self.answer(query, docs)
        sources = []
        for i, doc in enumerate(docs, start=1):
            md = doc.metadata or {}
            sources.append(
                {
                    "id": i,
                    "title": md.get("source_title", ""),
                    "law_id": md.get("law_id", ""),
                    "hierarchy": md.get("hierarchy_str", ""),
                    "preview": (doc.page_content[:220] + "...") if len(doc.page_content) > 220 else doc.page_content,
                }
            )
        return {"query": query, "answer": answer, "sources": sources, "debug": debug}


def print_result(result: Dict[str, object]) -> None:
    print("\n" + "=" * 72)
    print("ВОПРОС:")
    print(result["query"])
    print("=" * 72)
    print("DEBUG:", result["debug"])
    print("\nОТВЕТ:\n")
    print(result["answer"])
    print("\nИСТОЧНИКИ:")
    for src in result["sources"]:
        print(f"- [{src['id']}] {src['title']} ({src['law_id']})")
        if src["hierarchy"]:
            print(f"  {src['hierarchy']}")


def interactive_cli(rag: FreshFaissRAG) -> None:
    examples = [
        "Какие основания для увольнения работника по инициативе работодателя согласно ТК РФ?",
        "Что такое неустойка и как она рассчитывается по ГК РФ?",
        "Какие налоги должен платить ИП на УСН по НК РФ?",
        "Какая ответственность за нарушение ПДД по КоАП РФ?",
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
            num = input(f"Выберите (1-{len(examples)}): ").strip()
            try:
                idx = int(num) - 1
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

    parser = argparse.ArgumentParser(description="Fresh RAG over existing FAISS law database")
    parser.add_argument("--index-dir", default="faiss_indexes")
    parser.add_argument("--index-name", default="law_db")
    parser.add_argument("--data-dir", default="NPA3001")
    parser.add_argument("--query", default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    args = parser.parse_args()

    reranker_model = None if str(args.reranker_model).lower() in {"", "none", "off", "false"} else args.reranker_model
    rag = FreshFaissRAG(
        index_dir=Path(args.index_dir),
        index_name=args.index_name,
        data_dir=Path(args.data_dir),
        reranker_model=reranker_model,
    )
    if args.query:
        print_result(rag.ask(args.query, top_k=args.top_k))
    else:
        interactive_cli(rag)


if __name__ == "__main__":
    main()
