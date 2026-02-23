#!/usr/bin/env python3
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from rag30 import (
    OtusStyleRAG,
    collect_entity_law_markers,
    extract_entity_profiles,
    normalize_text,
    stable_doc_id,
    tokenize,
)

try:
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover
    CrossEncoder = None  # type: ignore[assignment]


LOGGER = logging.getLogger("retriever_enhanced")

GENERIC_LEXICAL_TOKENS: Set[str] = {
    "правов", "основ", "деятельн", "структур", "полномоч", "орган", "государствен", "власт",
    "вопрос", "укаж", "согласн", "российск", "федерац",
}
ORG_PROFILE_MARKERS = ("правов", "основ", "полномоч", "структур", "подотчет", "формирован", "определен")


@dataclass
class RetrievedCandidate:
    doc: Any
    doc_id: str
    score: float
    source: str
    rank: int


class EnhancedLegalRetriever:
    def __init__(self, rag: OtusStyleRAG):
        self.rag = rag
        self._cross_encoder: Optional[Any] = None
        self._cross_encoder_attempted = False

    @staticmethod
    def _focus_tokens(query_tokens: Set[str]) -> Set[str]:
        return {t for t in query_tokens if t not in GENERIC_LEXICAL_TOKENS and len(t) >= 4}

    @staticmethod
    def _lexical_score(query_tokens: Set[str], doc: Any) -> float:
        if not query_tokens:
            return 0.0
        md = doc.metadata or {}
        text = f"{md.get('source_title','')} {md.get('hierarchy_str','')} {doc.page_content[:1800]}"
        doc_tokens = tokenize(text)
        return len(query_tokens.intersection(doc_tokens)) / max(1, len(query_tokens))

    @staticmethod
    def _is_org_profile_intent(query: str) -> bool:
        q = normalize_text(query)
        return any(m in q for m in ORG_PROFILE_MARKERS) and bool(extract_entity_profiles(query))

    @staticmethod
    def _source_type_bonus(doc: Any, org_profile_intent: bool, entity_law_markers: Set[str]) -> float:
        md = doc.metadata or {}
        source_type = normalize_text(str(md.get("source_type", "")))
        meta_blob = normalize_text(f"{md.get('law_id', '')} {md.get('source_title', '')} {md.get('hierarchy_str', '')}")
        bonus = 0.0
        if org_profile_intent:
            if "федеральный конституционный закон" in source_type:
                bonus += 0.50
            elif "федеральный закон" in source_type:
                bonus += 0.40
            elif "конституция" in source_type:
                bonus += 0.30
            elif "кодекс" in source_type:
                bonus -= 0.20
        if entity_law_markers and any(marker in meta_blob for marker in entity_law_markers):
            bonus += 0.55
        return bonus

    def _get_cross_encoder(self) -> Optional[Any]:
        if self._cross_encoder_attempted:
            return self._cross_encoder
        self._cross_encoder_attempted = True
        if CrossEncoder is None:
            return None
        try:
            self._cross_encoder = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", max_length=512)
            LOGGER.info("CrossEncoder enabled: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        except Exception as exc:
            LOGGER.warning("CrossEncoder unavailable, fallback to lexical rerank: %s", exc)
            self._cross_encoder = None
        return self._cross_encoder

    def _vector_candidates(self, query: str, k: int) -> List[RetrievedCandidate]:
        out: List[RetrievedCandidate] = []
        results = self.rag.vector_store.similarity_search_with_score(query, k=k)
        for rank, (doc, distance) in enumerate(results, start=1):
            score = 1.0 / (1.0 + float(distance))
            out.append(
                RetrievedCandidate(
                    doc=doc,
                    doc_id=stable_doc_id(doc),
                    score=score,
                    source="vector",
                    rank=rank,
                )
            )
        return out

    def _baseline_candidates(self, query: str, top_k: int) -> Tuple[List[RetrievedCandidate], Dict[str, Any]]:
        docs, debug = self.rag.retrieve(query, top_k=top_k)
        out = [
            RetrievedCandidate(doc=d, doc_id=stable_doc_id(d), score=1.0 / (i + 1), source="baseline", rank=i + 1)
            for i, d in enumerate(docs)
        ]
        meta = {
            "law_hints": debug.law_hints,
            "explicit_law_hints": debug.explicit_law_hints,
            "inferred_law_hints": debug.inferred_law_hints,
            "entity_tokens": debug.entity_tokens,
            "query_variants": debug.query_variants,
            "law_docs": debug.law_docs,
            "practice_docs": debug.practice_docs,
            "lexical_law_docs": debug.lexical_law_docs,
        }
        return out, meta

    def search(
        self,
        query: str,
        top_k: int = 8,
        use_hybrid: bool = True,
        use_rerank: bool = True,
        return_debug: bool = True,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        query_tokens = tokenize(query)
        focus_tokens = self._focus_tokens(query_tokens)
        active_query_tokens = focus_tokens or query_tokens
        org_profile_intent = self._is_org_profile_intent(query)
        entity_law_markers = collect_entity_law_markers(extract_entity_profiles(query))
        baseline, baseline_debug = self._baseline_candidates(query, top_k=max(top_k, 8))

        if not use_hybrid:
            docs = [c.doc for c in baseline[:top_k]]
            return docs, {"pipeline": "baseline", "baseline_debug": baseline_debug}

        vector = self._vector_candidates(query, k=max(60, top_k * 8))
        fused: Dict[str, Tuple[Any, float]] = {}
        for c in baseline:
            source_bonus = self._source_type_bonus(c.doc, org_profile_intent, entity_law_markers)
            fused[c.doc_id] = (
                c.doc,
                fused.get(c.doc_id, (c.doc, 0.0))[1] + 0.55 / (60 + c.rank) + 0.45 * c.score + 0.15 * source_bonus,
            )
        for c in vector:
            lex = self._lexical_score(active_query_tokens, c.doc)
            source_bonus = self._source_type_bonus(c.doc, org_profile_intent, entity_law_markers)
            add = 0.45 / (60 + c.rank) + 0.35 * c.score + 0.20 * lex + 0.20 * source_bonus
            if c.doc_id in fused:
                doc, prev = fused[c.doc_id]
                fused[c.doc_id] = (doc, prev + add)
            else:
                fused[c.doc_id] = (c.doc, add)

        ranked = sorted(fused.values(), key=lambda x: x[1], reverse=True)
        docs = [d for d, _ in ranked]
        reranker_model = "none"

        if use_rerank:
            candidate_limit = max(top_k * 6, 50)
            candidates = docs[:candidate_limit]
            ce_model = self._get_cross_encoder()
            if ce_model is not None:
                try:
                    pairs = []
                    for d in candidates:
                        md = d.metadata or {}
                        text = f"{md.get('source_title','')} {md.get('hierarchy_str','')} {d.page_content[:1400]}"
                        pairs.append((query, text))
                    ce_scores = ce_model.predict(pairs, batch_size=16, show_progress_bar=False)
                    rescored: List[Tuple[float, Any]] = []
                    for i, d in enumerate(candidates):
                        ce = float(ce_scores[i])
                        lex = self._lexical_score(active_query_tokens, d)
                        source_bonus = self._source_type_bonus(d, org_profile_intent, entity_law_markers)
                        rescored.append((0.65 * ce + 0.25 * lex + 0.10 * source_bonus, d))
                    rescored.sort(key=lambda x: x[0], reverse=True)
                    docs = [d for _, d in rescored] + docs[candidate_limit:]
                    reranker_model = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
                except Exception as exc:
                    LOGGER.warning("CrossEncoder rerank failed, fallback lexical: %s", exc)
                    ce_model = None
            if ce_model is None:
                rescored = []
                for i, d in enumerate(candidates):
                    lex = self._lexical_score(active_query_tokens, d)
                    source_bonus = self._source_type_bonus(d, org_profile_intent, entity_law_markers)
                    rescored.append((0.68 * lex + 0.20 * source_bonus + 0.12 * (1.0 / (1.0 + i)), d))
                rescored.sort(key=lambda x: x[0], reverse=True)
                docs = [d for _, d in rescored] + docs[candidate_limit:]
                reranker_model = "lexical_source_prior"

        final_docs = docs[:top_k]
        debug = {
            "pipeline": "baseline+hybrid+rerank" if use_rerank else "baseline+hybrid",
            "baseline_debug": baseline_debug,
            "baseline_count": len(baseline),
            "vector_count": len(vector),
            "fused_count": len(fused),
            "final_count": len(final_docs),
            "reranker_model": reranker_model,
            "org_profile_intent": org_profile_intent,
            "focus_token_count": len(active_query_tokens),
        }
        if return_debug:
            return final_docs, debug
        return final_docs, {}
