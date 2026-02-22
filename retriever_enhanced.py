#!/usr/bin/env python3
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from rag30 import OtusStyleRAG, stable_doc_id, tokenize


LOGGER = logging.getLogger("retriever_enhanced")


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

    @staticmethod
    def _lexical_score(query_tokens: Set[str], doc: Any) -> float:
        if not query_tokens:
            return 0.0
        md = doc.metadata or {}
        text = f"{md.get('source_title','')} {md.get('hierarchy_str','')} {doc.page_content[:1800]}"
        doc_tokens = tokenize(text)
        return len(query_tokens.intersection(doc_tokens)) / max(1, len(query_tokens))

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
        baseline, baseline_debug = self._baseline_candidates(query, top_k=max(top_k, 8))

        if not use_hybrid:
            docs = [c.doc for c in baseline[:top_k]]
            return docs, {"pipeline": "baseline", "baseline_debug": baseline_debug}

        vector = self._vector_candidates(query, k=max(60, top_k * 8))
        fused: Dict[str, Tuple[Any, float]] = {}
        for c in baseline:
            fused[c.doc_id] = (c.doc, fused.get(c.doc_id, (c.doc, 0.0))[1] + 0.55 / (60 + c.rank) + 0.45 * c.score)
        for c in vector:
            lex = self._lexical_score(query_tokens, c.doc)
            add = 0.45 / (60 + c.rank) + 0.35 * c.score + 0.20 * lex
            if c.doc_id in fused:
                doc, prev = fused[c.doc_id]
                fused[c.doc_id] = (doc, prev + add)
            else:
                fused[c.doc_id] = (c.doc, add)

        ranked = sorted(fused.values(), key=lambda x: x[1], reverse=True)
        docs = [d for d, _ in ranked]

        if use_rerank:
            rescored: List[Tuple[float, Any]] = []
            for i, d in enumerate(docs[: max(top_k * 5, 20)]):
                lex = self._lexical_score(query_tokens, d)
                rescored.append((0.7 * lex + 0.3 * (1.0 / (1.0 + i)), d))
            rescored.sort(key=lambda x: x[0], reverse=True)
            docs = [d for _, d in rescored] + docs[max(top_k * 5, 20) :]

        final_docs = docs[:top_k]
        debug = {
            "pipeline": "baseline+hybrid+rerank" if use_rerank else "baseline+hybrid",
            "baseline_debug": baseline_debug,
            "baseline_count": len(baseline),
            "vector_count": len(vector),
            "fused_count": len(fused),
            "final_count": len(final_docs),
        }
        if return_debug:
            return final_docs, debug
        return final_docs, {}

