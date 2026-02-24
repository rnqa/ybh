#!/usr/bin/env python3
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from rag30 import OtusStyleRAG


LOGGER = logging.getLogger("retriever_enhanced")


class EnhancedLegalRetriever:
    def __init__(self, rag: OtusStyleRAG):
        self.rag = rag

    def search(
        self,
        query: str,
        top_k: int = 8,
        use_hybrid: bool = True,
        use_rerank: bool = True,
        return_debug: bool = True,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        docs, debug = self.rag.retrieve(query, top_k=top_k, use_hybrid=use_hybrid, use_rerank=use_rerank)
        retrieval = getattr(self.rag, "last_retrieval", {}) or {}

        baseline_debug = {}
        if hasattr(debug, "__dict__"):
            baseline_debug = dict(debug.__dict__)

        pipeline = "vector"
        if use_hybrid and use_rerank:
            pipeline = "hybrid+rerank"
        elif use_hybrid:
            pipeline = "hybrid"

        debug_payload = {
            "pipeline": pipeline,
            "baseline_debug": baseline_debug,
            "baseline_count": retrieval.get("vector_hits", 0),
            "vector_count": retrieval.get("vector_hits", 0),
            "bm25_count": retrieval.get("bm25_hits", 0),
            "fused_count": retrieval.get("fused_hits", 0),
            "final_count": len(docs),
            "reranker_model": retrieval.get("reranker_model", ""),
            "query_variants": retrieval.get("query_variants", baseline_debug.get("query_variants", [])),
            "raw": retrieval,
        }

        if return_debug:
            return docs, debug_payload
        return docs, {}
