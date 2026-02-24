#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger("build_faiss_langchain")


class STEmbeddings(Embeddings):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cuda")
        LOGGER.info("Embeddings model loaded: %s", model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0].tolist()


def load_jsonl_folder(folder: Path, min_chars: int) -> List[Document]:
    docs: List[Document] = []
    jsonl_files = list(folder.glob("*.jsonl"))
    LOGGER.info("JSONL files: %s", len(jsonl_files))

    for fp in jsonl_files:
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue

                text = (item.get("text") or item.get("content") or "").strip()
                if len(text) < min_chars:
                    continue

                md: Dict[str, Any] = {
                    "id": item.get("id", ""),
                    "source_type": item.get("source_type", ""),
                    "source_title": item.get("source_title", ""),
                    "hierarchy": item.get("hierarchy", []),
                    "hierarchy_str": item.get("hierarchy_str", ""),
                    "law_id": item.get("law_id", ""),
                    "chunk_id": item.get("chunk_id", ""),
                    "source_url": item.get("source_url", ""),
                }
                docs.append(Document(page_content=text, metadata=md))

    LOGGER.info("Loaded documents: %s", len(docs))
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LangChain FAISS index for rag30.py")
    parser.add_argument("--data-dir", default="NPA3001")
    parser.add_argument("--index-dir", default="faiss_indexes")
    parser.add_argument("--index-name", default="law_db")
    parser.add_argument("--embedding-model", default="cointegrated/LaBSE-en-ru")
    parser.add_argument("--min-chars", type=int, default=40)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Data dir not found: {data_dir}")

    docs = load_jsonl_folder(data_dir, min_chars=args.min_chars)
    if not docs:
        raise SystemExit("No documents found to index.")

    embeddings = STEmbeddings(args.embedding_model)
    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Building FAISS index: %s", args.index_name)
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(folder_path=str(index_dir), index_name=args.index_name)
    LOGGER.info("Saved index to %s", index_dir / f"{args.index_name}.faiss")


if __name__ == "__main__":
    main()
