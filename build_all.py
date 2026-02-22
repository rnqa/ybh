import os
import json
import uuid
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from tqdm import tqdm

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
EMBEDDING_CACHE = "./embeddings"
OUTPUT_DIR = "./db/ALL"

def sanitize_metadata(metadata):
    def convert(value):
        if isinstance(value, list):
            return ", ".join(map(str, value))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            return value
        else:
            return str(value)
    return {k: convert(v) for k, v in metadata.items()}

def load_chunks_from_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def main():
    print("📦 Загрузка документов...")
    all_documents = []
    all_metadatas = []

    for file_name in os.listdir("."):
        if file_name.endswith(".jsonl"):
            print(f"📄 Загрузка {file_name}...")
            chunks = load_chunks_from_jsonl(file_name)
            for chunk in chunks:
                text = chunk.get("text", "")
                if not text.strip():
                    continue
                metadata = {
                    "id": chunk.get("id", str(uuid.uuid4())),
                    "source_type": chunk.get("source_type"),
                    "source_title": chunk.get("source_title"),
                    "hierarchy": chunk.get("hierarchy"),
                    "hierarchy_str": chunk.get("hierarchy_str"),
                    "law_id": chunk.get("law_id"),
                    "chunk_id": chunk.get("chunk_id"),
                    "source_url": chunk.get("source_url"),
                    "page_number": chunk.get("page_number"),
                }
                all_documents.append(text)
                all_metadatas.append(sanitize_metadata(metadata))

    if not all_documents:
        print("❌ Нет документов для обработки.")
        return

    print(f"🔹 Найдено документов: {len(all_documents)}")

    print("🧠 Создание эмбеддингов...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        cache_folder=EMBEDDING_CACHE,
    )

    print(f"💾 Сохранение в ChromaDB: {OUTPUT_DIR}")
    Chroma.from_texts(
        texts=all_documents,
        embedding=embeddings,
        metadatas=all_metadatas,
        persist_directory=OUTPUT_DIR,
    ).persist()

    print("✅ Готово: база сохранена.")

if __name__ == "__main__":
    main()
