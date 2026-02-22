import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

# === Параметры ===
JSONL_FOLDER = Path("NPA3001")  # Папка с JSONL файлами
EMBEDDING_MODEL_NAME = "cointegrated/LaBSE-en-ru"
OUTPUT_DIR = Path("faiss_indexes")
OUTPUT_DIR.mkdir(exist_ok=True)

# === Загружаем модель эмбеддингов ===
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

def load_chunks_from_jsonl_folder(folder_path):
    all_texts = []
    all_metadatas = []

    jsonl_files = list(folder_path.glob("*.jsonl"))
    print(f"[INFO] Найдено {len(jsonl_files)} JSONL файлов в {folder_path}")

    for file_path in jsonl_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                c = json.loads(line)

                norm_text = c.get("text", "").strip()
                if len(norm_text) < 20:  # фильтр мусорных чанков
                    continue

                all_texts.append(norm_text)

                metadata = {
                    "id": c.get("id"),
                    "source_type": c.get("source_type"),
                    "source_title": c.get("source_title"),
                    "hierarchy": c.get("hierarchy"),
                    "hierarchy_str": c.get("hierarchy_str"),
                    "law_id": c.get("law_id"),
                    "chunk_id": c.get("chunk_id"),
                    "source_url": c.get("source_url")
                }
                all_metadatas.append(metadata)

    print(f"[INFO] Загружено {len(all_texts)} чанков из папки {folder_path}")
    return all_texts, all_metadatas

def create_faiss_index(texts, metadatas, index_name):
    print(f"[INFO] Векторизация {len(texts)} чанков для {index_name}...")
    vectors = embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    vectors = vectors.astype("float32")  # FAISS требует float32

    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    # Сохраняем индекс
    index_path = OUTPUT_DIR / f"{index_name}.faiss"
    faiss.write_index(index, str(index_path))
    print(f"[INFO] FAISS индекс сохранён: {index_path}")

    # Сохраняем метаданные
    metadata_path = OUTPUT_DIR / f"{index_name}_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadatas, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Метаданные сохранены: {metadata_path}")

    return index, metadatas

# === Индексация всех JSONL файлов ===
texts, metadatas = load_chunks_from_jsonl_folder(JSONL_FOLDER)
index, metadatas = create_faiss_index(texts, metadatas, "law_db")
