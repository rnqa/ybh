import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import chromadb
from chromadb.config import Settings

# === Конфигурация ===
DATA_PATH = "uk_rf.jsonl"  # путь к твоему .jsonl файлу
CHROMA_DB_PATH = "E:/HelperYoristBot/chroma_db"  # папка для базы

# === Инициализация клиента ChromaDB ===
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# === Имя коллекции ===
collection_name = "uk_law"

# === Удаление старой коллекции, если есть ===
existing_collections = [c.name for c in client.list_collections()]
if collection_name in existing_collections:
    client.delete_collection(name=collection_name)

# === Создание новой коллекции ===
collection = client.create_collection(name=collection_name)

# === Загрузка модели эмбеддингов ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Подготовка данных ===
print("🔄 Генерация эмбеддингов...")
texts = []
metadatas = []
ids = []

id_counts = defaultdict(int)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        chunk_id = item["chunk_id"]

        # Проверка на дубликат и добавление _1, _2, ...
        if id_counts[chunk_id] > 0:
            new_id = f"{chunk_id}_{id_counts[chunk_id]}"
        else:
            new_id = chunk_id
        id_counts[chunk_id] += 1

        texts.append(item["text"])
        metadatas.append({
            "law_id": item["law_id"],
            "chunk_id": item["chunk_id"],
            "hierarchy_str": item["hierarchy_str"],
            "source_url": item.get("source_url", "")  # на случай, если source_url может отсутствовать
        })
        ids.append(new_id)

# === Создание эмбеддингов ===
embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

# === Добавление в ChromaDB по батчам ===
print("💾 Добавление в ChromaDB...")
BATCH_SIZE = 5000
for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    collection.add(
        documents=texts[i:i + BATCH_SIZE],
        embeddings=embeddings[i:i + BATCH_SIZE],
        metadatas=metadatas[i:i + BATCH_SIZE],
        ids=ids[i:i + BATCH_SIZE],
    )

print(f"✅ Индекс построен и сохранён в {CHROMA_DB_PATH}")
