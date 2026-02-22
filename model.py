import requests
import json
from chromadb import PersistentClient
from dotenv import load_dotenv
import os

# === Загрузка переменных окружения (если есть .env) ===
load_dotenv()
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "uk_law")

# === Настройка клиента Chroma ===
client = PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

# === Функция для получения контекста по запросу ===
def get_context_from_query(query, top_k=5):
    results = collection.query(query_texts=[query], n_results=top_k)
    return results["documents"][0]

# === Пользовательский запрос ===
query = "Расскажи про понятие кражи в Уголовном кодексе РФ"
context_blocks = get_context_from_query(query)

# === Ограничиваем длину контекста 4000 символами ===
MAX_CONTEXT_LENGTH = 2000
final_context = ""
for block in context_blocks:
    if len(final_context) + len(block) <= MAX_CONTEXT_LENGTH:
        final_context += block + "\n---\n"
    else:
        break

# === Формируем системный prompt и запрос пользователя ===
system_prompt = (
    "Ты — юридический ассистент, который точно отвечает по Уголовному кодексу РФ. "
    "Отвечай строго на основе приведённого контекста. Если ответа нет в тексте, напиши: "
    "'В контексте нет информации по запросу.'"
)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"Контекст:\n{final_context}\n\nВопрос: {query}"}
]

# === Настройки запроса к LM Studio ===
url = "http://10.8.1.33:1234/v1/chat/completions"
headers = {"Content-Type": "application/json"}

payload = {
    "model": "google/gemma-3-12b",
    "messages": messages,
    "temperature": 0.3
}

# === Отправка запроса к LM Studio ===
try:
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    result = response.json()
    print(result["choices"][0]["message"]["content"])
except requests.exceptions.HTTPError as errh:
    print("❌ HTTP Error:", errh)
    print(response.text)
except Exception as e:
    print("❌ Другая ошибка:", e)
