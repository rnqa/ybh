import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

try:
    response = openai.models.list()
    print("✅ Ключ работает! Доступные модели:")
    for model in response.data:
        print(model.id)
except Exception as e:
    print("❌ Ошибка:", e)
