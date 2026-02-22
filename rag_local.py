import asyncio
import logging
import sys
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import OpenAI

from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart

from sentence_transformers import CrossEncoder


# ====================== КЛАСС ЮРИДИЧЕСКОГО АССИСТЕНТА С R2R ======================
class LegalAssistant:
    def __init__(self):
        self.embedding_dimension = 768
        self._init_embeddings()
        self._init_vector_db()
        self._init_llm()
        self._init_prompt_template()
        self._init_reranker()  # Инициализация CrossEncoder

    def _init_embeddings(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logging.info("Модель эмбеддингов загружена")

    def _init_vector_db(self):
        if not os.path.exists("./db/ALL"):
            os.makedirs("./db/ALL", exist_ok=True)

        self.db = Chroma(
            persist_directory="./db/ALL",
            embedding_function=self.embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )

        self.retriever = self.db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8}
        )
        logging.info("Векторная база данных загружена")

    def _init_llm(self):
        self.llm = OpenAI(
            base_url="http://127.0.0.1:1234/v1",
            api_key="not-needed",
            model="deepseek/deepseek-r1-0528-qwen3-8b",
            temperature=0.0,
            max_tokens=8076,
            timeout=720,
            max_retries=2
        )
        logging.info("LLM модель инициализирована")

    def _init_prompt_template(self):
        self.answer_template = """Ты — опытный юрист-консультант. Используй предоставленные юридические документы чтобы дать точный и развернутый ответ на вопрос.

ДОКУМЕНТЫ ДЛЯ ОСНОВЫ ОТВЕТА:
{context}

ВОПРОС: {question}

СФОРМУЛИРУЙ КАЧЕСТВЕННЫЙ ОТВЕТ:
- Начни с прямого ответа на вопрос
- Используй конкретные статьи, положения и нормы из документов
- Объясни значение терминов и процедур
- Структурируй ответ логически
- Сохраняй академическую точность и давай ссылки на источники

ОТВЕТ:"""
        logging.info("Шаблон подсказки инициализирован")

    def _init_reranker(self):
        logging.info("Загружаем CrossEncoder reranker...")
        self.reranker = CrossEncoder("BAAI/bge-reranker-large")
        logging.info("Reranker загружен")

    # ------------------ Форматирование документов ------------------
    def _format_docs_for_context(self, docs):
        if not docs:
            return "Информация не найдена в базе данных"

        formatted = []
        for i, doc in enumerate(docs):
            content = doc.page_content.strip()
            source_info = ""
            if doc.metadata.get('source_title'):
                source_info = f" ({doc.metadata['source_title']})"
            if doc.metadata.get('hierarchy'):
                source_info += f" - {doc.metadata['hierarchy']}"

            formatted.append(f"[Документ {i + 1}{source_info}]:\n{content}")

        return "\n\n" + "=" * 60 + "\n" + "\n\n".join(formatted) + "\n" + "=" * 60

    def _format_docs_with_sources(self, docs):
        if not docs:
            return []

        sources = []
        for doc in docs:
            source_title = doc.metadata.get('source_title', 'Неизвестный источник')
            law_id = doc.metadata.get('law_id', '')
            hierarchy = doc.metadata.get('hierarchy', '')

            source_desc = source_title
            if law_id:
                source_desc += f" ({law_id})"
            if hierarchy:
                source_desc += f" - {hierarchy}"

            if source_desc not in sources:
                sources.append(source_desc)

        return sources

    def _truncate_text(self, text, max_chars=2000):
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n\n[...текст усечён...]"

    # ------------------ R2R через CrossEncoder ------------------
    def rerank_docs(self, query: str, docs, top_k: int = 3):
        if not docs:
            return [], []

        pairs = [(query, self._truncate_text(doc.page_content, 2000)) for doc in docs]
        scores = self.reranker.predict(pairs)

        scored_sorted = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        top_docs = [d for d, s in scored_sorted[:top_k]]

        logging.info("Rerank results (top 5):")
        for i, (d, s) in enumerate(scored_sorted[:5], 1):
            meta = d.metadata.get('source_title', 'N/A')
            logging.info(f"{i}. score={s:.2f} — {meta}")

        return scored_sorted, top_docs

    # ------------------ Основной метод ask ------------------
    def ask(self, question: str):
        try:
            logging.info(f"Обработка вопроса: {question}")

            docs = self.retriever.invoke(question)
            if not docs:
                logging.info("Ретривер не вернул документов")
                return "⚠️ Не найдено релевантных нормативных фрагментов в базе."

            sources = self._format_docs_with_sources(docs)
            logging.info(f"Найдено документов: {len(docs)}")
            for i, doc in enumerate(docs):
                logging.info(f"Док {i + 1}: {doc.metadata.get('source_title', 'N/A')}")

            _, top_docs = self.rerank_docs(question, docs, top_k=3)
            context = self._format_docs_for_context(top_docs)

            final_prompt = self.answer_template.format(context=context, question=question)
            try:
                raw_answer = self.llm.invoke(final_prompt)
                answer = str(raw_answer).strip()
            except Exception as e:
                logging.error(f"Ошибка при генерации ответа: {e}", exc_info=True)
                answer = "❌ Произошла ошибка при генерации ответа."

            if sources:
                sources_section = "\n\n────────────────────\n📚 Использованные источники:\n"
                for i, source in enumerate(sources, 1):
                    sources_section += f"{i}. {source}\n"
                return answer + sources_section
            else:
                return answer + "\n\n⚠️ Для более точного ответа уточните вопрос"

        except Exception as e:
            logging.error(f"Ошибка: {e}", exc_info=True)
            return "❌ Произошла техническая ошибка. Пожалуйста, попробуйте задать вопрос еще раз."


# ====================== TELEGRAM-БОТ ======================
TELEGRAM_TOKEN = "8213924445:AAHdhRBlrWxSZ3k3Ol7FVw80z2ck2woN9X4"

assistant = LegalAssistant()

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()


@dp.message(CommandStart())
async def start_cmd(message: types.Message):
    welcome_text = """
👋 Добро пожаловать в юридический ассистент!

Я помогу вам разобраться в вопросах российского законодательства.

🔍 Как задавать вопросы:
• Конкретно и четко формулируйте вопрос
• Указывайте интересующие вас законы или статьи
• Задавайте вопросы по структуре, полномочиям, процедурам

Примеры вопросов:
• Какие полномочия у прокуратуры?
• Что регулирует Уголовно-процессуальный кодекс?
• Как проходит судебное разбирательство?

Задавайте ваш вопрос ⬇️
    """
    await message.answer(welcome_text)


@dp.message()
async def handle_question(message: types.Message):
    user_question = message.text.strip()
    if not user_question:
        await message.answer("❌ Пожалуйста, задайте вопрос")
        return

    await message.answer("🔍 Анализирую законодательную базу...")

    try:
        answer = assistant.ask(user_question)

        if len(answer) > 4000:
            parts = []
            current_part = ""
            for line in answer.split('\n'):
                if len(current_part + line) < 4000:
                    current_part += line + '\n'
                else:
                    parts.append(current_part.strip())
                    current_part = line + '\n'
            if current_part:
                parts.append(current_part.strip())

            for i, part in enumerate(parts):
                if i == 0:
                    await message.answer(part)
                else:
                    await message.answer(f"(продолжение {i + 1})\n{part}")
        else:
            await message.answer(answer)

    except Exception as e:
        logging.error(f"Ошибка в боте: {e}", exc_info=True)
        await message.answer("❌ Произошла ошибка при обработке запроса.")


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("👋 Бот остановлен")
        sys.exit(0)
