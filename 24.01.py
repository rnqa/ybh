from dotenv import load_dotenv
import asyncio
import logging
import sys
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAI

from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart

load_dotenv("config.env")
# ====================== ЮРИДИЧЕСКИЙ АССИСТЕНТ ======================

class LegalAssistant:
    def __init__(self):
        self.embedding_dimension = 768
        self._init_embeddings()
        self._init_vector_db()
        self._init_llm()
        self._init_chain()

    def _init_embeddings(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        logging.info("Эмбеддинги загружены")

    def _init_vector_db(self):
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
        logging.info("Векторная база загружена")

    def _init_llm(self):
        self.llm = OpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=6000,
            timeout=60,
            max_retries=2
        )
        logging.info("OpenAI LLM инициализирован")

    def _format_docs_for_context(self, docs):
        if not docs:
            return "Информация в базе данных не найдена."

        formatted = []
        for i, doc in enumerate(docs):
            content = doc.page_content.strip()

            source_info = ""
            if doc.metadata.get("source_title"):
                source_info += f"{doc.metadata['source_title']}"

            if doc.metadata.get("hierarchy"):
                source_info += f" — {doc.metadata['hierarchy']}"

            formatted.append(
                f"[Документ {i + 1}: {source_info}]\n{content}"
            )

        return "\n\n" + "=" * 60 + "\n" + "\n\n".join(formatted) + "\n" + "=" * 60

    def _format_sources(self, docs):
        sources = []
        for doc in docs:
            title = doc.metadata.get("source_title", "Неизвестный источник")
            law_id = doc.metadata.get("law_id")
            hierarchy = doc.metadata.get("hierarchy")

            source = title
            if law_id:
                source += f" ({law_id})"
            if hierarchy:
                source += f" — {hierarchy}"

            if source not in sources:
                sources.append(source)

        return sources

    def _init_chain(self):
        template = """
Ты — цифровой юридический ассистент для студентов юридических вузов РФ.



ДОКУМЕНТЫ:
{context}

ВОПРОС:
{question}

ТРЕБОВАНИЯ К ОТВЕТУ:
- Четкий и прямой ответ
- Ссылки на статьи и нормы
- Понятные объяснения терминов
- Строгий юридический стиль
- Без домыслов и фантазии

ОТВЕТ:
"""

        prompt = ChatPromptTemplate.from_template(template)

        self.chain = (
            RunnableParallel({
                "context": self.retriever | self._format_docs_for_context,
                "question": RunnablePassthrough()
            })
            | prompt
            | self.llm
            | StrOutputParser()
        )

        logging.info("RAG-цепочка инициализирована")

    def ask(self, question: str) -> str:
        try:
            docs = self.retriever.invoke(question)
            sources = self._format_sources(docs)

            answer = self.chain.invoke(question)

            if sources:
                answer += "\n\n────────────────────\n📚 Использованные источники:\n"
                for i, src in enumerate(sources, 1):
                    answer += f"{i}. {src}\n"

            answer += "\n\nОтвет основан на законодательстве РФ и предназначен для учебных целей."

            return answer

        except Exception as e:
            logging.error("Ошибка ассистента", exc_info=True)
            return "❌ Произошла техническая ошибка. Попробуйте позже."


# ====================== TELEGRAM-БОТ ======================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN не задан")

assistant = LegalAssistant()
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()


@dp.message(CommandStart())
async def start_cmd(message: types.Message):
    text = (
        "👋 *Юридический ассистент*\n\n"
        "Я помогаю разбираться в нормах российского права.\n\n"
        "📌 Примеры вопросов:\n"
        "• Какие полномочия у прокуратуры?\n"
        "• Что регулирует ГК РФ?\n"
        "• Каков порядок судебного разбирательства?\n\n"
        "Задайте вопрос ⬇️"
    )
    await message.answer(text, parse_mode="Markdown")


@dp.message()
async def handle_question(message: types.Message):
    question = message.text.strip()
    if not question:
        await message.answer("❌ Вопрос не может быть пустым")
        return

    await message.answer("🔍 Анализирую законодательство...")

    answer = assistant.ask(question)

    if len(answer) <= 4000:
        await message.answer(answer)
    else:
        parts = []
        current = ""

        for line in answer.split("\n"):
            if len(current) + len(line) < 4000:
                current += line + "\n"
            else:
                parts.append(current)
                current = line + "\n"

        parts.append(current)

        for i, part in enumerate(parts):
            prefix = "" if i == 0 else f"(продолжение {i + 1})\n"
            await message.answer(prefix + part)


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Бот остановлен")
        sys.exit(0)
